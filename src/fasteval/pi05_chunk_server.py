"""π0.5 정책 서버 + 공식 '행동 묶음 재생'(--replay-action-chunk-size K) 응답. WSL openpi venv 에서 돌린다.

    cd ~/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 .venv/bin/python /mnt/c/behavior-2026/src/fasteval/pi05_chunk_server.py \\
        --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id turning_on_radio --policy.config pi05_b1k \\
        --policy.dir ~/checkpoints/pi05_turning_on_radio/pi05_turn_on_the_radio --control_mode receding_horizon \\
        --action_horizon 16 --port 8000
    평가기: run_eval_radio.ps1 -ChunkSize 16   (= 공식 인자 --replay-action-chunk-size 16)

왜 필요한가 (docs\\평가기_가속설계.md): 이 PC 에서 정책 왕복 한 번이 WSL 을 건너며 평균 28 ms 걸리고,
receding horizon 16 이면 16 스텝 중 15 스텝은 서버가 관측을 쓰지도 않고 버퍼의 다음 행동을 꺼내 줄 뿐이다.

무엇이 같은가 (결과 동일 근거)
- openpi 공식 서버(scripts/b1k/serve_b1k.py)와 같은 정책 객체(B1KPolicyWrapper, receding_horizon)를 그대로 쓴다.
- 묶음 요청이 오면 평소처럼 act() 로 첫 행동을 만들고(이때 16 스텝 경계라 추론이 돈다), 같은 버퍼에서 이어지는
  K-1 개를 꺼내 action_chunk 로 준다. 그리고 버퍼 위치와 스텝 카운터를 K-1 만큼 넘겨, 다음 요청이 공식 경로와
  같은 스텝(16 의 배수)에서 추론하게 한다. 난수(JAX key)는 추론 횟수가 같으므로 같은 순서로 쪼개진다.
- 공식 경로에서 경계 사이 스텝의 관측은 act() 안에서 크기 조정만 되고 버려진다 -> 안 보내도 결과가 같다.
  (src\\fasteval\\verify_chunk_equivalence.py 가 같은 관측열에서 두 경로의 행동을 비트 단위로 비교한다.)
- K 가 재계획 경계를 넘으면 거부한다 (공식 문서: "choose K so replay never crosses the policy's replanning boundary").

검은 화면 방어(--black_fill last, 기본): 카메라 RGB 가 전부 0 이면 그 카메라의 직전 정상 프레임으로 바꿔 정책에 넣는다
(BlackFrameFiller). --black_fill off 면 openpi 공식 서버와 똑같이 받은 그대로 쓴다.

언어: 파이썬 -- 정책이 JAX 파이썬 객체라서 붙는 얇은 접착부(버퍼 인덱싱 몇 줄)만 여기 있다. 무거운 계산은 JAX/XLA(C++/CUDA).
"""
from __future__ import annotations

import dataclasses
import logging
import socket
import time
import traceback
from copy import deepcopy

import numpy as np
import tyro
import websockets

from openpi.configs.tasks import TASK_REGISTRY
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_b1k_server as _srv
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as _config

CHUNK_KEY = "__action_chunk_size__"  # omnigibson/eval/utils/network_utils.py ACTION_CHUNK_REQUEST_KEY


class BlackFrameFiller:
    """정책 쪽 방어: 카메라 RGB 가 전부 0(검은 화면)이면 그 카메라·그 환경의 직전 정상 프레임으로 바꿔 넣는다.

    - 과거에 받은 RGB 만 쓰므로 관측 제한(RGB·depth·proprio) 안이다. 평가기·중계는 관측을 안 바꾸고, 바꾸는 것은 정책(우리 제출물)이다.
    - 에피소드 첫 프레임부터 검으면 채울 과거가 없다 -> 그대로 넘기고 센다(지어낸 영상을 넣지 않는다).
    - depth 는 바꾸지 않는다(이 PC 실측: RGB 가 검을 때도 depth 는 0 이 아니었다).
    - 묶음 재생(K=16)이면 서버는 16 스텝마다의 프레임만 보므로 채우는 프레임도 16 스텝 전 것이다.
    """

    def __init__(self, mode: str = "last"):
        self.mode = mode
        self.last = {}  # (키, 환경) -> 마지막 정상 프레임
        self.stats = {"frames": 0, "black": 0, "filled": 0, "no_history": 0}

    def reset(self):
        self.last = {}

    def __call__(self, obs: dict) -> dict:
        for key in [k for k in obs if k.endswith("::rgb")]:
            arr = np.asarray(obs[key])
            batched = arr.ndim == 4
            frames = arr if batched else arr[None]
            out = None
            for b in range(frames.shape[0]):
                self.stats["frames"] += 1
                if frames[b].any():
                    self.last[(key, b)] = frames[b].copy()
                    continue
                self.stats["black"] += 1
                if self.mode != "last":
                    continue
                prev = self.last.get((key, b))
                if prev is None:
                    self.stats["no_history"] += 1
                    continue
                if out is None:
                    out = frames.copy()
                out[b] = prev
                self.stats["filled"] += 1
            if out is not None:
                obs[key] = out if batched else out[0]
        if self.stats["black"] and self.stats["black"] % 50 == 1:
            logging.warning(f"[black-fill] 검은 RGB 프레임 누적 {self.stats}")
        return obs


class ChunkedB1KPolicy:
    """B1KPolicyWrapper 를 감싸 묶음 요청이면 (첫 행동, 묶음) 을, 아니면 (행동, None) 을 돌려준다."""

    def __init__(self, inner: B1KPolicyWrapper, black_fill: str = "last"):
        self.inner = inner
        self.filler = BlackFrameFiller(black_fill)

    def reset(self):
        logging.info(f"[black-fill] 에피소드 끝 통계 {self.filler.stats}")
        self.filler.reset()
        self.inner.reset()

    def act(self, obs: dict):
        k = int(np.asarray(obs.pop(CHUNK_KEY, 0)))
        obs = self.filler(obs)
        a0 = self.inner.act(obs)
        if k <= 1:
            return a0, None
        w = self.inner
        if w.control_mode != "receding_horizon":
            raise ValueError("행동 묶음 재생은 receding_horizon 에서만 공식 경로와 같다")
        idx = w.sequence_indices[:, 0] - 1  # a0 의 버퍼 위치 (act 가 이미 1 올렸다)
        if np.any(idx + k > w.action_horizon):
            raise ValueError(f"K={k} 가 재계획 경계를 넘는다 (위치 {idx.tolist()}, 경계 {w.action_horizon})")
        b = np.arange(w.action_buffer.shape[0])
        chunk = np.stack([w.action_buffer[b, 0, idx + j] for j in range(k)], axis=1)  # (B, K, A)
        w.sequence_indices[:, 0] += k - 1
        w.step_counter += k - 1
        a0_np = a0.cpu().numpy()
        chunk = chunk.astype(a0_np.dtype, copy=False)
        if a0_np.ndim == 1:
            chunk = chunk[0]
        if not np.array_equal(chunk[..., 0, :], a0_np):
            raise RuntimeError("묶음 첫 행동이 act() 결과와 다르다")
        return a0, chunk


class ChunkServer(_srv.WebsocketPolicyServer):
    """openpi websocket_b1k_server 의 처리 루프와 같고, 응답에 action_chunk 만 더한다."""

    async def _handler(self, websocket):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = _srv.Packer()
        await websocket.send(packer.pack(self._metadata))
        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = _srv.unpackb(await websocket.recv(), strict_map_key=False)
                if "reset" in result:
                    self._policy.reset()
                    continue
                obs = deepcopy(result)
                infer_time = time.monotonic()
                action, chunk = self._policy.act(obs)
                infer_time = time.monotonic() - infer_time
                resp = {"action": action.cpu().numpy()}
                if chunk is not None:
                    resp["action_chunk"] = chunk
                resp["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    resp["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                await websocket.send(packer.pack(resp))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logging.error(f"Error in connection from {websocket.remote_address}:\n{traceback.format_exc()}")
                await websocket.close(code=1011, reason="Internal server error")
                raise


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Args:
    """scripts/b1k/serve_b1k.py 와 같은 인자."""

    robot: str
    task: str
    policy: Checkpoint
    repo_id: str | None = None
    control_mode: str = "receding_horizon"
    action_horizon: int = 16
    port: int = 8000
    # 검은 RGB 프레임 방어: last = 직전 정상 프레임으로 바꿈, off = 공식 서버와 똑같이 그대로
    black_fill: str = "last"


def build_policy(args: Args) -> tuple[B1KPolicyWrapper, dict]:
    """serve_b1k.py main() 과 같은 순서로 정책을 만든다."""
    task_bucket, task_name = args.task.split("/")
    task_prompt = TASK_REGISTRY[task_bucket][task_name]
    config = _config.get_config(args.policy.config)
    norm_stats_repo_id = args.repo_id or args.task
    config = dataclasses.replace(
        config, data=dataclasses.replace(config.data, repo_id=norm_stats_repo_id, robot_config_name=args.robot)
    )
    policy = _policy_config.create_trained_policy(config, args.policy.dir, default_prompt=task_prompt)
    return B1KPolicyWrapper(
        policy=policy,
        robot=args.robot,
        text_prompt=task_prompt,
        control_mode=args.control_mode,
        action_horizon=args.action_horizon,
        max_len=config.model.action_horizon,
    ), policy.metadata


def main(args: Args) -> None:
    wrapper, metadata = build_policy(args)
    logging.info("Creating chunk-capable server (host: %s)", socket.gethostname())
    policy = ChunkedB1KPolicy(wrapper, black_fill=args.black_fill)
    ChunkServer(policy=policy, host="0.0.0.0", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
