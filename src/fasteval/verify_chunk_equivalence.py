"""행동 묶음 재생(pi05_chunk_server.py)이 공식 경로(매 스텝 요청)와 비트 단위로 같은 행동을 내는지 확인한다.

    cd ~/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 .venv/bin/python /mnt/c/behavior-2026/src/fasteval/verify_chunk_equivalence.py \\
        --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id turning_on_radio --policy.config pi05_b1k \\
        --policy.dir ~/checkpoints/pi05_turning_on_radio/pi05_turn_on_the_radio [--frames <trace_images.npz>]

방법 (JSBSim fdm_verify 와 같은 틀: 같은 입력을 두 경로에 넣고 출력 전부를 비교, 누적 없음)
- 정책을 한 번 올리고 JAX 난수 상태를 저장해 둔다. 경로 A(공식: 스텝마다 act) -> 난수 되돌림 -> 경로 B(묶음) 순서로 돌린다.
- 관측열: 스텝마다 다른 무작위 proprio·영상 (--frames 를 주면 실제 평가기 영상에서 뽑음). 배치 1 과 2.
- 판정: 모든 스텝의 행동이 비트 동일이면 통과(exit 0).
- 음성 대조 1: 경로 B 에서 16 스텝째 관측의 픽셀 하나를 바꾸면 그 뒤 행동이 달라져야 한다(비교가 관측 변화를 잡는지).
- 음성 대조 2(성질 확인): 경로 A 에서 경계가 아닌 스텝(5)의 관측을 바꿔도 행동은 그대로여야 한다
  (= 공식 경로도 경계 사이 관측을 안 쓴다 -> 묶음 재생이 같은 이유).
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np
import tyro

sys.path.insert(0, "/mnt/c/behavior-2026/src/fasteval")
from pi05_chunk_server import CHUNK_KEY, Args, ChunkedB1KPolicy, build_policy  # noqa: E402

CAMS = ("robot::robot:zed_link:Camera:0::rgb", "robot::robot:left_realsense_link:Camera:0::rgb",
        "robot::robot:right_realsense_link:Camera:0::rgb")


@dataclasses.dataclass
class VArgs(Args):
    frames: str = ""
    steps: int = 64


def make_obs_seq(T, B, frames_path):
    r = np.random.default_rng(1)
    real = None
    if frames_path:
        z = np.load(frames_path)
        real = {c: [z[k] for k in z.files if k.endswith(c)] for c in CAMS}
    seq = []
    for t in range(T):
        o = {"robot::proprio": r.standard_normal((B, 61)).astype(np.float32) * 0.3,
             "robot::cam_rel_poses": r.standard_normal((B, 21)).astype(np.float32),
             "task_id": np.zeros((B, 1), np.int64)}
        for c in CAMS:
            if real and real[c]:
                base = real[c][t % len(real[c])]
                o[c] = np.stack([np.roll(base, t + b, axis=1) for b in range(B)])
            else:
                o[c] = r.integers(0, 255, (B, 224, 224, 4), dtype=np.uint8)
        seq.append(o)
    return seq


def run_a(w, seq):
    w.reset()
    return np.stack([w.act({k: v.copy() for k, v in o.items()}).cpu().numpy() for o in seq])


def run_b(cw, seq, k):
    cw.reset()
    out = []
    for t in range(0, len(seq), k):
        o = {kk: v.copy() for kk, v in seq[t].items()}
        o[CHUNK_KEY] = np.int64(k)
        a0, chunk = cw.act(o)
        out.extend(np.moveaxis(chunk, -2, 0))  # (K, B, A)
    return np.stack(out)


def main(a: VArgs) -> int:
    w, _ = build_policy(a)
    pol = w.policy
    cw = ChunkedB1KPolicy(w, black_fill="off")  # 등가 검증은 공식 서버와 같은 입력 그대로
    k = a.action_horizon
    bad = 0
    for B in (1, 2):
        seq = make_obs_seq(a.steps, B, a.frames)
        rng0 = pol._rng
        A = run_a(w, seq)
        pol._rng = rng0
        Bm = run_b(cw, seq, k)
        same = np.array_equal(A, Bm)
        d = float(np.abs(A - Bm).max())
        print(f"[배치 {B}] 공식 경로 vs 묶음 재생: 행동 {A.shape} {'비트 동일' if same else '다름'} (최대|차이| {d:.3e})")
        bad += not same
        # 음성 대조 1: 16 스텝째(추론하는 스텝) 관측 한 픽셀 변경 -> 달라져야 정상
        seq2 = [dict(o) for o in seq]
        seq2[k] = {kk: v.copy() for kk, v in seq[k].items()}
        seq2[k][CAMS[0]][0, 100, 100, 0] ^= 0xFF
        pol._rng = rng0
        B2 = run_b(cw, seq2, k)
        caught = not np.array_equal(A[k:], B2[k:]) and np.array_equal(A[:k], B2[:k])
        print(f"  음성 대조 1 (추론 스텝 {k} 픽셀 하나 변경): {'잡았다 (정상)' if caught else '못 잡았다 -- 비교가 무디다'}")
        bad += not caught
        # 성질 확인: 경계 사이 스텝(5) 관측 변경 -> 공식 경로 행동 그대로여야
        seq3 = [dict(o) for o in seq]
        seq3[5] = {kk: v.copy() for kk, v in seq[5].items()}
        seq3[5][CAMS[0]][:] = 0
        seq3[5]["robot::proprio"][:] += 1.0
        pol._rng = rng0
        A3 = run_a(w, seq3)
        ignored = np.array_equal(A, A3)
        print(f"  성질 확인 (경계 사이 스텝 5 관측 변경 -> 공식 경로 행동 불변): {'맞다' if ignored else '아니다 -- 묶음 재생이 달라질 수 있다'}")
        bad += not ignored
        pol._rng = rng0
    print("통과" if not bad else f"실패 {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(VArgs)))
