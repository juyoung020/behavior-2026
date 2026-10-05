"""평가기 프로세스 안 접착부: 안쪽 정책(VLA 등 act/reset 을 가진 아무 정책) 앞에 계획기 + scenemap(물체 지도·자세) 연결을 붙인다.

파이썬은 여기까지만(공식 평가기가 파이썬이라서). 하는 일은 바이트를 소켓에 넣고 결정을 안쪽 정책에 넘기는 것뿐이다.
계획·경계 감시·keyframe 정책·위치 추정·영상 변환은 전부 Rust(simlink = bagent link)가 한다.
프로토콜은 src/agent/planner/src/link.rs 머리말(머리 12 B + 몸통, 리틀 엔디언).

스텝 하나(환경 하나):
  1. 전 스텝 ACK 를 읽는다(이미 와 있다) → hold(이번 스텝이 경계일 수 있음), want(보낼 카메라)
  2. STEP = 요약(proprio 61, f32) + want 가 고른 영상(평가기 텐서 그대로: RGBA u8, 깊이 f32 m).
     평가기가 넣어 주는 cam_rel_poses 는 보내지 않는다(시뮬레이터 카메라·전역 자세 API 로 만든 값 — 규칙 해석상 안 씀).
     카메라 외부 자세는 link 가 proprio 관절값 + R1Pro 순기구학으로 만든다(src/agent/planner/src/fk.rs).
     영상은 미리 잡아 둔 버퍼로 복사(memcpy)만 하고 보내기는 뒤 스레드가 한다 → 시뮬레이터 스레드는 안 막힌다
  3. hold 면 결정 ACK 를 기다린다(시뮬레이터 시간 정지, 점수 영향 없음) → 문장·flush·단계 번호를 안쪽 정책에 넣는다
     (안쪽 정책에 prompt 속성 / reset_env(env) / set_stage(env, stage, fixed) 가 있을 때만)
  4. 안쪽 정책 act

(옛 안쪽 정책 = 네이티브 π0.5 엔진 — 10-06 지움. VLA = RecallVLA, robot-agent training/vla.)
"""
from __future__ import annotations

import json
import os
import pathlib
import queue
import socket
import struct
import sys
import threading
import time

import numpy as np

MAGIC = struct.unpack("<I", b"BLNK")[0]
T_HELLO, T_RESET, T_STEP, T_BYE = 1, 2, 3, 4
T_HELLO_ACK, T_ACK = 0x81, 0x83
F_WAIT = 1
KIND_RGBA8, KIND_RGB8, KIND_DEPTH_F32 = 0, 1, 2
ROLES = ("head", "left_wrist", "right_wrist")
_HDR = struct.Struct("<IHHI")
_STEP = struct.Struct("<QdIHHHHI")
_FRAME = struct.Struct("<BBHIII")
_ACK = struct.Struct("<QBBHI")
HERE = pathlib.Path(__file__).resolve().parent


def want_rgb(cam):
    return 1 << (2 * cam)


def want_depth(cam):
    return 1 << (2 * cam + 1)


class _Ring:
    """카메라·종류마다 보낼 버퍼 3칸. 한 칸을 다시 쓰기 전에 그 칸을 쓴 메시지가 다 나갔는지 확인한다."""

    def __init__(self, n=3):
        self.n = n
        self.bufs = {}
        self.pos = {}
        self.used_by = {}

    def take(self, key, arr, wait_sent):
        slots = self.bufs.setdefault(key, [None] * self.n)
        i = self.pos.get(key, 0)
        self.pos[key] = (i + 1) % self.n
        seq = self.used_by.get((key, i))
        if seq is not None:
            wait_sent(seq)
        b = slots[i]
        if b is None or b.shape != arr.shape or b.dtype != arr.dtype:
            b = slots[i] = np.empty(arr.shape, arr.dtype)
        np.copyto(b, arr)
        return b, (key, i)

    def mark(self, slot, seq):
        self.used_by[slot] = seq


class SimLinkClient:
    """link 연결 하나(환경 여러 개면 환경마다 hold/want 를 따로 든다)."""

    def __init__(self, addr: str, hello: dict, connect_timeout: float = 60.0, async_send: bool = True):
        host, port = addr.rsplit(":", 1)
        t0 = time.time()
        while True:
            try:
                self.sock = socket.create_connection((host, int(port)), timeout=5.0)
                break
            except OSError as e:
                if time.time() - t0 > connect_timeout:
                    raise ConnectionError(f"simlink {addr} 에 못 붙음: {e}")
                time.sleep(1.0)
        s = self.sock
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 << 20)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        s.settimeout(None)
        n = max(1, int(hello.get("num_envs", 1)))
        self.hold = [True] * n
        self.want = [0x3F] * n
        self.outstanding = []  # ACK 를 아직 안 읽은 환경 번호(보낸 순서)
        self.ring = _Ring()
        self.async_send = async_send
        self.seq_queued = 0
        self.seq_sent = 0
        self.cv = threading.Condition()
        self.err = None
        self.q = queue.Queue()
        if async_send:
            self.th = threading.Thread(target=self._worker, name="simlink-send", daemon=True)
            self.th.start()
        body = json.dumps(hello).encode()
        self._send_now([_HDR.pack(MAGIC, T_HELLO, 0, len(body)), body])
        ty, b = self._read_msg()
        if ty != T_HELLO_ACK:
            raise ConnectionError(f"HELLO_ACK 대신 {ty}")
        self.hello_ack = json.loads(b)

    # ---- 보내기 ----
    def _worker(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            seq, parts = item
            try:
                for p in parts:
                    self.sock.sendall(p)
            except OSError as e:
                self.err = e
            with self.cv:
                self.seq_sent = seq
                self.cv.notify_all()

    def _wait_sent(self, seq):
        with self.cv:
            while self.seq_sent < seq and self.err is None:
                self.cv.wait(1.0)
        if self.err is not None:
            raise self.err

    def _send_now(self, parts):
        if self.async_send:
            self._enqueue(parts)
            self._wait_sent(self.seq_queued)
        else:
            for p in parts:
                self.sock.sendall(p)

    def _enqueue(self, parts):
        self.seq_queued += 1
        if self.async_send:
            self.q.put((self.seq_queued, parts))
        else:
            for p in parts:
                self.sock.sendall(p)
            self.seq_sent = self.seq_queued  # 보내기 스레드 없음: 보낸 즉시 끝
        return self.seq_queued

    # ---- 받기 ----
    def _recv_exact(self, n):
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0
        while got < n:
            k = self.sock.recv_into(mv[got:], n - got)
            if k == 0:
                raise ConnectionError("simlink 연결 닫힘")
            got += k
        return bytes(buf)

    def _read_msg(self):
        magic, ty, _flags, ln = _HDR.unpack(self._recv_exact(_HDR.size))
        if magic != MAGIC:
            raise ConnectionError(f"magic {magic:#x}")
        return ty, self._recv_exact(ln) if ln else b""

    def _read_ack(self):
        ty, b = self._read_msg()
        if ty != T_ACK:
            raise ConnectionError(f"ACK 대신 {ty}")
        step, hold, has, want, n = _ACK.unpack_from(b)
        dec = json.loads(b[_ACK.size:_ACK.size + n]) if has else None
        return step, bool(hold), want, dec

    def drain(self, upto_env=None):
        """밀린 ACK 를 읽는다. upto_env 를 주면 그 환경 것까지만. 반환: [(env, 결정)]"""
        out = []
        while self.outstanding:
            e = self.outstanding.pop(0)
            _step, hold, want, dec = self._read_ack()
            self.hold[e] = hold
            self.want[e] = want
            if dec is not None:
                out.append((e, dec))
            if upto_env is not None and e == upto_env:
                break
        return out

    # ---- 한 스텝 ----
    def reset(self):
        self.drain()
        self._enqueue([_HDR.pack(MAGIC, T_RESET, 0, 0)])
        self.hold = [True] * len(self.hold)
        self.want = [0x3F] * len(self.want)

    def step(self, env, step, proprio, crp, frames, t_client=0.0):
        """frames: [(cam, kind, np.ndarray)] (연속 메모리). 반환: (기다렸나, [결정])"""
        wait = self.hold[env]
        heads, datas, total = [], [], 0
        slots = []
        for cam, kind, arr in frames:
            b, slot = self.ring.take((env, cam, kind), arr, self._wait_sent)
            slots.append(slot)
            h, w = arr.shape[0], arr.shape[1]
            heads.append(_FRAME.pack(cam, kind, 0, h, w, b.nbytes))
            datas.append(memoryview(b).cast("B"))
            total += b.nbytes
        p = np.ascontiguousarray(proprio, dtype=np.float32)
        c = np.ascontiguousarray(crp, dtype=np.float32)
        small = b"".join([_STEP.pack(step, t_client, F_WAIT if wait else 0, env, p.size, c.size, len(frames), 0),
                          p.tobytes(), c.tobytes(), *heads])
        ln = len(small) + total
        seq = self._enqueue([_HDR.pack(MAGIC, T_STEP, 0, ln) + small, *datas])
        for s in slots:
            self.ring.mark(s, seq)
        self.outstanding.append(env)
        decs = []
        if wait:
            decs = [d for e, d in self.drain(upto_env=env)]
        return wait, decs, ln

    def close(self):
        try:
            self.drain()
            self._send_now([_HDR.pack(MAGIC, T_BYE, 0, 0)])
        except Exception:
            pass
        if self.async_send:
            self.q.put(None)
        try:
            self.sock.close()
        except OSError:
            pass


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach()
        if getattr(x, "is_cuda", False):
            x = x.cpu()
        return x.numpy()
    return np.asarray(x)


class IntegPolicy:
    """LocalPolicy.policy 자리. inner = act/reset 을 가진 아무 정책.

    관측 키는 첫 act 에서 뒷부분으로 찾는다(로봇 이름이 설정마다 다르다)."""

    CAM_SUFFIX = ("zed_link:Camera:0::rgb", "left_realsense_link:Camera:0::rgb", "right_realsense_link:Camera:0::rgb")

    def __init__(self, inner, link: SimLinkClient | None, log_path=None, apply_prompt=True, apply_stage=True,
                 send_cam_rel_poses=False):
        self.inner = inner
        self.link = link
        self.rgb_keys = self.depth_keys = None
        self.prop_key = self.crp_key = None
        self.apply_prompt = apply_prompt
        self.apply_stage = apply_stage
        self.send_crp = send_cam_rel_poses  # 시험·비교용(평가 경로는 False)
        self.stage_warned = False
        self.step = 0
        self.log_path = log_path
        self.rows = []
        self.decisions = []
        self.t_last = None

    def _resolve_keys(self, obs):
        ks = list(obs.keys())
        self.rgb_keys = [next((k for k in ks if k.endswith(sfx)), None) for sfx in self.CAM_SUFFIX]
        self.depth_keys = [k.rsplit("::", 1)[0] + "::depth_linear" if k else None for k in self.rgb_keys]
        self.prop_key = next(k for k in ks if k.endswith("::proprio"))
        self.crp_key = next((k for k in ks if k.endswith("::cam_rel_poses")), None)
        print(f"[simlink] 관측 키: rgb {self.rgb_keys}, 깊이 {[k in obs for k in self.depth_keys if k]}, "
              f"proprio {self.prop_key}, cam_rel_poses {self.crp_key}", flush=True)

    # 결정 → 안쪽 정책
    def _apply(self, env, d):
        applied = []
        pr = d.get("prompt")
        if self.apply_prompt and pr and hasattr(self.inner, "prompt") and pr != self.inner.prompt:
            self.inner.prompt = pr
            applied.append("prompt")
        if d.get("flush") and hasattr(self.inner, "reset_env"):
            self.inner.reset_env(env)
            applied.append("flush")
        st = d.get("stage")
        if self.apply_stage and st is not None:
            if hasattr(self.inner, "set_stage"):
                self.inner.set_stage(env, int(st), fixed=int(d.get("stage_mode", 1)) == 1)
                applied.append("stage")
            elif not self.stage_warned:
                self.stage_warned = True
                print("[simlink] 안쪽 정책에 단계 입력이 없다 — 단계 번호는 기록만", flush=True)
        rec = {"env": env, "step": self.step, "applied": applied, **d}
        if self.log_path:  # 결정은 드물다: 바로 적는다(평가기가 끝에 프로세스를 바로 닫아도 남게)
            with open(pathlib.Path(self.log_path).with_name("decisions.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            self.decisions.append(rec)
        if d.get("kind") != "stage":
            print(f"[simlink] step {self.step} env {env}: {d.get('kind')} ({d.get('trigger')}) "
                  f"text={d.get('text')!r} stage={d.get('stage')} est={d.get('stage_est')} applied={applied} "
                  f"decide {d.get('decide_ms')} ms", flush=True)

    def act(self, obs):
        now = time.perf_counter()
        if self.link is not None and self.prop_key is None:
            self._resolve_keys(obs)
        if self.link is not None:
            prop = obs[self.prop_key]
            batched = prop.ndim == 2
            n_env = prop.shape[0] if batched else 1
            crp_all = obs.get(self.crp_key) if (self.crp_key and self.send_crp) else None
            for b in range(n_env):
                t0 = time.perf_counter()
                for e, d in self.link.drain(upto_env=b):
                    self._apply(e, d)
                t1 = time.perf_counter()
                want = self.link.want[b]
                frames = []
                for cam in range(3):
                    if self.rgb_keys[cam] is None:
                        continue
                    if want & want_rgb(cam) and self.rgb_keys[cam] in obs:
                        a = _np(obs[self.rgb_keys[cam]])
                        a = a[b] if batched else a
                        frames.append((cam, KIND_RGBA8 if a.shape[-1] == 4 else KIND_RGB8, np.ascontiguousarray(a)))
                    if want & want_depth(cam) and self.depth_keys[cam] in obs:
                        a = _np(obs[self.depth_keys[cam]])
                        a = a[b] if batched else a
                        frames.append((cam, KIND_DEPTH_F32, np.ascontiguousarray(a, dtype=np.float32)))
                p = _np(prop)
                p = p[b] if batched else p
                c = _np(crp_all) if crp_all is not None else np.zeros(0, np.float32)
                c = (c[b] if batched else c).reshape(-1) if c.size else c
                wait, decs, nbytes = self.link.step(b, self.step, p, c, frames, t_client=t1)
                t2 = time.perf_counter()
                for d in decs:
                    self._apply(b, d)
                self.rows.append((self.step, b, int(wait), want, len(frames), nbytes, (t1 - t0) * 1e6, (t2 - t1) * 1e6,
                                  (now - self.t_last) * 1e3 if self.t_last else float("nan")))
        self.step += 1
        self.t_last = now
        if self.log_path and self.step % 200 == 0:
            self.flush()
        return self.inner.act(obs)

    def reset(self):
        self.flush()
        if self.link is not None:
            self.link.reset()
        self.inner.reset()
        self.step = 0
        self.t_last = None

    def flush(self):
        if hasattr(self.inner, "flush"):
            self.inner.flush()
        if not self.log_path:
            return
        new = not os.path.exists(self.log_path)
        with open(self.log_path, "a", encoding="utf-8") as f:
            if new:
                f.write("step,env,hold,want,frames,bytes,drain_us,send_us,step_ms\n")
            for r in self.rows:
                f.write(",".join(f"{x:.1f}" if isinstance(x, float) else str(x) for x in r) + "\n")
        self.rows = []
        dpath = str(pathlib.Path(self.log_path).with_name("decisions.jsonl"))
        with open(dpath, "a", encoding="utf-8") as f:
            for d in self.decisions:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        self.decisions = []

    def close(self):
        self.flush()
        if self.link is not None:
            self.link.close()


def camera_specs(wrapper: str):
    """평가기 카메라 크기·K (eval_utils 상수, 래퍼 해상도에 맞춰 비례)."""
    from omnigibson.eval.utils import eval_utils as U

    K = U.CAMERA_INTRINSICS["R1Pro"]
    native = {"head": U.HEAD_RESOLUTION, "left_wrist": U.WRIST_RESOLUTION, "right_wrist": U.WRIST_RESOLUTION}
    out = []
    for role in ROLES:
        h, w = native[role] if wrapper == "rgbd" else (224, 224)
        s = w / native[role][1]
        k = K[role]
        out.append({"name": role, "w": int(w), "h": int(h),
                    "k": [float(k[0][0] * s), float(k[1][1] * s), float(k[0][2] * s), float(k[1][2] * s)]})
    return out
