"""replaysrv(Rust) 와 tools/replay_policy_server.py(파이썬)가 같은 요청열에 같은 응답·같은 관측 기록을 내는지 본다(시뮬레이터 없이).

    wsl -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/src/fasteval/replaysrv/verify.sh

- 요청열: 평가기와 같은 모양의 관측(proprio·cam_rel_poses·task_id int64·카메라 RGBA uint8, RGB-D 면 depth float32 도)을
  난수(씨앗 고정)로 만들고, 가운데 NaN 도 넣는다. 에피소드 1: 배치 1, 40 요청 / reset / 에피소드 2: 배치 2(기록 환경 수와 달라 0 번 환경 행동을 퍼뜨림), 12 요청.
- 두 서버에 같은 행동열(actions.npz)·같은 --perturb 를 주고: (1) 응답 행동 바이트, (2) 응답 msgpack 구조(키·dtype·shape),
  (3) server_log.npz 의 키·dtype·shape·값(NaN 위치 포함)이 같은지 본다. server_timing 값(시간)은 뺀다.
"""
import argparse
import functools
import sys
import time
import urllib.request

import msgpack
import numpy as np
import websockets.sync.client


def pack_data(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    return obj


def unpack_data(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    return obj


packer = msgpack.Packer(default=pack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data)


def make_obs(r, n, rgbd, step):
    head, wrist = ((720, 720), (480, 480)) if rgbd else ((224, 224), (224, 224))
    prop = r.standard_normal((n, 61)).astype(np.float32)
    if step == 7:
        prop[0, 3] = np.nan  # NaN 이 기록·비교에서 같게 다뤄지는지
    obs = {"robot::proprio": prop,
           "robot::cam_rel_poses": r.standard_normal((n, 21)).astype(np.float32),
           "task_id": np.full((n, 1), 7, np.int64)}
    for name, (h, w) in (("zed_link", head), ("left_realsense_link", wrist), ("right_realsense_link", wrist)):
        obs[f"robot::robot:{name}:Camera:0::rgb"] = r.integers(0, 255, (n, h, w, 4), dtype=np.uint8)
        if rgbd:
            obs[f"robot::robot:{name}:Camera:0::depth_linear"] = r.random((n, h, w), dtype=np.float32)
    return obs


def wait_health(port):
    for _ in range(100):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1).status == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise SystemExit(f"port {port} healthz 안 됨")


def drive(port, rgbd):
    wait_health(port)
    out = []
    c = websockets.sync.client.connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None, ping_interval=60, ping_timeout=300)
    md = unpackb(c.recv())
    out.append(("metadata", repr(md)))
    r = np.random.default_rng(1234)
    c.send(packer.pack({"reset": True}))
    for s in range(40):
        c.send(packer.pack(make_obs(r, 1, rgbd, s)))
        resp = unpackb(c.recv())
        a = resp["action"]
        out.append((f"ep1 step {s}", (sorted(resp.keys()), a.dtype.str, a.shape, a.tobytes(), sorted(resp["server_timing"].keys()))))
    c.send(packer.pack({"reset": True}))
    for s in range(12):
        c.send(packer.pack(make_obs(r, 2, rgbd, 100 + s)))
        resp = unpackb(c.recv())
        a = resp["action"]
        out.append((f"ep2 step {s}", (sorted(resp.keys()), a.dtype.str, a.shape, a.tobytes(), sorted(resp["server_timing"].keys()))))
    c.close()
    return out


def cmp_logs(pa, pb):
    za, zb = np.load(pa), np.load(pb)
    bad = []
    if sorted(za.files) != sorted(zb.files):
        bad.append(f"키 다름: {sorted(set(za.files) ^ set(zb.files))}")
    for k in sorted(set(za.files) & set(zb.files)):
        x, y = za[k], zb[k]
        if x.dtype != y.dtype or x.shape != y.shape:
            bad.append(f"{k}: dtype/shape {x.dtype}{x.shape} vs {y.dtype}{y.shape}")
        elif x.dtype.kind == "f":
            if not (np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(np.nan_to_num(x), np.nan_to_num(y))):
                bad.append(f"{k}: 값 다름")
        elif not np.array_equal(x, y):
            bad.append(f"{k}: 값 다름")
    return bad, len(za.files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-port", type=int, required=True)
    ap.add_argument("--rs-port", type=int, required=True)
    ap.add_argument("--py-log", required=True)
    ap.add_argument("--rs-log", required=True)
    ap.add_argument("--rgbd", action="store_true")
    a = ap.parse_args()
    ra = drive(a.py_port, a.rgbd)
    rb = drive(a.rs_port, a.rgbd)
    time.sleep(1.0)  # 두 서버가 연결 끝 기록을 다 쓰도록
    bad = [f"{x[0]}: 응답 다름" for x, y in zip(ra, rb) if x != y]
    if len(ra) != len(rb):
        bad.append(f"응답 수 {len(ra)} vs {len(rb)}")
    lb, nk = cmp_logs(a.py_log, a.rs_log)
    bad += lb
    tag = "RGB-D" if a.rgbd else "224 RGB"
    if bad:
        print(f"다름 ({tag}):")
        for b in bad[:20]:
            print("  " + b)
        return 1
    print(f"같음 ({tag}): 응답 {len(ra)} 개(행동 바이트·키·dtype·shape), 관측 기록 {nk} 키(값·NaN 위치·dtype·shape)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
