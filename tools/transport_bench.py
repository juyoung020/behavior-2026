"""정책 왕복(웹소켓 + msgpack)만 따로 잰다 -- 시뮬레이터 없이, 평가기 클라이언트와 같은 방식으로.

    python C:\\behavior-2026\\tools\\transport_bench.py --port 8020 [--rgbd] [--nodelay] [--n 300]

- 관측 딕셔너리는 평가기가 보내는 것과 같은 모양(배치 1): proprio(61) · 카메라 3 대 RGBA · cam_rel_poses(21) · task_id.
  --rgbd 면 공식 RGB-D 래퍼 크기(머리 720x720, 손목 480x480, depth_linear float32 포함).
- 연결은 omnigibson/eval/utils/network_utils.py 의 WebsocketClientPolicy 와 같은 인자
  (websockets.sync.client.connect, compression=None, max_size=None, ping_interval=60, ping_timeout=300).
- --nodelay: 클라이언트 소켓에 TCP_NODELAY 를 켠다 (Nagle 알고리즘 끔) -- 지연 ACK 와 엉켜 생기는 ~40 ms 대기 확인용.
- 서버는 tools\\replay_policy_server.py (재생 행동을 돌려줌, 추론 없음).
"""
import argparse
import functools
import socket
import time

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


def make_obs(rgbd):
    r = np.random.default_rng(0)
    head, wrist = ((720, 720), (480, 480)) if rgbd else ((224, 224), (224, 224))
    obs = {"robot::proprio": r.standard_normal((1, 61)).astype(np.float32),
           "robot::cam_rel_poses": r.standard_normal((1, 21)).astype(np.float32),
           "task_id": np.zeros((1, 1), np.int64)}
    for name, (h, w) in (("zed_link", head), ("left_realsense_link", wrist), ("right_realsense_link", wrist)):
        obs[f"robot::robot:{name}:Camera:0::rgb"] = r.integers(0, 255, (1, h, w, 4), dtype=np.uint8)
        if rgbd:
            obs[f"robot::robot:{name}:Camera:0::depth_linear"] = r.random((1, h, w), dtype=np.float32)
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--rgbd", action="store_true")
    ap.add_argument("--nodelay", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    packer = msgpack.Packer(default=pack_data)
    unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data)
    conn = websockets.sync.client.connect(f"ws://{a.host}:{a.port}", compression=None, max_size=None,
                                          ping_interval=60, ping_timeout=300)
    unpackb(conn.recv())
    if a.nodelay:
        conn.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    nd = conn.socket.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
    conn.send(packer.pack({"reset": True}))
    obs = make_obs(a.rgbd)
    rtt, pk = [], []
    for i in range(a.n):
        t0 = time.perf_counter()
        data = packer.pack(obs)
        t1 = time.perf_counter()
        conn.send(data)
        resp = unpackb(conn.recv())
        t2 = time.perf_counter()
        pk.append(t1 - t0)
        rtt.append(t2 - t1)
        assert "action" in resp
    conn.close()
    rtt = np.array(rtt[10:]) * 1e3
    pk = np.array(pk[10:]) * 1e3
    slow = (rtt > 30).mean() * 100
    print(f"port {a.port} {'RGBD' if a.rgbd else 'RGB224'} 요청 {len(data) / 1e6:.2f} MB  TCP_NODELAY={nd}  "
          f"pack {pk.mean():.2f} ms  왕복 평균 {rtt.mean():.2f} / 중앙 {np.median(rtt):.2f} / p95 {np.percentile(rtt, 95):.2f} ms"
          f"  30 ms 넘는 비율 {slow:.0f}%")


if __name__ == "__main__":
    main()
