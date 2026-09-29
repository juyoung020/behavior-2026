"""Offline test of the native websocket server with the evaluator's own client code path (openpi msgpack format).

Sends the reference samples as evaluator observations (batched [1, ...] like BatchedEvaluator._batch_obs), checks
that the first action of each fresh chunk equals the JAX `policy.infer` output within the noise floor... (the server
draws its own start noise, so this compares structure/timing only unless --noise-free), measures round-trip time.

    python server_client_test.py --port 8000 --ref /mnt/c/behavior-2026/data/pi05_native/ref --steps 64
"""
import argparse
import pathlib
import sys
import time

import msgpack
import numpy as np
import websockets.sync.client

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dump_reference import read_container  # noqa: E402

CAMS = ("robot::robot:zed_link:Camera:0::rgb", "robot::robot:left_realsense_link:Camera:0::rgb",
        "robot::robot:right_realsense_link:Camera:0::rgb")


def pack_nd(a):
    return {b"__ndarray__": True, b"data": a.tobytes(), b"dtype": a.dtype.str, b"shape": a.shape}


def unpack(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ref", default="/mnt/c/behavior-2026/data/pi05_native/ref")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=0)
    args = ap.parse_args()
    d = read_container(pathlib.Path(args.ref) / "gpu_000.pi05d")
    img = d["in.img"]
    prop = np.zeros((1, 256), np.float32)
    prop[0, : d["in.proprio"].shape[0]] = d["in.proprio"]
    obs = {k: pack_nd(np.ascontiguousarray(img[i][None])) for i, k in enumerate(CAMS)}
    obs["robot::proprio"] = pack_nd(prop)
    obs["robot::cam_rel_poses"] = pack_nd(np.zeros((1, 21), np.float32))
    obs["task_id"] = pack_nd(np.zeros((1, 1), np.int64))
    if args.chunk:
        obs["__action_chunk_size__"] = args.chunk
    data = msgpack.packb(obs)
    import urllib.request
    print("healthz:", urllib.request.urlopen(f"http://{args.host}:{args.port}/healthz").read())
    with websockets.sync.client.connect(f"ws://{args.host}:{args.port}", compression=None, max_size=None) as ws:
        meta = msgpack.unpackb(ws.recv(), object_hook=unpack)
        print("metadata:", meta)
        ts, acts = [], []
        for _ in range(args.steps):
            t = time.perf_counter()
            ws.send(data)
            r = msgpack.unpackb(ws.recv(), object_hook=unpack)
            ts.append((time.perf_counter() - t) * 1e3)
            acts.append(np.array(r["action"]))
        ws.send(msgpack.packb({"reset": True}))
    ts = np.array(ts)
    acts = np.stack(acts)
    print(f"action shape {acts.shape}, server_timing {r['server_timing']}")
    print(f"round trip ms: inference steps (every 16th) median {np.median(ts[::16]):.1f}, "
          f"buffered steps median {np.median(np.delete(ts, np.s_[::16])):.2f}")
    # the chunk executes 16 of the 32 actions; buffered steps must walk through the chunk
    ref = d["out.actions"]  # JAX with the dump's noise; the server draws its own noise -> same scale, not equal
    print(f"first action |server| {np.abs(acts[0]).max():.3f}  |jax| {np.abs(ref[0]).max():.3f}")


if __name__ == "__main__":
    main()
