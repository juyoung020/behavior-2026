"""Smoke test of the Python module in the evaluator's Python (Windows, conda env `behavior`):
engine load -> infer on reference samples -> compare with JAX `policy.infer` output -> timing & memory.

    python smoke_test.py --weights C:/behavior-2026/data/pi05_native/pi05_radio.pi05w --ref C:/behavior-2026/data/pi05_native/ref
"""
import argparse
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "build_win"))
import _pi05native as N  # noqa: E402


def read_container(path):
    raw = pathlib.Path(path).read_bytes()
    assert raw[:8] == b"PI05W\0\0\1"
    mlen = int(np.frombuffer(raw[8:16], np.uint64)[0])
    doff = int(np.frombuffer(raw[16:24], np.uint64)[0])
    dt = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "u8": np.uint8}
    out = {}
    for line in raw[24:24 + mlen].decode().splitlines():
        f = line.split()
        if f and f[0] == "t" and f[2] in dt and f[1].split(".")[0] in ("in", "out", "host"):
            off, nb, nd = int(f[3]), int(f[4]), int(f[5])
            shape = tuple(int(x) for x in f[6:6 + nd])
            out[f[1]] = np.frombuffer(raw, dt[f[2]], count=nb // np.dtype(dt[f[2]]).itemsize,
                                      offset=doff + off).reshape(shape)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    t0 = time.perf_counter()
    eng = N.create(args.weights, 0)
    info = N.info(eng)
    print(f"load {time.perf_counter() - t0:.1f}s  {info}")
    ah, ad = info["action_horizon"], info["action_dim"]
    out = np.zeros((ah, ad), np.float64)
    worst = 0.0
    for i in range(args.samples):
        p = pathlib.Path(args.ref) / f"gpu_{i:03d}.pi05d"
        if not p.exists():
            break
        d = read_container(p)
        img = d["in.img"]
        prompt = bytes(d["in.prompt"]).decode()
        t = N.infer(eng, img[0], img[1], img[2], d["in.proprio"], prompt, d["in.noise"], out)
        ref = d["out.actions"]
        err = np.abs(out - ref).max()
        worst = max(worst, err)
        print(f"sample {i}: max|ours - jax| = {err:.3e} (robot units)  tokens {t['tokens']}  total {t['total_ms']:.1f} ms")
    # timing through the Python boundary, receding-horizon act() path
    d = read_container(pathlib.Path(args.ref) / "gpu_000.pi05d")
    img = d["in.img"]
    prompt = bytes(d["in.prompt"]).decode()
    a = np.zeros(ad, np.float32)
    ts = []
    for _ in range(args.iters):
        s = time.perf_counter()
        N.infer(eng, img[0], img[1], img[2], d["in.proprio"], prompt, None, out)
        ts.append((time.perf_counter() - s) * 1e3)
    print(f"infer via Python: median {np.median(ts):.2f} ms  min {np.min(ts):.2f} ms")
    new_count = 0
    for k in range(64):
        new, _ = N.act(eng, 0, img[0], img[1], img[2], d["in.proprio"], prompt, 16, a)
        new_count += new
    print(f"act(): 64 steps -> {new_count} inferences (expect 4)")
    print(f"worst max|diff| over samples {worst:.3e}")


if __name__ == "__main__":
    main()
