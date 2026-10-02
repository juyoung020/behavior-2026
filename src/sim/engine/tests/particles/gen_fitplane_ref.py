# 층 0(particles) 정답지: sampling_utils.fit_plane (평행 광선 적중점 → 평면 중심·법선). 중간값(평균, x.T@x, U)도 적는다.
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_fitplane_ref.py --out ~/engine-data/particles/fitplane
import argparse
import os

import numpy as np
import torch as th

import omnigibson.utils.sampling_utils as SU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(41)
    K = 256
    P = np.zeros((a.n, K, 3), np.float32)
    ks = np.zeros(a.n, np.int32)
    out = np.zeros((a.n, 3 + 3 + 9 + 9), np.float32)
    for i in range(a.n):
        k = int(rng.integers(3, 33)) if rng.random() < 0.7 else int(rng.integers(33, K + 1))
        # 평면 위(가끔 약간 벗어난) 적중점: 격자 광선이 표면에 맞은 꼴
        o = rng.uniform(-2, 2, 3)
        n = rng.normal(0, 1, 3)
        n /= np.linalg.norm(n)
        u = np.cross(n, rng.normal(0, 1, 3))
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        g = rng.uniform(-0.02, 0.02, (k, 2))
        pts = o + g[:, :1] * u + g[:, 1:] * v + n * rng.normal(0, rng.choice([0, 1e-5, 1e-3]), (k, 1))
        pts = pts.astype(np.float32)
        t = th.from_numpy(pts)
        ctr, normal = SU.fit_plane(t, [])
        x = t - t.mean(dim=0)
        M = x.T @ x
        U = th.linalg.svd(M).U
        P[i, :k] = pts
        ks[i] = k
        out[i] = np.concatenate([ctr.numpy(), normal.numpy(), M.numpy().ravel(), U.numpy().ravel()])
    np.save(os.path.join(a.out, "pts.npy"), P)
    np.save(os.path.join(a.out, "k.npy"), ks)
    np.save(os.path.join(a.out, "out.npy"), out)
    print("done", a.n)


if __name__ == "__main__":
    main()
