"""기준 자료 자체 검산: 덤프의 기하·기준 prim 행렬·카메라만으로 광선을 쏴서(numpy, 전수 삼각형) 공식 RTX depth_linear 와 비교한다.
렌더러(작업자 B)와 무관하게 "뜬 장면·카메라·규약(행렬 전치, 카메라 -Z, 픽셀 중심, depth = image plane 거리)"이 맞는지 본다.

    python check_capture_depth.py <덤프 폴더> [--frames 150,500] [--n 300] [--seed 0]

출력: 카메라마다 표본 픽셀 수, 둘 다 맞음/한쪽만, |차| 중앙·p90·최대(mm). 규약이 틀리면 수십 cm~m 단위로 어긋난다.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def usd_to_aff(M):
    M = np.asarray(M, np.float64)
    A = np.zeros((3, 4))
    A[:, :3] = M[:3, :3].T
    A[:, 3] = M[3, :3]
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--frames", default="")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    S = json.load(open(os.path.join(a.dump, "scene.json"), encoding="utf-8"))
    G = np.load(os.path.join(a.dump, "geom.npz"))
    npts, ntri = G["n_points"], G["n_tri"]
    p0 = np.concatenate([[0], np.cumsum(npts)])
    t0 = np.concatenate([[0], np.cumsum(ntri)])
    frames = sorted(f for f in os.listdir(a.dump) if f.startswith("frame_") and f.endswith(".npz"))
    if a.frames:
        want = {int(x) for x in a.frames.split(",")}
        frames = [f for f in frames if int(f[6:10]) in want]
    rng = np.random.default_rng(a.seed)
    for fn in frames:
        Z = np.load(os.path.join(a.dump, fn))
        k = int(fn[6:10])
        AW = Z["anchor_world"]
        vis = Z["mesh_visible"]
        # 세계 삼각형 전부 (double)
        V0, E1, E2 = [], [], []
        for mi, m in enumerate(S["meshes"]):
            if not vis[mi]:
                continue
            g = m["geom"]
            P = G["points"][p0[g]:p0[g + 1]].astype(np.float64)
            T = G["tri"][t0[g]:t0[g + 1]].astype(np.int64)
            W = np.asarray(m["rel"]) @ AW[m["anchor"]]  # USD 행 벡터: p_w = p * rel * anchor
            Pw = np.c_[P, np.ones(len(P))] @ W
            Pw = Pw[:, :3]
            v0, v1, v2 = Pw[T[:, 0]], Pw[T[:, 1]], Pw[T[:, 2]]
            V0.append(v0); E1.append(v1 - v0); E2.append(v2 - v0)
        V0, E1, E2 = np.concatenate(V0), np.concatenate(E1), np.concatenate(E2)
        cams = json.loads(str(Z["cams_json"]))
        print(f"== 프레임 {k}: 삼각형 {len(V0):,}")
        for role, c in cams.items():
            key = f"img::{role}::depth_linear"
            if key not in Z.files:
                continue
            D = Z[key].reshape(Z[key].shape[0], Z[key].shape[1])
            H, Wd = D.shape
            M = np.asarray(c["world"], np.float64)
            ax = [M[i, :3] / np.linalg.norm(M[i, :3]) for i in range(3)]
            o = M[3, :3]
            p = c["params"]
            tanx = float(p["horizontalAperture"]) / 2 / float(p["focalLength"])
            tany = tanx * H / Wd
            rs, cs = rng.integers(0, H, a.n), rng.integers(0, Wd, a.n)
            ours = np.full(a.n, np.inf)
            for i in range(a.n):
                x = (cs[i] + 0.5) / Wd * 2 - 1
                y = 1 - (rs[i] + 0.5) / H * 2
                d = ax[0] * (x * tanx) + ax[1] * (y * tany) - ax[2]  # 카메라 -Z 가 시선, 시선 성분 = 1
                pv = np.cross(d, E2)
                det = np.einsum("ij,ij->i", E1, pv)
                ok = np.abs(det) > 1e-12
                inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
                s = o - V0
                u = np.einsum("ij,ij->i", s, pv) * inv
                q = np.cross(s, E1)
                v = (q @ d) * inv
                t = np.einsum("ij,ij->i", E2, q) * inv
                hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4)
                if hit.any():
                    ours[i] = t[hit].min()
            off = D[rs, cs].astype(np.float64)
            vo, vr = np.isfinite(off) & (off > 0) & (off < 1e6), np.isfinite(ours)
            both = vo & vr
            dd = np.abs(off[both] - ours[both]) * 1e3
            print(f"  {role:11s} {Wd}x{H} tan {tanx:.4f}: 표본 {a.n}, 둘 다 {both.sum()}, 공식만 {int((vo & ~vr).sum())}, 우리만 {int((vr & ~vo).sum())}"
                  + (f" | |차| mm 중앙 {np.median(dd):.3f} p90 {np.percentile(dd, 90):.3f} 최대 {dd.max():.1f}"
                     f", 1 cm 안 {np.mean(dd <= 10) * 100:.1f}%" if both.any() else ""))


if __name__ == "__main__":
    main()
