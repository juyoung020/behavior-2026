# 층 0(particles) 정답지: scipy.spatial.Delaunay.find_simplex (메시 부피 안 판정, geom_prim.py:205).
# 메시마다 삼각분할 자료를 내보내고(엔진 입력과 같은 모양), 경계 근처 점을 섞은 묶음 하나를 공식처럼 한 번에 찾는다.
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_delaunay_ref.py --out ~/engine-data/particles/delaunay
import argparse
import os

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--meshes", type=int, default=200)
    ap.add_argument("--pts", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for m in range(a.meshes):
        kind = m % 4
        if kind == 0:  # 볼록 껍질 꼭짓점 (fillable 메시 모양)
            p = rng.normal(0, 1, (60, 3)) * rng.uniform(0.05, 0.5, 3)
            p = p[ConvexHull(p).vertices]
        elif kind == 1:  # 상자 모양 (축 정렬 면 — 동률·퇴화 많음)
            e = rng.uniform(0.05, 0.4, 3)
            g = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * e
            p = np.vstack([g, rng.uniform(-1, 1, (int(rng.integers(0, 12)), 3)) * e])
        elif kind == 2:  # 메시 꼭짓점 구름 (float32 에서 온 값)
            p = rng.uniform(-0.3, 0.3, (int(rng.integers(8, 200)), 3)).astype(np.float32).astype(np.float64)
        else:  # 원기둥 비슷한 그릇
            t = np.linspace(0, 2 * np.pi, 17)[:-1]
            r, h = rng.uniform(0.05, 0.3), rng.uniform(0.05, 0.4)
            p = np.vstack([np.c_[r * np.cos(t), r * np.sin(t), np.full(16, -h)], np.c_[r * np.cos(t), r * np.sin(t), np.full(16, h)]])
            p = p.astype(np.float32).astype(np.float64)
        tri = Delaunay(p)
        lo, hi = p.min(0), p.max(0)
        q = rng.uniform(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), (a.pts, 3))
        # 경계 근처: 겉면 삼각형 위의 점을 몇 ulp 흔듦, 꼭짓점 자신, 모서리
        hull = ConvexHull(p)
        nb = a.pts // 2
        f = hull.simplices[rng.integers(len(hull.simplices), size=nb)]
        w = rng.dirichlet([1, 1, 1], nb)
        onf = (p[f] * w[:, :, None]).sum(1)
        q[:nb] = onf
        q[nb : nb + 100] = p[rng.integers(len(p), size=100)]
        q = q.astype(np.float32).astype(np.float64)  # 공식 입력은 float32 → double
        for _ in range(int(rng.integers(0, 3))):
            q[:nb] = np.nextafter(q[:nb], np.where(rng.random((nb, 3)) < 0.5, np.inf, -np.inf))
        res = tri.find_simplex(q)
        d = os.path.join(a.out, f"mesh_{m:03d}")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "transform.npy"), np.ascontiguousarray(tri.transform.reshape(-1, 12)))
        np.save(os.path.join(d, "neighbors.npy"), np.ascontiguousarray(tri.neighbors.astype(np.int32)))
        np.save(os.path.join(d, "equations.npy"), np.ascontiguousarray(tri.equations))
        np.save(os.path.join(d, "misc.npy"), np.r_[tri.paraboloid_scale, tri.paraboloid_shift, tri.min_bound, tri.max_bound])
        np.save(os.path.join(d, "q.npy"), q)
        np.save(os.path.join(d, "res.npy"), res.astype(np.int32))
    print("done meshes", a.meshes)


if __name__ == "__main__":
    main()
