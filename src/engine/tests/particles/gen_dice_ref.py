# 층 0(particles) 정답지: 다지기 입자 자리 — PhysicalParticleSystem.generate_particles_from_link (system_base.py:896) 를 가짜 링크에 붙여 그대로.
#   DicingRule(transition_rules.py:1018): generate_particles_from_link(obj, root_link, check_contact=False, use_visual_meshes=False)
#   격자 = th.arange(lo + r, hi - r, 2r) 세 축 → meshgrid → 충돌 메시(볼록, Delaunay find_simplex)마다 안 판정 OR → generate_particles(위치)
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_dice_ref.py --out ~/engine-data/particles/dice
import argparse
import os

import numpy as np
import torch as th
from scipy.spatial import ConvexHull

import omnigibson.systems.system_base as SB
import omnigibson.utils.transform_utils as T
from omnigibson.prims.geom_prim import GeomPrim
from omnigibson.prims.rigid_prim import RigidPrim


class FakeMesh:
    check_points_in_volume = GeomPrim.check_points_in_volume
    check_local_points_in_volume = GeomPrim.check_local_points_in_volume
    delaunay_triangulation = GeomPrim.__dict__["delaunay_triangulation"]

    def __init__(self, pts, tf):
        self._mesh_type = "Mesh"
        self.points = th.from_numpy(pts)
        self.scaled_transform = tf


class FakeLink:
    check_points_in_volume = RigidPrim.check_points_in_volume

    def __init__(self, lo, hi, meshes):
        self.visual_aabb = (th.from_numpy(lo), th.from_numpy(hi))
        self.visual_aabb_extent = th.from_numpy(hi) - th.from_numpy(lo)
        self.collision_meshes = meshes
        self.visual_meshes = meshes
        self.name = "fake"


class FakeSys:
    generate_particles_from_link = SB.PhysicalParticleSystem.generate_particles_from_link
    particle_particle_rest_distance = SB.PhysicalParticleSystem.__dict__["particle_particle_rest_distance"]
    initialized = True

    def __init__(self, r):
        self.particle_radius = th.tensor(r, dtype=th.float32)
        self.got = None

    def generate_particles(self, positions, **kw):
        self.got = positions


def rand_tf(rng):
    x, y, z, w = (lambda q: q / np.linalg.norm(q))(rng.standard_normal(4))
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (z * x - y * w)],
                  [2 * (x * y - z * w), 1 - 2 * (z * z + x * x), 2 * (y * z + x * w)],
                  [2 * (z * x + y * w), 2 * (y * z - x * w), 1 - 2 * (y * y + x * x)]])
    M = np.eye(4)
    M[:3, :3] = np.diag(rng.uniform(0.5, 1.5, 3)) @ R
    M[3, :3] = rng.uniform(-2, 2, 3)
    return th.tensor(M, dtype=th.float32).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.default_rng(29)
    for shp in [(4,), (3, 4), (5, 4), (2, 5), (3, 6)]:
        T.quat2mat(th.rand(*shp))
    for c in range(a.n):
        d = os.path.join(a.out, f"case_{c:03d}")
        os.makedirs(d, exist_ok=True)
        nm = int(rng.integers(1, 4))
        meshes, mpts, mtf = {}, [], []
        tf = rand_tf(rng)
        allw = []
        for k in range(nm):
            p = (rng.normal(0, 1, (40, 3)) * rng.uniform(0.02, 0.12, 3) + rng.normal(0, 0.03, 3))
            p = p[ConvexHull(p).vertices].astype(np.float32)
            t = tf if rng.random() < 0.7 else rand_tf(rng)
            meshes[f"m{k}"] = FakeMesh(p, t)
            pp = np.zeros((64, 3), np.float32)
            pp[: len(p)] = p
            mpts.append(pp)
            mtf.append(t.numpy())
            allw.append((np.c_[p, np.ones(len(p))] @ t.numpy().astype(np.float64).T)[:, :3])
        allw = np.vstack(allw)
        lo = allw.min(0).astype(np.float32)
        hi = allw.max(0).astype(np.float32)
        r = np.float32(rng.uniform(0.004, 0.02))
        if np.any(hi - lo < 2 * r):
            r = np.float32((hi - lo).min() / 2.5)
        s = FakeSys(float(r))
        s.generate_particles_from_link(obj=None, link=FakeLink(lo, hi, meshes), use_visual_meshes=False, check_contact=False)
        np.save(os.path.join(d, "lo.npy"), lo)
        np.save(os.path.join(d, "hi.npy"), hi)
        np.save(os.path.join(d, "r.npy"), np.array([r], np.float32))
        np.save(os.path.join(d, "mesh_pts.npy"), np.stack(mpts))
        np.save(os.path.join(d, "mesh_n.npy"), np.array([len(m.points) for m in meshes.values()], np.int32))
        np.save(os.path.join(d, "mesh_tf.npy"), np.ascontiguousarray(np.stack(mtf)))
        np.save(os.path.join(d, "pos.npy"), np.ascontiguousarray(s.got.numpy().astype(np.float32)))
        # 메시마다 Delaunay (엔진 입력 모양)
        for k, m in enumerate(meshes.values()):
            tri = m.delaunay_triangulation
            np.save(os.path.join(d, f"dl{k}_transform.npy"), np.ascontiguousarray(tri.transform.reshape(-1, 12)))
            np.save(os.path.join(d, f"dl{k}_neighbors.npy"), np.ascontiguousarray(tri.neighbors.astype(np.int32)))
            np.save(os.path.join(d, f"dl{k}_equations.npy"), np.ascontiguousarray(tri.equations))
            np.save(os.path.join(d, f"dl{k}_misc.npy"), np.r_[tri.paraboloid_scale, tri.paraboloid_shift, tri.min_bound, tri.max_bound])
    print("done", a.n)


if __name__ == "__main__":
    main()
