# 층 0(particles) 정답지: Contains / Filled (object_states/contains.py, filled.py) — 공식 함수를 가짜 물체에.
#   물리 입자 중심 = 강체 위치 + T.quat2mat(방향) @ offset (macro_particle_system.py:1302)
#   ContainedParticles: 용기 메타링크 메시(Mesh=Delaunay / 원기둥 등) 안 판정, Contains = 개수 > 0
#   Filled = (2r)^3 * 개수 / link.volume > 0.2
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_contain_ref.py --out ~/engine-data/particles/contain
import argparse
import os

import numpy as np
import torch as th
from scipy.spatial import ConvexHull

import omnigibson.object_states.contains as CT
import omnigibson.object_states.filled as FL
import omnigibson.systems.macro_particle_system as MPS
import omnigibson.utils.transform_utils as T
from omnigibson.prims.geom_prim import GeomPrim
from omnigibson.prims.rigid_prim import RigidPrim


class FakeView:
    def __init__(self, tfs):
        self.tfs = tfs

    def get_transforms(self):
        return self.tfs.clone()


class FakeSys:
    get_particles_position_orientation = MPS.MacroPhysicalParticleSystem.get_particles_position_orientation
    name = "diced__apple"

    def __init__(self, tfs, off, r):
        self.particles_view = FakeView(tfs)
        self._particle_offset = off
        self.particle_radius = r
        self.n_particles = len(tfs)


class FakeScene:
    def is_visual_particle_system(self, system_name):
        return False

    def is_physical_particle_system(self, system_name):
        return True


class FakeMesh:
    check_points_in_volume = GeomPrim.check_points_in_volume
    check_local_points_in_volume = GeomPrim.check_local_points_in_volume
    delaunay_triangulation = GeomPrim.__dict__["delaunay_triangulation"]

    def __init__(self, kind, pts, attrs, tf):
        self._mesh_type, self.points, self._attrs, self.scaled_transform = kind, pts, attrs, tf

    def get_attribute(self, k):
        return self._attrs[k]


class FakeLink:
    check_points_in_volume = RigidPrim.check_points_in_volume

    def __init__(self, meshes, volume):
        self.visual_meshes = meshes
        self.volume = volume


class FakeObj:
    def __init__(self):
        self.scene = FakeScene()
        self.states = {}


def rand_q(rng, n):
    q = rng.standard_normal((n, 4))
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(31)
    for shp in [(4,), (3, 4), (5, 4), (2, 5), (3, 6)]:
        T.quat2mat(th.rand(*shp))
    rows = []
    for c in range(a.n):
        d = os.path.join(a.out, f"case_{c:03d}")
        os.makedirs(d, exist_ok=True)
        n = int(rng.integers(1, 300))
        center = rng.uniform(-1, 1, 3)
        size = rng.uniform(0.05, 0.3)
        pos = (center + rng.normal(0, size, (n, 3))).astype(np.float32)
        tfs = th.from_numpy(np.c_[pos, rand_q(rng, n)].astype(np.float32))
        off = th.from_numpy(rng.normal(0, 0.005, 3).astype(np.float32))
        r = th.tensor(float(rng.uniform(0.003, 0.02)), dtype=th.float32)
        sys = FakeSys(tfs, off, r)
        kind = ["Mesh", "Cylinder", "Cube", "Sphere", "Cone"][c % 5]
        x, y, z, w = rand_q(rng, 1)[0].astype(np.float64)
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (z * x - y * w)],
                      [2 * (x * y - z * w), 1 - 2 * (z * z + x * x), 2 * (y * z + x * w)],
                      [2 * (z * x + y * w), 2 * (y * z - x * w), 1 - 2 * (y * y + x * x)]])
        M = np.eye(4)
        M[:3, :3] = np.diag(np.full(3, size * rng.uniform(1, 3)) if kind != "Mesh" else rng.uniform(0.5, 1.5, 3)) @ R
        M[3, :3] = center
        tf = th.tensor(M, dtype=th.float32).T
        pts = rng.normal(0, 1, (50, 3)) * rng.uniform(0.05, 0.3, 3)
        pts = pts[ConvexHull(pts).vertices].astype(np.float32)
        mesh = FakeMesh(kind, th.from_numpy(pts), {"radius": 0.5, "height": 1.0, "size": 1.0}, tf)
        volume = float(rng.uniform(0.2, 1.2) * (2 * float(r)) ** 3 * n / 0.2)  # 0.2 문턱 근처가 나오게
        link = FakeLink({"m": mesh}, volume)
        obj = FakeObj()
        cst = type("C", (), {"obj": obj, "link": link})()
        data = CT.ContainedParticles._get_value(cst, sys)
        obj.states[CT.ContainedParticles] = type("S", (), {"get_value": lambda self, s: data, "link": link})()
        fst = type("F", (), {"obj": obj})()
        filled = FL.Filled._get_value(fst, sys)
        np.save(os.path.join(d, "tfs.npy"), tfs.numpy())
        np.save(os.path.join(d, "off.npy"), off.numpy())
        np.save(os.path.join(d, "mesh_tf.npy"), np.ascontiguousarray(tf.numpy()))
        np.save(os.path.join(d, "centers.npy"), np.ascontiguousarray(data.positions.numpy()))
        np.save(os.path.join(d, "in_volume.npy"), data.in_volume.numpy().astype(np.uint8))
        np.save(os.path.join(d, "scal.npy"), np.array([float(r), volume], np.float64))
        np.save(os.path.join(d, "kind.npy"), np.array([c % 5, int(data.n_in_volume), int(filled)], np.int32))
        if kind == "Mesh":
            tri = mesh.delaunay_triangulation
            np.save(os.path.join(d, "dl_transform.npy"), np.ascontiguousarray(tri.transform.reshape(-1, 12)))
            np.save(os.path.join(d, "dl_neighbors.npy"), np.ascontiguousarray(tri.neighbors.astype(np.int32)))
            np.save(os.path.join(d, "dl_equations.npy"), np.ascontiguousarray(tri.equations))
            np.save(os.path.join(d, "dl_misc.npy"), np.r_[tri.paraboloid_scale, tri.paraboloid_shift, tri.min_bound, tri.max_bound])
        rows.append(filled)
    print("done", a.n, "filled", sum(rows))


if __name__ == "__main__":
    main()
