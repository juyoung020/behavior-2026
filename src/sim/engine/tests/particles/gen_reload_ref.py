# 층 0(particles) 정답지: 거시 물리 입자 재적재 왕복 — removing_objects 의 dump_state/load_state 가 입자 계에 하는 일.
#   dump (system_base.py:358 BaseSystem._dump_state, _store_local_poses=False): 중심 = 원점 + T.quat2mat(q) @ offset
#        (macro_particle_system.py:1302) → 장면 좌표로 _transform_poses(scene.pose_inv) (위치 mm, 방향 mat2quat(rot @ quat2mat(q)))
#   load (:381): _transform_poses(scene.pose) → set_particles_position_orientation(positions, orientations) (:1325).
#        위치·방향을 둘 다 주면 offset 을 빼는 갈래를 건너뛰어 "중심"이 그대로 원점이 된다 (공식 동작 — 재적재마다 입자가 R@offset 만큼 옮겨짐).
#   이 파일은 공식 BaseSystem._dump_state/_load_state 자체를 가짜 뷰·장면에 붙여 부른다.
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_reload_ref.py --out ~/engine-data/particles/reload
import argparse
import os

import numpy as np
import torch as th

import omnigibson.systems.macro_particle_system as MPS
import omnigibson.systems.system_base as SB
import omnigibson.utils.transform_utils as T


class View:
    def __init__(self, tfs):
        self.tfs = tfs
        self.got = None

    def get_transforms(self):
        return self.tfs.clone()

    def set_transforms(self, x, indices):
        self.got = x.clone()


class Sys:
    get_particles_position_orientation = MPS.MacroPhysicalParticleSystem.get_particles_position_orientation
    set_particles_position_orientation = MPS.MacroPhysicalParticleSystem.set_particles_position_orientation
    _transform_poses = SB.BaseSystem._transform_poses
    _dump_state = SB.BaseSystem._dump_state
    _load_state = SB.BaseSystem._load_state
    _store_local_poses = False
    name = "fake"

    def __init__(self, tfs, off, scene):
        self.particles_view = View(tfs)
        self._particle_offset = off
        self.n_particles = len(tfs)
        self.scene = scene
        self.min_scale = th.ones(3)
        self.max_scale = th.ones(3)


class Scene:
    def __init__(self, p):
        self.pose = T.pose2mat((th.tensor(p, dtype=th.float32), th.tensor([0.0, 0.0, 0.0, 1.0])))
        self.pose_inv = th.linalg.inv_ex(self.pose).inverse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--no-warm", action="store_true", help="mat2quat 을 먼저 여러 모양으로 부르지 않음 (커널 이력 민감도 확인용)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for shp in [(4,), (3, 4), (5, 4), (2, 5), (3, 6)]:  # 평가기처럼 동적 모양 커널로 (warm_torch_compile)
        T.quat2mat(th.rand(*shp))
    for shp in [] if a.no_warm else [(3, 3), (4, 3, 3), (6, 3, 3)]:
        T.mat2quat(th.eye(3).expand(*shp).contiguous())
    rng = np.random.default_rng(53)
    tf_all, off_all, out_all, n_all, sp_all = [], [], [], [], []
    for c in range(a.n):
        n = int(rng.integers(1, 80))
        q = rng.normal(0, 1, (n, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        tfs = th.tensor(np.c_[rng.uniform(-3, 3, (n, 3)), q], dtype=th.float32)
        off = th.tensor(rng.normal(0, 0.01, 3), dtype=th.float32)
        sp = [0.0, 0.0, 0.0] if c % 4 else rng.uniform(-20, 20, 3).tolist()  # 평가기 장면 0 은 원점, 몇 사례는 옮긴 장면
        s = Sys(tfs, off, Scene(sp))
        s._load_state(s._dump_state())
        sp_all.append(np.stack([s.scene.pose.numpy(), s.scene.pose_inv.numpy()]))
        tf_all.append(tfs.numpy())
        off_all.append(off.numpy())
        out_all.append(s.particles_view.got.numpy())
        n_all.append(n)
    np.save(os.path.join(a.out, "n.npy"), np.array(n_all, np.int32))
    np.save(os.path.join(a.out, "tfs.npy"), np.ascontiguousarray(np.concatenate(tf_all)))
    np.save(os.path.join(a.out, "off.npy"), np.stack(off_all))
    np.save(os.path.join(a.out, "scene_pose.npy"), np.stack(sp_all))
    np.save(os.path.join(a.out, "out.npy"), np.ascontiguousarray(np.concatenate(out_all)))
    print("done", a.n, "입자", sum(n_all))


if __name__ == "__main__":
    main()
