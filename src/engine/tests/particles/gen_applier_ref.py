# 층 0(particles) 정답지: 시각 입자 도포기(분무기) 의 앞뒤 — 공식 메서드를 가짜 물체에 붙여 그대로.
#   A. ParticleApplier._sample_particle_locations_from_projection_volume (particle_modifier.py:1430): th.rand(2,2)·cos/sin·원뿔 → 세계 광선 시작·끝
#   B. VisualParticleSystem.sample_scales_by_group (system_base.py:672) + _compute_relative_group_scales (:634)
#   C. ParticleApplier._apply_particles_at_raycast_hits (:1312) → MacroVisualParticleSystem.generate_group_particles (:507)
#      → set_particle_position_orientation (:830, inv_ex(link_tf) @ 전역 행렬) : 붙은 입자 국소 행렬
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_applier_ref.py --out ~/engine-data/particles/applier
import argparse
import math
import os

import numpy as np
import torch as th

import omnigibson.object_states.particle_modifier as PM
import omnigibson.systems.macro_particle_system as MPS
import omnigibson.systems.system_base as SB
import omnigibson.utils.transform_utils as T
from omnigibson.object_states.saturated import ModifiedParticles

Vis = MPS.MacroVisualParticleSystem
MPS.absolute_prim_path_to_scene_relative = lambda scene, p: p  # prim 경로 변환만 (값 무관)


class St:
    def __init__(self):
        self.c = {}

    def get_value(self, system):
        return self.c.get(system.name, 0)

    def set_value(self, system, v):
        self.c[system.name] = v


class FakeLinkPose:
    def __init__(self, pos, quat, tf, name="base_link"):
        self._p, self._q, self.scaled_transform, self.name = pos, quat, tf, name

    def get_position_orientation(self):
        return self._p.clone(), self._q.clone()


class FakeObj:
    def __init__(self, name, scale, aabb_ext, link):
        self.name, self.scale, self.aabb_extent = name, scale, aabb_ext
        self.links = {"base_link": link}
        self.prim_path = f"/World/scene_0/{name}"
        self.states = {ModifiedParticles: St()}


class FakeTemplate:
    def __init__(self, ext):
        self.aabb_extent = ext


class FakeSys:
    sample_scales_by_group = SB.VisualParticleSystem.sample_scales_by_group
    _compute_relative_group_scales = SB.VisualParticleSystem._compute_relative_group_scales
    sample_scales = SB.BaseSystem.sample_scales if hasattr(SB, "BaseSystem") else SB.VisualParticleSystem.sample_scales
    generate_group_particles = Vis.generate_group_particles
    set_particle_position_orientation = Vis.set_particle_position_orientation
    get_group_name = SB.VisualParticleSystem.get_group_name
    name = "insectifuge"

    def __init__(self, rel, tmpl_ext, objs):
        self._scale_relative_to_parent = rel
        self.particle_object = FakeTemplate(tmpl_ext)
        self.min_scale = th.ones(3)
        self.max_scale = th.ones(3)
        self._group_objects = {o.name: o for o in objs}
        self._group_scales = {}
        self._group_particles = {o.name: {} for o in objs}
        self.particles = {}
        self._particles_info = {}
        self._particles_local_mat = {}
        self._CLIP_INTO_OBJECTS = False
        self.scene = None
        self.groups = set(self._group_objects)
        self.cnt = 0

    def _validate_group(self, group):
        if self._scale_relative_to_parent and group not in self._group_scales:
            self._group_scales[group] = self._compute_relative_group_scales(group=group)

    def _is_cloth_obj(self, obj):
        return False

    def create_attachment_group(self, obj):
        pass

    def add_particle(self, relative_prim_path, scale, idn=None):
        nm = f"p{self.cnt}"
        self.cnt += 1
        self.particles[nm] = None
        return type("P", (), {"name": nm})()

    def _modify_particle_local_mat(self, name, mat):
        self._particles_local_mat[name] = mat


def rq(rng):
    q = rng.normal(0, 1, 4)
    return (q / np.linalg.norm(q)).astype(np.float32)


def tf_of(p, q, s):
    x, y, z, w = [float(v) for v in q]
    M = np.eye(4)
    M[0, :3] = [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (z * x - y * w)]
    M[1, :3] = [2 * (x * y - z * w), 1 - 2 * (z * z + x * x), 2 * (y * z + x * w)]
    M[2, :3] = [2 * (z * x + y * w), 2 * (y * z - x * w), 1 - 2 * (y * y + x * x)]
    M[:3, :3] = np.diag(s.astype(np.float64)) @ M[:3, :3]
    M[3, :3] = p
    return th.tensor(M, dtype=th.float32).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(47)
    for shp in [(4,), (3, 4), (5, 4), (2, 5), (3, 6)]:
        T.quat2mat(th.rand(*shp))
    th.manual_seed(3)
    rows = []
    for c in range(a.n):
        st = th.get_rng_state().numpy().copy()
        # 분무기
        ap_scale = th.tensor(np.full(3, rng.uniform(0.7, 1.3)) if rng.random() < 0.7 else rng.uniform(0.7, 1.3, 3), dtype=th.float32)
        ap_p = th.tensor(rng.uniform(-1, 1, 3), dtype=th.float32)
        ap_q = th.tensor(rq(rng))
        ap_link = FakeLinkPose(ap_p, ap_q, tf_of(ap_p.numpy(), ap_q.numpy(), ap_scale.numpy()))
        atom = FakeObj("atomizer", ap_scale, th.tensor(rng.uniform(0.05, 0.3, 3), dtype=th.float32), ap_link)
        # 맞은 물체 (둘)
        hits_objs = []
        for k in range(2):
            s = th.tensor(np.full(3, rng.uniform(0.5, 2)) if rng.random() < 0.5 else rng.uniform(0.5, 2, 3), dtype=th.float32)
            p = th.tensor(rng.uniform(-1, 1, 3), dtype=th.float32)
            q = th.tensor(rq(rng))
            hits_objs.append(FakeObj(f"tree{k}", s, th.tensor(rng.uniform(0.2, 3.0, 3), dtype=th.float32),
                                     FakeLinkPose(p, q, tf_of(p.numpy(), q.numpy(), s.numpy()))))
        rel = bool(rng.random() < 0.6)
        sys = FakeSys(rel, th.tensor(rng.uniform(0.005, 0.05, 3), dtype=th.float32), [atom] + hits_objs)
        # A. 투영 원뿔 표본
        ext = th.tensor(rng.uniform(0.2, 1.5, 3), dtype=th.float32)
        fa = type("A", (), {})()
        fa._projection_mesh_params = {"type": "Cone", "extents": ext}
        fa.link, fa.obj = ap_link, atom
        fa._get_max_particles_limit_per_step = lambda system: 2
        startp, endp = PM.ParticleApplier._sample_particle_locations_from_projection_volume(fa, sys)
        # B. 척도 (분무기 무리)
        scales = sys.sample_scales_by_group(group="atomizer", n=2)
        avg = th.pow(th.prod(atom.scale), 1 / 3)
        cdims = scales * sys.particle_object.aabb_extent.reshape(1, 3) * avg
        # C. 적중 두 개에 입자 붙이기 (적중 = 맞은 물체 링크, 자리·방향 임의)
        hits = []
        for k in range(2):
            hp = th.tensor(rng.uniform(-1, 1, 3), dtype=th.float32)
            hq = th.tensor(rq(rng))
            hits.append((hp, th.zeros(3), hq, f"/World/scene_0/tree{int(rng.integers(2))}/base_link"))
        fa.obj.scene = type("S", (), {"object_registry": lambda self, key, val, default: sys._group_objects[val.split("/")[-1]],
                                      "is_visual_particle_system": lambda self, system_name: True})()
        fa.conditions = {sys.name: []}
        PM.ParticleApplier._apply_particles_at_raycast_hits(fa, sys, hits, scales)
        lm = np.stack([sys._particles_local_mat[f"p{i}"].numpy() for i in range(2)])
        rows.append(dict(state=st, ap_scale=ap_scale.numpy(), ap_p=ap_p.numpy(), ap_q=ap_q.numpy(), ext=ext.numpy(), rel=rel,
                         atom_aabb=atom.aabb_extent.numpy(), tmpl=sys.particle_object.aabb_extent.numpy(),
                         hit_scale=np.stack([o.scale.numpy() for o in hits_objs]), hit_tf=np.stack([o.links["base_link"].scaled_transform.numpy() for o in hits_objs]),
                         hit_p=np.stack([h[0].numpy() for h in hits]), hit_q=np.stack([h[2].numpy() for h in hits]),
                         hit_obj=np.array([int(h[3].split("/")[-2][-1]) for h in hits]),
                         start=startp.numpy(), end=endp.numpy(), scales=scales.numpy(), cdims=cdims.numpy(), lm=lm))
    np.save(os.path.join(a.out, "state.npy"), np.stack([r["state"] for r in rows]))
    flat = lambda k: np.stack([np.asarray(r[k], np.float32).ravel() for r in rows])
    for k in ["ap_scale", "ap_p", "ap_q", "ext", "rel", "atom_aabb", "tmpl", "hit_scale", "hit_tf", "hit_p", "hit_q", "hit_obj", "start", "end",
              "scales", "cdims", "lm"]:
        np.save(os.path.join(a.out, k + ".npy"), np.ascontiguousarray(flat(k)))
    print("done", a.n)


if __name__ == "__main__":
    main()
