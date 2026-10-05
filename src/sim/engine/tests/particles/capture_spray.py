"""층 0(particles) 실제 과제 기록 (뿌리기): 공식 평가기(무수정)로 spraying_for_bugs 등 한 판을 불러, 분무기를 대상 위에 세우고 켠 채
매 스텝 순간이동시키며, 도포기 _modify_particles 한 번마다 입력(난수 상태·링크 자세·척도·투영 범위·무리 척도 범위)과
장면 질의(raycast_all 요청·적중, PhysX 공식 값) 그리고 결과(새 입자의 무리·링크·국소 행렬, 그 순간 링크 scaled_transform)를 적는다.
test_spray_capture 가 같은 입력과 공식 적중으로 같은 광선 요청·같은 국소 행렬을 내는지 본다.
  WSL (GPU 대기열 안에서): bash run_capture_spray.sh <과제> [스텝=60] [인스턴스=0]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "capture")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch as th
    from omegaconf import OmegaConf

    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.eval import evaluator as E
    from omnigibson.eval.evaluator import resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
    import omnigibson.object_states.particle_modifier as PM
    from omnigibson.object_states import ToggledOn
    from physx_capture import install_no_render

    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    install_no_render()
    seed = seed_everything(DEFAULT_EVAL_SEED)

    events = []
    cur_step = [-1]
    cur = [None]
    orig_mod = PM.ParticleApplier._modify_particles

    def mod_wrap(self, system):
        link = self.link
        p, q = link.get_position_orientation()
        grp = system.get_group_name(obj=self.obj)
        ev = dict(step=cur_step[0], obj=self.obj.name, system=system.name, rng=th.get_rng_state().numpy().tolist(),
                  link_p=p.numpy().tolist(), link_q=q.numpy().tolist(), obj_scale=self.obj.scale.numpy().tolist(),
                  ext=self._projection_mesh_params["extents"].numpy().tolist(), ptype=self._projection_mesh_params["type"],
                  rel=bool(system._scale_relative_to_parent), tmpl=system.particle_object.aabb_extent.numpy().tolist(),
                  before=list(system.particles.keys()) if system.particles else [], rays=[], ignore=sorted(l.prim_path for l in self.obj.links.values()))
        cur[0] = ev
        r = orig_mod(self, system)
        # 무리 척도 범위 (이번에 처음 만들어졌을 수 있음 — 호출 뒤 값)
        if grp in system._group_scales:
            ev["grp_min"], ev["grp_max"] = [x.numpy().tolist() for x in system._group_scales[grp]]
        new = [n for n in (system.particles.keys() if system.particles else []) if n not in set(ev["before"])]
        ev["new"] = []
        for n in new:
            info = system._particles_info[n]
            ev["new"].append(dict(name=n, obj=info["obj"].name, link=info["link"].prim_path, lm=system._particles_local_mat[n].numpy().tolist(),
                                  link_tf=info["link"].scaled_transform.numpy().tolist()))
        events.append(ev)
        cur[0] = None
        return r

    PM.ParticleApplier._modify_particles = mod_wrap

    inst = resolve_instance_ids(a.task, [a.instance], mode="public_test")
    cfg = OmegaConf.create(dict(
        env_wrapper={"_target_": "omnigibson.eval.wrappers.DefaultWrapper"}, policy_name="local",
        model={"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None}, headless=True,
        partial_scene_load=True, max_steps=a.steps + 10, write_video=False, mode="public_test", seed=seed, num_envs=1,
        task={"name": a.task}, robot=OmegaConf.load("/mnt/c/behavior-2026/src/sim/configs/r1pro_robot.yaml")))
    ev_ = E.BatchedEvaluator(cfg)
    try:
        ev_.load_batch({0: int(inst[0])})
        scene = og.sim.scenes[0]
        # 장면 질의 가로채기 (값만 읽음): sampling_utils.raytest (필터·가장 가까운 것 고른 뒤 결과)
        import omnigibson.utils.sampling_utils as SU

        orig_rt = SU.raytest

        def rt(start_point, end_point, *args, **kw):
            r = orig_rt(start_point, end_point, *args, **kw)
            if cur[0] is not None:
                h = None if not r.get("hit") else (r["position"].tolist(), r["normal"].tolist(), float(r["distance"]), r["rigidBody"])
                cur[0]["rays"].append((th.as_tensor(start_point).tolist(), th.as_tensor(end_point).tolist(), h))
            return r

        SU.raytest = rt
        apps = [o for o in scene.objects if PM.ParticleApplier in o.states]
        scope = ev_.instance_eval_states[0].env_accessor.object_scope
        targets = [e for k, e in scope.items() if e is not None and hasattr(e, "category") and e not in apps and k.split(".")[0] in ("pot_plant", "tree")]
        print("[particles] 도포기:", [o.name for o in apps], "대상:", [t.name for t in targets], flush=True)
        atom = apps[0]
        st_ = atom.states[PM.ParticleApplier]
        print("[particles] 진단: 활성 계", list(scene.active_systems.keys()), "| 도포기 조건 계", list(st_.conditions.keys()),
              "| systems_to_check", st_.systems_to_check,
              "| 계 초기화", {k: scene.get_system(k, force_init=False).initialized for k in st_.conditions.keys()}, flush=True)
        orig_upd = PM.ParticleModifier._update

        def upd(self):
            if isinstance(self, PM.ParticleApplier) and cur_step[0] in (0, 1, 5, 10, 30):
                print(f"[particles] 진단 스텝 {cur_step[0]}: _current_step {self._current_step} 확인 계 {self.systems_to_check} "
                      f"켜짐 {self.obj.states[ToggledOn].get_value()} 조건 {[bool(c(self.obj)) for c in self.conditions['insectifuge']]}", flush=True)
            return orig_upd(self)

        PM.ParticleModifier._update = upd
        for step in range(a.steps):
            cur_step[0] = step
            t = targets[(step // 10) % len(targets)]
            c = t.aabb_center
            hi = t.aabb[1]
            atom.set_position_orientation(position=th.tensor([c[0], c[1], hi[2] + 0.15], dtype=th.float32), orientation=th.tensor([-0.70710678, 0, 0, 0.70710678]))
            atom.set_linear_velocity(th.zeros(3))
            atom.set_angular_velocity(th.zeros(3))
            atom.states[ToggledOn].set_value(True)
            ev_._step_fn([0])
        with open(os.path.join(a.out, "spray_events.json"), "w") as f:
            json.dump(events, f)
        n_new = sum(len(e["new"]) for e in events)
        print(f"[particles] 기록 끝: 도포 사건 {len(events)}, 광선 {sum(len(e['rays']) for e in events)}, 새 입자 {n_new}", flush=True)
    finally:
        td = getattr(og, "tempdir", None)
        if td and os.path.isdir(td) and os.path.basename(td).startswith("tmp"):
            shutil.rmtree(td, ignore_errors=True)
            print(f"[particles] 임시 장면 폴더 지움: {td}", flush=True)
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
