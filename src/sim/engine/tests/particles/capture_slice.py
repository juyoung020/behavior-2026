"""층 0(particles) 실제 과제 기록 (자르기·다지기): 공식 평가기(무수정)로 chopping_wood 등 한 판을 불러 자르개를 대상 위에 두어
SlicerActive 가 켜지게 기다린 뒤 겹치게 해 공식 전이를 일으키고, 전이 입력·출력을 가로채 적는다:
  SlicingRule.transition 입력(자를 것 자세·척도·object_parts) / 출력(반쪽 bb 자세·크기)
  TransitionRuleAPI.execute_transition 뒤: 새 물체의 범주·모델·척도·ig:nativeBB·ig:offsetBaseLink·set_position_orientation 인자·PhysX 자세
  DicingRule 이면 생긴 입자 계·개수·위치 (다지기)
test_slice_capture 가 slicing.h 로 같은 값을 내는지 본다. 같은 실행에서 뜬 OVD 는 없다(이 스크립트는 기록 전용 — 물리 틀 수확은 리드 도구).
  WSL (GPU 대기열 안에서): bash run_capture_slice.sh <과제> [스텝=160] [인스턴스=0] [자르개 키] [대상 키]
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "capture")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--slicer", default="ax.n.01_1")
    ap.add_argument("--target", default="log.n.01_1")
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
    import omnigibson.transition_rules as TR
    from omnigibson.objects.dataset_object import DatasetObject
    from physx_capture import install_no_render

    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    install_no_render()
    seed = seed_everything(DEFAULT_EVAL_SEED)
    rec = {"slice": [], "added": [], "setpose": [], "dice": []}
    cur_step = [-1]
    L = lambda t: t.detach().cpu().numpy().tolist()

    orig_tr = TR.SlicingRule.transition

    def tr_wrap(self, object_candidates):
        ins = []
        for o in object_candidates["sliceable"]:
            p, q = o.get_position_orientation()
            ins.append(dict(name=o.name, pos=L(p), orn=L(q), scale=L(o.scale),
                            parts=json.loads(json.dumps(o.metadata["object_parts"], default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))))
        res = orig_tr(self, object_candidates)
        outs = [dict(bb_pos=L(x.bb_pos), bb_orn=L(x.bb_orn), name=x.obj.name, category=x.obj.category,
                     bbox=L(x.obj._load_config["bounding_box"])) for x in res.add]
        rec["slice"].append(dict(step=cur_step[0], ins=ins, outs=outs, removed=[o.name for o in res.remove]))
        return res

    TR.SlicingRule.transition = tr_wrap

    orig_dice = TR.DicingRule.transition
    import omnigibson.systems.macro_particle_system as MPS
    import omnigibson.systems.system_base as SB

    gen_rec = []
    orig_gfl = SB.PhysicalParticleSystem.generate_particles_from_link

    def gfl_wrap(self, obj, link, use_visual_meshes=True, **kw):
        lo, hi = link.visual_aabb
        meshes = link.visual_meshes if use_visual_meshes else link.collision_meshes
        gen_rec.append(dict(system=self.name, radius=float(self.particle_radius), lo=L(lo), hi=L(hi), offset=L(self._particle_offset),
                            meshes=[dict(type=m._mesh_type, points=L(m.points) if m._mesh_type == "Mesh" else None, tf=L(m.scaled_transform))
                                    for m in meshes.values()],
                            rng=th.get_rng_state().numpy().tolist(), n_before=self.n_particles))
        return orig_gfl(self, obj, link, use_visual_meshes=use_visual_meshes, **kw)

    SB.PhysicalParticleSystem.generate_particles_from_link = gfl_wrap
    orig_gp = MPS.MacroPhysicalParticleSystem.generate_particles

    def gp_wrap(self, positions, orientations=None, **kw):
        if gen_rec and "centers" not in gen_rec[-1]:
            gen_rec[-1]["centers"] = L(positions)
        r = orig_gp(self, positions, orientations=orientations, **kw)
        if gen_rec and "frames" not in gen_rec[-1]:
            tfs = self.particles_view.get_transforms()
            gen_rec[-1]["frames"] = L(tfs)
            gen_rec[-1]["rng_after"] = th.get_rng_state().numpy().tolist()
        return r

    MPS.MacroPhysicalParticleSystem.generate_particles = gp_wrap

    def dice_wrap(self, object_candidates):
        names = [o.name for o in object_candidates["diceable"]]
        n0 = len(gen_rec)
        res = orig_dice(self, object_candidates)
        rec["dice"].append(dict(step=cur_step[0], diced=names, gens=gen_rec[n0:]))
        return res

    TR.DicingRule.transition = dice_wrap

    orig_set = DatasetObject.set_bbox_center_position_orientation

    def set_wrap(self, position=None, orientation=None):
        rec["setpose"].append(dict(step=cur_step[0], name=self.name, bb_pos=None if position is None else L(position),
                                   bb_orn=None if orientation is None else L(orientation), scale=L(self.scale),
                                   native_bbox=L(self.native_bbox), offset=L(self.base_link_offset),
                                   center=L(self.scaled_bbox_center_in_base_frame)))
        r = orig_set(self, position=position, orientation=orientation)
        p, q = self.get_position_orientation()
        rec["setpose"][-1].update(after_pos=L(p), after_orn=L(q))
        return r

    DatasetObject.set_bbox_center_position_orientation = set_wrap

    inst = resolve_instance_ids(a.task, [a.instance], mode="public_test")
    cfg = OmegaConf.create(dict(
        env_wrapper={"_target_": "omnigibson.eval.wrappers.DefaultWrapper"}, policy_name="local",
        model={"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None}, headless=True,
        partial_scene_load=True, max_steps=a.steps + 10, write_video=False, mode="public_test", seed=seed, num_envs=1,
        task={"name": a.task}, robot=OmegaConf.load("/mnt/c/behavior-2026/src/sim/configs/r1pro_openpi.yaml")))
    ev_ = E.BatchedEvaluator(cfg)
    try:
        ev_.load_batch({0: int(inst[0])})
        scope = ev_.instance_eval_states[0].env_accessor.object_scope
        slicer = scope[a.slicer]
        zero = th.zeros(3)
        wait, hold, cut = 75, 40, -1
        for step in range(a.steps):
            cur_step[0] = step
            target = scope.get(a.target)
            alive = target is not None and og.sim.stage.GetPrimAtPath(target.prim_path).IsValid()
            if not alive:
                if cut < 0:
                    cut = step
                    print(f"[particles] 스텝 {step}: {a.target} 잘림", flush=True)
                if step < cut + 3:
                    pos, _ = slicer.get_position_orientation()
                    slicer.set_position_orientation(position=pos + th.tensor([0.0, 0.0, 1.0]))
                    slicer.set_linear_velocity(zero)
                    slicer.set_angular_velocity(zero)
            else:
                tpos, _ = target.get_position_orientation()
                off = 0.8 if step < wait else 0.05
                slicer.set_position_orientation(position=tpos + th.tensor([0.0, 0.0, off]))
                slicer.set_linear_velocity(zero)
                slicer.set_angular_velocity(zero)
            ev_._step_fn([0])
            if cut >= 0 and step > cut + 5:
                break
        with open(os.path.join(a.out, "slice_events.json"), "w") as f:
            json.dump(rec, f)
        print(f"[particles] 기록 끝: 자르기 {len(rec['slice'])}, 새 물체 자세 {len(rec['setpose'])}, 다지기 {len(rec['dice'])}", flush=True)
    finally:
        td = getattr(og, "tempdir", None)
        if td and os.path.isdir(td) and os.path.basename(td).startswith("tmp"):
            shutil.rmtree(td, ignore_errors=True)
            print(f"[particles] 임시 장면 폴더 지움: {td}", flush=True)
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
