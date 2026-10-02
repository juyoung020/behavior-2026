"""층 0 탐침 (base 창 = 상태 뜨기·되돌리기, B9): 공식 평가기(무수정)에서 텐서 쓰기(RigidPrimView/ArticulationView)와 psi(put_to_sleep/wake_up)가
PhysX 에 언제 닿는지 잰다. 잠든 물체마다 다른 실험을 하나씩 하고 psi.is_sleeping 을 (a) 호출 직후 (b) 다음 환경 스텝 뒤 에 읽는다.
  텐서 쓰기가 즉시 + 자동 깨움이면: 쓰기 직후 깨어 있음.  다음 simulate 직전에 모여 닿으면: 직후 잠, 스텝 뒤 깨어 있음.
실험 (물체마다 하나):
  W      자세 쓰기만 (root_link.set_position_orientation 같은 값)
  V      속도 쓰기만 (0)
  WS     자세·속도 쓰기 → sleep()        (EntityPrim._load_state 차례)
  SW     sleep() → 자세 쓰기
  L      obj.load_state(obj.dump_state())  (전체)
  AJ     관절체: 관절 위치·속도 쓰기만
  AL     관절체: load_state(dump_state())
  bash run_probe_load_order.sh [과제=chopping_wood] [먼저 돌 스텝=60]
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
    ap.add_argument("--task", default="chopping_wood")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--settle", type=int, default=60)
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
    from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
    from physx_capture import install_no_render

    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    install_no_render()
    seed = seed_everything(DEFAULT_EVAL_SEED)
    inst = resolve_instance_ids(a.task, [a.instance], mode="public_test")
    cfg = OmegaConf.create(dict(
        env_wrapper={"_target_": "omnigibson.eval.wrappers.DefaultWrapper"}, policy_name="local",
        model={"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None}, headless=True,
        partial_scene_load=True, max_steps=a.settle + 20, write_video=False, mode="public_test", seed=seed, num_envs=1,
        task={"name": a.task}, robot=OmegaConf.load("/mnt/c/behavior-2026/src/sim/configs/r1pro_openpi.yaml")))
    ev_ = E.BatchedEvaluator(cfg)
    res = {"exp": []}
    try:
        ev_.load_batch({0: int(inst[0])})
        for _ in range(a.settle):
            ev_._step_fn([0])
        scene = og.sim.scenes[0]

        def asleep_links(o):
            return [bool(l.is_asleep) for l in o.links.values() if isinstance(l, RigidDynamicPrim)]

        objs = sorted([o for o in scene.objects if not o.kinematic_only and o.is_asleep], key=lambda o: o.name)
        free = [o for o in objs if o.n_joints == 0 and not o.fixed_base]
        art = [o for o in objs if o.n_joints > 0]
        print(f"[probe] 잠든 물체 {len(objs)} (자유 강체 {len(free)}, 관절체 {len(art)})", flush=True)
        z = th.zeros(3)
        plan = []

        def W(o):
            p, q = o.root_link.get_position_orientation()
            o.root_link.set_position_orientation(p, q)

        def V(o):
            o.root_link.set_linear_velocity(z)
            o.root_link.set_angular_velocity(z)

        exps = {
            "W": lambda o: [W(o)],
            "V": lambda o: [V(o)],
            "WS": lambda o: [W(o), V(o), o.sleep()],
            "SW": lambda o: [o.sleep(), W(o)],
            "L": lambda o: [o.load_state(o.dump_state(serialized=False), serialized=False)],
        }
        for i, k in enumerate(exps):
            if i < len(free):
                plan.append((k, free[i]))
        if art:
            def AJ(o):
                o.set_joint_positions(o.get_joint_positions())
                o.set_joint_velocities(o.get_joint_velocities())
            plan.append(("AJ", art[0]))
            if len(art) > 1:
                plan.append(("AL", art[1]))
            exps["AJ"] = lambda o: [AJ(o)]
            exps["AL"] = exps["L"]
        for k, o in plan:
            before = o.is_asleep
            exps[k](o)
            res["exp"].append(dict(exp=k, name=o.name, articulated=bool(o.articulated), fixed=bool(o.fixed_base), before=bool(before),
                                   right_after=bool(o.is_asleep), links_after=asleep_links(o)))
        ev_._step_fn([0])
        for r, (k, o) in zip(res["exp"], plan):
            r["after_step"] = bool(o.is_asleep)
            print(f"[probe] {r['exp']:3s} {r['name']:40s} 관절체 {r['articulated']} 고정 {r['fixed']}: 전 {r['before']} 직후 {r['right_after']} 스텝 뒤 {r['after_step']}",
                  flush=True)
        with open(os.path.join(a.out, "probe.json"), "w") as f:
            json.dump(res, f, indent=1)
    finally:
        td = getattr(og, "tempdir", None)
        if td and os.path.isdir(td) and os.path.basename(td).startswith("tmp"):
            shutil.rmtree(td, ignore_errors=True)
            print(f"[probe] 임시 장면 폴더 지움: {td}", flush=True)
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
