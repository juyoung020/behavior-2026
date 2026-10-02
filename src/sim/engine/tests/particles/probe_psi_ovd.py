"""층 0 탐침 (B9, 리드 부탁 09-30): psi.wake_up / put_to_sleep / load_state / 텐서 자세 쓰기가 PhysX 에 어떤 호출로 들어가고 OVD 에 남는지.
physx_capture.py --script 로 부른다(OVD·곁기록을 함께 뜸). 스텝마다 한 가지 조작만 한 물체에 해서, 뒤에 ovd_dump --trace <링크 이름> 으로
그 창(스텝 사이) OVD 명령을 본다. 조작 목록은 [probe] 줄 (스텝, 조작, 물체, 링크 prim 경로).
  bash run_probe_psi_ovd.sh
"""
import os

STATE = {"plan": None}


def on_before_apply(ev, step, cap):
    import omnigibson as og
    from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim

    scene = og.sim.scenes[0]
    if step < 60:
        return
    if STATE["plan"] is None:
        objs = sorted([o for o in scene.objects if not o.kinematic_only and o.is_asleep], key=lambda o: o.name)
        free = [o for o in objs if not o.articulated and o.n_joints == 0 and not o.fixed_base]
        art = [o for o in objs if o.articulated and o.n_joints > 0]
        F = lambda i: free[i] if i < len(free) else None
        A = art[0] if art else None
        STATE["plan"] = {
            60: ("psi.wake_up", F(0)),
            62: ("psi.put_to_sleep", F(0)),
            64: ("psi.wake_up(art)", A),
            66: ("psi.put_to_sleep(art)", A),
            68: ("load_state(잠, 같은 자세)", F(1)),
            70: ("set_position_orientation(같은 값)", F(2)),
            72: ("set_linear_velocity(0)", F(3)),
            74: ("psi.wake_up(깨어 있는 것)", F(0)),
        }
        for s, (k, o) in sorted(STATE["plan"].items()):
            if o is not None:
                links = [l.prim_path for l in o.links.values() if isinstance(l, RigidDynamicPrim)]
                print(f"[probe] 계획 스텝 {s}: {k} {o.name} 링크 {links} 관절체 {o.articulation_root_path if o.articulated else '-'}", flush=True)
    item = STATE["plan"].get(step)
    if item is None or item[1] is None:
        return
    k, o = item
    import torch as th

    before = bool(o.is_asleep)
    if k.startswith("psi.wake_up"):
        o.wake()
    elif k.startswith("psi.put_to_sleep"):
        o.sleep()
    elif k.startswith("load_state"):
        o.load_state(o.dump_state(serialized=False), serialized=False)
    elif k.startswith("set_position_orientation"):
        p, q = o.root_link.get_position_orientation()
        o.root_link.set_position_orientation(p, q)
    elif k.startswith("set_linear_velocity"):
        o.root_link.set_linear_velocity(th.zeros(3))
    print(f"[probe] 스텝 {step}: {k} {o.name} 잠 {before} -> {bool(o.is_asleep)}", flush=True)
