# 층 0(omni) 정답지: 공식 Robot 의 보조 잡기 메서드(robot.py)를 가짜 로봇 객체에 묶어 그대로 돌린다 (물리 없음).
# 손가락 접촉·광선 적중·링크 위치는 무작위 시나리오로 넣고, 매 서브스텝 판단(잡기 시도/놓기)과 팔 상태를 적는다.
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_ag_ref.py --out ~/engine-data/omni/ag
import argparse
import os
import struct
import types

import networkx as nx
import numpy as np
import torch as th

import omnigibson as og
import omnigibson.robots.robot as RR
from omnigibson.controllers import ControlType
from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
from omnigibson.robots.robot import Robot
from omnigibson.utils.constants import JointType

N_LINK = 12  # 후보 링크 풀 (물체 5 개 + 로봇 링크 1 + 등록 안 된 것 1)
LINK_OBJ = [0, 0, 1, 1, 2, 3, 3, 4, 4, -1, 5, 2]   # -1: 등록 안 됨
LINK_DYN = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
ROBOT_LINK = 10  # 로봇 자기 링크 (접촉 목록에서 걸러짐)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    og.sim = types.SimpleNamespace(get_physics_dt=lambda: 1.0 / 120)  # isaacsim PhysicsContext: 1 / timeStepsPerSecond

    arms = ["left", "right"]
    fingers = {arm: [types.SimpleNamespace(prim_path=f"/World/robot/{arm}_finger{i}") for i in range(2)] for arm in arms}
    paths = [f"/World/obj{LINK_OBJ[i]}/link{i}" if LINK_OBJ[i] >= 0 else f"/World/free/link{i}" for i in range(N_LINK)]
    paths[ROBOT_LINK] = "/World/robot/base_link"
    link_pos = {}

    class FakeDyn(RigidDynamicPrim):  # isinstance(RigidDynamicPrim) 통과용 (속성은 평범한 값으로 덮음)
        prim_path = None
        get_position_orientation = None

        def __init__(self):
            pass

    def mk_link(i):
        if LINK_DYN[i]:
            l = FakeDyn()
        else:
            l = types.SimpleNamespace()
        l.get_position_orientation = lambda i=i: (link_pos[i], th.tensor([0, 0, 0, 1.0]))
        l.prim_path = paths[i]
        return l

    objs = {}
    for i in range(N_LINK):
        o = LINK_OBJ[i]
        if o < 0 or i == ROBOT_LINK:
            continue
        objs.setdefault(o, types.SimpleNamespace(name=f"obj{o}", links={}, prim_path=f"/World/obj{o}"))
        objs[o].links[f"link{i}"] = mk_link(i)
    by_path = {v.prim_path: v for v in objs.values()}

    eef_pos = {arm: th.zeros(3) for arm in arms}
    control = {arm: th.zeros(2) for arm in arms}
    upper = th.full((28,), 0.05)
    events = []
    scen = {}

    fake = types.SimpleNamespace()
    fake.is_manipulation = True
    fake.arm_names = arms
    fake.default_arm = "left"
    fake.grasping_mode = "assisted"
    fake._grasping_direction = "lower"
    fake._controllers = {f"gripper_{arm}": (arm, 0) for arm in arms}
    fake.joint_upper_limits = upper
    fake._ag_obj_in_hand = {arm: None for arm in arms}
    fake._ag_obj_constraints = {arm: None for arm in arms}
    fake._ag_obj_constraint_params = {arm: None for arm in arms}
    fake._ag_release_counter = {arm: None for arm in arms}
    fake._ag_grasp_counter = {arm: None for arm in arms}
    fake.finger_links = fingers
    fake.link_prim_paths = ["/World/robot/base_link"] + [f.prim_path for arm in arms for f in fingers[arm]]
    fake.eef_links = {arm: types.SimpleNamespace(get_position_orientation=lambda arm=arm: (eef_pos[arm], th.tensor([0, 0, 0, 1.0])))
                      for arm in arms}
    fake.scene = types.SimpleNamespace(object_registry=lambda key, val, default=None: by_path.get(val, default))

    # _find_gripper_contacts 는 RigidContactAPI 를 부르므로 결과를 시나리오로 대신 (반환 형식 그대로)
    def fake_contacts(arm="default"):
        s = scen[arm]
        link_paths = set(fake.link_prim_paths)
        data, rcl = set(), {}
        for i in s["contact"]:
            if paths[i] in link_paths:
                continue
            data.add(paths[i])
            rcl.setdefault(paths[i], set()).update(s["touch"][i])
        return data, rcl

    def fake_ray(arm="default"):
        return {paths[i] for i in scen[arm]["ray"]}

    def fake_release(arm="default"):
        events.append(("release", arm, -1))
        fake._ag_obj_constraints[arm] = None
        fake._ag_obj_constraint_params[arm] = None
        fake._ag_release_counter[arm] = 0

    def fake_maybe(target_obj, target_link_name, arm):
        li = int(target_link_name[4:])
        events.append(("try", arm, li))
        if scen[arm]["success"]:
            fake._ag_obj_constraints[arm] = object()
            fake._ag_obj_in_hand[arm] = target_obj
            fake._ag_obj_constraint_params[arm] = {}

    fake._find_gripper_contacts = fake_contacts
    fake._find_gripper_raycast_collisions = fake_ray
    fake._release_grasp = fake_release
    fake._maybe_establish_grasp = fake_maybe
    fake._calculate_in_hand_object = lambda arm="default": Robot._calculate_in_hand_object(fake, arm=arm)
    fake._handle_release_window = lambda arm="default": Robot._handle_release_window(fake, arm=arm)

    # ControllerView: 손가락 제어 (마지막 위치 목표) 와 제어 종류
    RR.ControllerView = types.SimpleNamespace(
        get_dof_idx=lambda g: th.tensor([24, 25]) if g == "left" else th.tensor([26, 27]),
        get_control=lambda g, i: control[g],
        get_control_type=lambda g: ControlType.POSITION,
    )

    apply_state = {arm: False for arm in arms}
    contact_state = {arm: set() for arm in arms}
    ray_state = {arm: set() for arm in arms}
    touch_state = {}
    with open(os.path.join(a.out, "ag.bin"), "wb") as f:
        f.write(struct.pack("<iii", a.steps, N_LINK, ROBOT_LINK))
        f.write(np.array(LINK_OBJ, np.int32).tobytes())
        f.write(np.array(LINK_DYN, np.uint8).tobytes())
        for step in range(a.steps):
            for arm in arms:
                if rng.random() < 0.01:
                    apply_state[arm] = not apply_state[arm]
                # 손가락 위치 목표: 잡는 중이면 상한보다 작게, 아니면 상한 (가끔 정확히 상한/살짝 아래)
                if apply_state[arm]:
                    control[arm] = th.tensor([rng.choice([0.0, 0.01, 0.049999]), 0.05], dtype=th.float32)
                else:
                    control[arm] = th.tensor([0.05, 0.05], dtype=th.float32)
                # 접촉·광선·손가락 접촉은 한동안 유지되다 가끔 바뀐다 (잡기 창 0.3 초 = 36 서브스텝을 채우는 경우가 나오게)
                if rng.random() < 0.02 or arm not in touch_state:
                    contact_state[arm] = set(rng.choice(N_LINK, rng.integers(0, 4), replace=False).tolist())
                    ray_state[arm] = {i for i in contact_state[arm] if rng.random() < 0.8} | (
                        {int(rng.integers(0, N_LINK))} if rng.random() < 0.3 else set())
                    touch_state[arm] = {}
                    for i in range(N_LINK):
                        r = rng.random()
                        fs = fingers[arm]
                        t = {fs[0].prim_path, fs[1].prim_path} if r < 0.7 else ({fs[0].prim_path} if r < 0.9 else set())
                        if rng.random() < 0.1:
                            t.add(fingers["right" if arm == "left" else "left"][0].prim_path)
                        touch_state[arm][i] = t
                if rng.random() < 0.01 and ray_state[arm]:
                    ray_state[arm] = set(list(ray_state[arm])[1:])
                ray = set(ray_state[arm])
                touch = touch_state[arm]
                scen[arm] = {"contact": contact_state[arm], "ray": ray, "touch": touch, "success": rng.random() < 0.7}
                eef_pos[arm] = th.tensor(rng.standard_normal(3), dtype=th.float32)
            for i in range(N_LINK):
                link_pos[i] = th.tensor(rng.standard_normal(3), dtype=th.float32)
            events.clear()
            Robot._handle_assisted_grasping(fake)
            # 기록: 팔마다 입력 + 사건 + 상태
            for arm in arms:
                s = scen[arm]
                f.write(struct.pack("<B", 1 if bool(th.any(control[arm] < upper[[24, 25]])) else 0))
                f.write(np.array(eef_pos[arm].numpy(), np.float32).tobytes())
                cmask = np.zeros(N_LINK, np.uint8)
                for i in s["contact"]:
                    cmask[i] = 1
                rmask = np.zeros(N_LINK, np.uint8)
                for i in s["ray"]:
                    rmask[i] = 1
                nf = np.array([len(s["touch"][i] & {x.prim_path for x in fingers[arm]}) for i in range(N_LINK)], np.uint8)
                f.write(cmask.tobytes())
                f.write(rmask.tobytes())
                f.write(nf.tobytes())
                f.write(np.stack([link_pos[i].numpy() for i in range(N_LINK)]).astype(np.float32).tobytes())
                f.write(struct.pack("<B", 1 if s["success"] else 0))
                ev = [e for e in events if e[1] == arm]
                et, tl = (0, -1) if not ev else ((1, -1) if ev[0][0] == "release" else (2, ev[0][2]))
                inh = fake._ag_obj_in_hand[arm]
                inh_id = -1 if inh is None else int(inh.name[3:])
                rc = fake._ag_release_counter[arm]
                gc = fake._ag_grasp_counter[arm]
                f.write(struct.pack("<Bi iii", et, tl, inh_id, -1 if rc is None else rc, -1 if gc is None else gc))

    # 관절 종류 (_get_assisted_grasp_joint_type)
    with open(os.path.join(a.out, "jt.bin"), "wb") as f:
        n = 5000
        f.write(struct.pack("<i", n))
        for _ in range(n):
            mass = float(np.float32(rng.choice([rng.uniform(0, 20), 10.0, np.nextafter(np.float32(10), np.float32(11))])))
            fixed = bool(rng.random() < 0.4)
            # 링크 사슬: root - l1 - l2, 관절 종류 무작위
            g = nx.DiGraph()
            jt = [JointType.JOINT_FIXED if rng.random() < 0.5 else JointType.JOINT_REVOLUTE for _ in range(2)]
            g.add_edge("root", "l1", joint_type=jt[0])
            g.add_edge("l1", "l2", joint_type=jt[1])
            name = str(rng.choice(["root", "l1", "l2"]))
            tobj = types.SimpleNamespace(links={name: types.SimpleNamespace(mass=mass)}, fixed_base=fixed, root_link_name="root",
                                         articulation_tree=g)
            r = Robot._get_assisted_grasp_joint_type(fake, tobj, name)
            anc = {"root": False, "l1": jt[0] != JointType.JOINT_FIXED,
                   "l2": jt[1] != JointType.JOINT_FIXED or jt[0] != JointType.JOINT_FIXED}[name]
            code = 0 if r is None else (1 if r == "FixedJoint" else 2)
            f.write(struct.pack("<fBBBB", mass, fixed, name == "root", anc, code))
    print("done")


if __name__ == "__main__":
    main()
