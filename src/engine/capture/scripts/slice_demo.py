"""층 0 정답: 판 도중 강체 삭제·추가(자르기) 한 번을 공식 평가기로 일으킨다 (physx_capture.py --script 로 부름, 문서 20.4).

chopping_wood 기준: 도끼(ax, 자르개)를 첫 통나무 위 공중에 매 스텝 순간이동(속도 0)시켜 SlicerActive 가 켜지게 기다린 뒤
(닿지 않고 2.0 초), 통나무에 겹치게 옮겨 SlicingRule 이 통나무를 지우고 반쪽 둘을 넣게 한다. 자른 뒤에는 도끼를 멀리 치운다.
조작은 OmniGibson API(set_position_orientation·set_velocity)로만 한다 -> PhysX 쓰기는 곁기록·OVD 에 남는다.
환경 변수: SLICE_SLICER(기본 ax.n.01_1), SLICE_TARGET(기본 log.n.01_1), SLICE_WAIT(기본 75 스텝), SLICE_HOLD(기본 40 스텝)
"""
import os

import torch as th

STATE = {"done": False, "cut_step": -1, "removed_seen": False}


def _obj(ev, key):
    import omnigibson as og

    scope = ev.env.task.object_scopes[0]  # 판 하나 (num_envs = 1). 값은 물체 자체(또는 감싼 것)
    ent = scope.get(key)
    obj = getattr(ent, "wrapped_obj", ent)
    if obj is None:
        return None
    try:  # 지워진 물체는 prim 이 사라진다
        return obj if og.sim.stage.GetPrimAtPath(obj.prim_path).IsValid() else None
    except Exception:
        return None


def on_before_apply(ev, step, cap):
    slicer_key = os.environ.get("SLICE_SLICER", "ax.n.01_1")
    target_key = os.environ.get("SLICE_TARGET", "log.n.01_1")
    wait = int(os.environ.get("SLICE_WAIT", "75"))
    hold = int(os.environ.get("SLICE_HOLD", "40"))
    if step == 0:
        keys = list(ev.env.task.object_scopes[0].keys())
        print(f"[slice] 스텝 0: 범위 이름 {keys}", flush=True)
    slicer = _obj(ev, slicer_key)
    target = _obj(ev, target_key)
    if slicer is None:
        return
    zero = th.zeros(3)
    if target is None:  # 잘렸다 (원본이 지워짐) -> 도끼를 멀리 치우고 끝
        if not STATE["removed_seen"]:
            STATE["removed_seen"] = True
            STATE["cut_step"] = step
            print(f"[slice] 스텝 {step}: {target_key} 가 없어짐 (자르기 일어남)", flush=True)
        if step < STATE["cut_step"] + 3:
            pos, _ = slicer.get_position_orientation()
            slicer.set_position_orientation(position=pos + th.tensor([0.0, 0.0, 1.0]))
            slicer.set_linear_velocity(zero)
            slicer.set_angular_velocity(zero)
        return
    tpos, _ = target.get_position_orientation()
    if step < wait:  # 대상 위 공중에서 기다림 (닿지 않음)
        slicer.set_position_orientation(position=tpos + th.tensor([0.0, 0.0, 0.8]))
        slicer.set_linear_velocity(zero)
        slicer.set_angular_velocity(zero)
    elif step < wait + hold:  # 대상에 겹치게
        slicer.set_position_orientation(position=tpos + th.tensor([0.0, 0.0, 0.05]))
        slicer.set_linear_velocity(zero)
        slicer.set_angular_velocity(zero)
        if step == wait:
            print(f"[slice] 스텝 {step}: 자르개를 대상에 겹침", flush=True)
