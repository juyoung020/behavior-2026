"""엔진 상태 -> 평가기 관측·trace 값 (OmniGibson 과 같은 식을 같은 torch 함수로).

물리 상태(관절값·링크 자세)는 엔진(engine_core.EngineCore)에서, 그 뒤 가공은 OmniGibson 원본 식을 그대로 옮겨 부른다
(robots/robot.py:1541 _get_proprioception_dict, :3056 get_relative_eef_pose, :1605 _get_base_qvel_for_proprioception,
eval/utils/eval_utils.py:121 PROPRIOCEPTION_INDICES, evaluator.py:576 _preprocess_obs). transform_utils 는 원본 모듈을 import 한다.
"""
from __future__ import annotations

import ast
import json
import os

import numpy as np
import torch as th

PROPRIO_R1PRO = ["base_qvel", "arm_left_qpos", "arm_left_qvel", "eef_left_pos", "eef_left_quat", "gripper_left_qpos",
                 "gripper_left_qvel", "arm_right_qpos", "arm_right_qvel", "eef_right_pos", "eef_right_quat",
                 "gripper_right_qpos", "gripper_right_qvel", "trunk_qpos", "trunk_qvel"]


def read_s1_groups(rec_dir):
    g = {}
    for line in open(os.path.join(rec_dir, "s1_setup.txt")):
        k = line.split()
        if k and k[0] == "group":
            g[k[1]] = [int(x) for x in k[2:]]
    return g


class RobotView:
    """한 판의 로봇: 엔진 몸체 경로와 dof 묶음."""

    def __init__(self, rec_dir, env_idx=0):
        sc = json.load(open(os.path.join(rec_dir, "scope.json")))[env_idx]
        r = sc["robot"]
        self.name = r["name"]
        self.prim = r["prim_path"]
        self.base_link = r["base_footprint_link"]
        self.root_link = r["root_link"]
        links = r["links"] if isinstance(r["links"], dict) else ast.literal_eval(r["links"])
        self.links = links
        self.eef = {arm: links[n] for arm, n in r["eef_link_names"].items()}
        sensors = r.get("sensors")
        self.sensors = sensors if isinstance(sensors, dict) else ast.literal_eval(sensors or "{}")
        g = read_s1_groups(rec_dir)
        self.base_idx = list(range(6))  # 바닥 가상 관절 6 개 (x, y, z, rx, ry, rz) = dof 0..5
        self.base_control_idx = g["base"]
        self.trunk_idx = g["trunk"]
        self.arm_idx = {"left": g["arm_left"], "right": g["arm_right"]}
        self.grip_idx = {"left": g["gripper_left"], "right": g["gripper_right"]}
        self.objects = {k: v for k, v in sc["objects"].items()}


def _t(x):
    return th.as_tensor(np.asarray(x, np.float32))


def base_pose(eng, rv):
    p = eng.body_pose(rv.base_link)
    return _t(p[:3]), _t(p[3:])


def proprio(eng, rv, T):
    q, v = eng.joint_state()
    jp, jv = _t(q), _t(v)
    pos, quat = base_pose(eng, rv)
    d = {}
    for arm in ("left", "right"):
        d[f"arm_{arm}_qpos"] = jp[rv.arm_idx[arm]]
        d[f"arm_{arm}_qvel"] = jv[rv.arm_idx[arm]]
        e = eng.body_pose(rv.eef[arm])
        ep, eq = T.relative_pose_transform(_t(e[:3]), _t(e[3:]), pos, quat)
        d[f"eef_{arm}_pos"], d[f"eef_{arm}_quat"] = ep, eq
        d[f"gripper_{arm}_qpos"] = jp[rv.grip_idx[arm]]
        d[f"gripper_{arm}_qvel"] = jv[rv.grip_idx[arm]]
    d["trunk_qpos"] = jp[rv.trunk_idx]
    d["trunk_qvel"] = jv[rv.trunk_idx]
    base_qvel = jv[rv.base_control_idx]
    yaw = jp[rv.base_idx][5]
    c, s = th.cos(yaw), th.sin(yaw)
    d["base_qvel"] = th.stack([c * base_qvel[0] + s * base_qvel[1], -s * base_qvel[0] + c * base_qvel[1], base_qvel[2]])
    return th.cat([d[k] for k in PROPRIO_R1PRO]), (jp, jv, pos, quat)


# ---------------- 카메라 자세 (evaluator.py:576 _preprocess_obs 의 cam_rel_poses) ----------------
# 공식: camera.get_position_orientation() = fabric 세계 행렬(double) -> RemoveScaleShear().ExtractRotationQuat() / ExtractTranslation() -> float32
# (utils/usd_utils.py:2138 get_world_pose). fabric 세계 행렬 = 카메라 local(double, scope.json camera_chain) @ 링크 세계 행렬.
# 링크 세계 행렬은 PhysX 자세(float32)에서 pxr GfMatrix4d::SetRotate 식(double)으로 만든다 — fabric 의 계층 재계산과는 double 마지막 자리에서
# 다를 수 있으나 float32 로 내릴 때 가려진다(추정, trace 비교로 확인).
def _rot_pxr(x, y, z, r):
    m = np.zeros((3, 3))
    m[0][0] = 1.0 - 2.0 * (y * y + z * z)
    m[0][1] = 2.0 * (x * y + z * r)
    m[0][2] = 2.0 * (z * x - y * r)
    m[1][0] = 2.0 * (x * y - z * r)
    m[1][1] = 1.0 - 2.0 * (z * z + x * x)
    m[1][2] = 2.0 * (y * z + x * r)
    m[2][0] = 2.0 * (z * x + y * r)
    m[2][1] = 2.0 * (y * z - x * r)
    m[2][2] = 1.0 - 2.0 * (y * y + x * x)
    return m


def _mat_to_quat_xyzw(R):
    """정규직교 3x3 (행벡터 규약, 행 = 축) -> 쿼터니언 (x,y,z,w), double. pxr GfRotation 과 같은 Shepperd 분기 (추정)."""
    # 행벡터 규약 행렬의 전치가 열벡터 회전 행렬
    m = R.T
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    if q[3] < 0:  # pxr 는 부호를 정하지 않는다 -> 비교에서 확인
        pass
    return q


def camera_world(eng, cam):
    """cam = scope.json camera_chain[이름] -> (pos float32 (3,), quat xyzw float32 (4,))."""
    p = eng.body_pose(cam["link"]).astype(np.float64)
    W = np.eye(4)
    W[:3, :3] = _rot_pxr(p[3], p[4], p[5], p[6])
    W[3, :3] = p[:3]
    M = W
    for c in cam["chain"]:
        M = np.array(c["local"]).reshape(4, 4) @ M
    R = M[:3, :3].copy()
    R /= np.linalg.norm(R, axis=1, keepdims=True)  # RemoveScaleShear (정규직교에 가까운 행렬: 행 길이만 정리, 추정)
    q = _mat_to_quat_xyzw(R)
    return th.as_tensor(M[3, :3].astype(np.float32)), th.as_tensor(q.astype(np.float32))


def cam_rel_poses(eng, rv, cams, order, T):
    pos, quat = base_pose(eng, rv)
    out = []
    for name in order:
        cp, cq = camera_world(eng, cams[name])
        out.append(th.cat(T.relative_pose_transform(cp, cq, pos, quat)))
    return th.cat(out)


def load_usdrt_gf():
    """공식이 쓰는 usdrt.Gf(_Gf.so) 를 Kit 없이 불러온다. LD_LIBRARY_PATH 에 libpython·omni.usd.libs/bin·usdrt.scenegraph/bin 이 있어야 한다
    (src/engine/eval/run_ported_engine.sh). 없으면 None."""
    import glob
    import sys

    E = os.path.join(os.path.dirname(th.__file__), "..", "isaacsim", "extscache")
    try:
        u = glob.glob(os.path.join(E, "omni.usd.libs-*"))[0]
        d = glob.glob(os.path.join(E, "usdrt.scenegraph-*"))[0]
        for p in (u, d):
            if p not in sys.path:
                sys.path.insert(0, p)
        from usdrt import Gf

        return Gf
    except Exception:
        return None


def camera_world_gf(eng, cam, Gf):
    """camera_world 과 같되 RemoveScaleShear·ExtractRotationQuat 를 공식 usdrt.Gf 로 (usd_utils.py:2148)."""
    p = eng.body_pose(cam["link"]).astype(np.float64)
    W = np.eye(4)
    W[:3, :3] = _rot_pxr(p[3], p[4], p[5], p[6])
    W[3, :3] = p[:3]
    M = W
    for c in cam["chain"]:
        M = np.array(c["local"]).reshape(4, 4) @ M
    m = Gf.Matrix4d(*[float(v) for v in M.reshape(-1)])
    q = m.RemoveScaleShear().ExtractRotationQuat()
    im = q.GetImaginary()
    pos = th.tensor(m.ExtractTranslation(), dtype=th.float32)
    quat = th.tensor([im[0], im[1], im[2], q.GetReal()], dtype=th.float32)
    return pos, quat


def cam_rel_poses_gf(eng, rv, cams, order, T, Gf):
    pos, quat = base_pose(eng, rv)
    out = []
    for name in order:
        cp, cq = camera_world_gf(eng, cams[name], Gf)
        out.append(th.cat(T.relative_pose_transform(cp, cq, pos, quat)))
    return th.cat(out)
