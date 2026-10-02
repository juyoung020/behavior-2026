# 층 0(omni) 정답지: 공식 OmniGibson 제어기 클래스를 물리 없이 직접 만들어(평가기 r1pro.yaml 설정 그대로) 무작위 입력을 넣고
# 목표·드라이브 목표를 적는다. 물리 상태(바닥·뿌리 자세, 관절 위치)는 ControllableObjectViewAPI 를 가짜로 바꿔 넣는다.
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_ctrl_ref.py --out ~/engine-data/omni/ctrl
# 출력: config.bin(자유도 배치·한계), func.bin(numba/numpy 변환 함수 단위), ctrl.bin(제어기 한 스텝씩), wrap.bin(rz 감기)
import argparse
import math
import os
import struct

import numpy as np
import torch as th

import omnigibson.utils.transform_utils_np as NT
from omnigibson.controllers.holonomic_base_joint_controller import HolonomicBaseJointController
from omnigibson.controllers.joint_controller import JointController
from omnigibson.controllers.multi_finger_gripper_controller import MultiFingerGripperController
from omnigibson.utils.geometry_utils import wrap_angle
from omnigibson.utils.usd_utils import ControllableObjectViewAPI
from omnigibson.utils import backend_utils as _bu

# 평가기와 같은 numpy 제어기 백엔드 (simulator.py:394, macros.py:152 USE_NUMPY_CONTROLLER_BACKEND = True)
_bu._compute_backend.set_methods_from_backend(_bu._ComputeNumpyBackend)

N_DOF = 28
BASE = [0, 1, 5]
TRUNK = [6, 7, 8, 9]
ARM = [[10, 11, 12, 13, 14, 15, 16], [17, 18, 19, 20, 21, 22, 23]]
GRIP = [[24, 25], [26, 27]]

# R1Pro URDF 한계 (datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf). 가상 base 관절은 한계 없음.
TRUNK_LIM = [(-1.1345, 1.8326, 2.5), (-2.7925, 2.5307, 2.5), (-1.8326, 1.5708, 2.5), (-3.0543, 3.0543, 2.5)]
ARM_LIM = [
    [(-4.4506, 1.3090, 7.1209), (-0.1745, 3.1416, 7.1209), (-2.356196, 2.356196, 8.3776), (-2.0944, 0.3491, 8.3776),
     (-2.356196, 2.356196, 10.4720), (-1.047198, 1.047198, 10.4720), (-1.5708, 1.5708, 10.4720)],
    [(-4.4506, 1.3090, 7.1209), (-3.1416, 0.1745, 7.1209), (-2.356196, 2.356196, 8.3776), (-2.0944, 0.3491, 8.3776),
     (-2.356196, 2.356196, 10.4720), (-1.047198, 1.047198, 10.4720), (-1.5708, 1.5708, 10.4720)],
]


def build_limits():
    lo = np.full(N_DOF, -1e4, np.float32)
    hi = np.full(N_DOF, 1e4, np.float32)
    vhi = np.zeros(N_DOF, np.float32)
    has = np.zeros(N_DOF, bool)
    for i in range(6):  # 가상 base: 선속도 1.5, 각속도 pi (robot.py:1272)
        vhi[i] = 1.5 if i < 3 else np.float32(th.pi)
    for k, d in enumerate(TRUNK):
        lo[d], hi[d], vhi[d] = TRUNK_LIM[k]
        has[d] = True
    for a in range(2):
        for k, d in enumerate(ARM[a]):
            lo[d], hi[d], vhi[d] = ARM_LIM[a][k]
            has[d] = True
        for d in GRIP[a]:
            lo[d], hi[d], vhi[d] = 0.0, 0.05, 0.25
            has[d] = True
    return {
        "position": (th.from_numpy(lo), th.from_numpy(hi)),
        "velocity": (th.from_numpy(-vhi), th.from_numpy(vhi)),
        "effort": (th.from_numpy(np.full(N_DOF, -100, np.float32)), th.from_numpy(np.full(N_DOF, 100, np.float32))),
        "has_limit": th.from_numpy(has),
    }


class FakeView:
    """ControllableObjectViewAPI 대신: 판(멤버)마다 자세·관절 위치를 주고, 드라이브 목표 쓰기를 받아 적는다."""

    def __init__(self, n):
        self.n = n
        self.paths = [f"/World/scene_{i}/controllable__r1pro__robot_r1" for i in range(n)]
        self.base = {}
        self.root = {}
        self.q = np.zeros((n, N_DOF), np.float32)
        self.qd = np.zeros((n, N_DOF), np.float32)
        self.pos_t = np.full((n, N_DOF), np.nan, np.float32)
        self.vel_t = np.full((n, N_DOF), np.nan, np.float32)

    def idx(self, p):
        return self.paths.index(p)

    def install(self):
        V = self
        ControllableObjectViewAPI.get_position_orientation = classmethod(lambda cls, p: V.base[p])
        ControllableObjectViewAPI.get_root_position_orientation = classmethod(lambda cls, p: V.root[p])
        ControllableObjectViewAPI.get_joint_positions = classmethod(lambda cls, p: V.q[V.idx(p)].copy())
        ControllableObjectViewAPI.get_all_joint_positions = classmethod(lambda cls, p: V.q.copy())
        ControllableObjectViewAPI.get_all_joint_velocities = classmethod(lambda cls, p, estimate=False: V.qd.copy())
        ControllableObjectViewAPI.get_member_view_indices = classmethod(lambda cls, r, ps: [V.idx(x) for x in ps])

        def set_pos(cls, r, rows, controls, dof_idx):
            for i, row in enumerate(rows):
                V.pos_t[row, dof_idx] = controls[i]

        def set_vel(cls, r, rows, controls, dof_idx):
            for i, row in enumerate(rows):
                V.vel_t[row, dof_idx] = controls[i]

        ControllableObjectViewAPI.set_all_joint_position_targets = classmethod(set_pos)
        ControllableObjectViewAPI.set_all_joint_velocity_targets = classmethod(set_vel)


def rand_quat(rng, tilt):
    yaw = rng.uniform(-4, 4)
    ax = rng.standard_normal(3) * tilt
    q = np.array([ax[0], ax[1], math.sin(yaw / 2), math.cos(yaw / 2)])
    if rng.random() < 0.1:
        q = rng.standard_normal(4)
    q = q / np.linalg.norm(q)
    return q.astype(np.float32)  # float32 로 바꾼 뒤라 단위에서 끝비트만큼 벗어남 (PhysX 가 주는 값과 같은 성격)


def special(rng, x):
    r = rng.random()
    if r < 0.02:
        return np.float32(0.0)
    if r < 0.04:
        return np.float32(-0.0)
    if r < 0.05:
        return np.float32(np.nan)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    lim = build_limits()

    # ---- config.bin: n_dof, 자유도 배치, 한계
    with open(os.path.join(a.out, "config.bin"), "wb") as f:
        f.write(struct.pack("<i", N_DOF))
        f.write(np.array(BASE + TRUNK + ARM[0] + ARM[1] + GRIP[0] + GRIP[1], np.int32).tobytes())
        for arr in (lim["position"][0], lim["position"][1], lim["velocity"][0], lim["velocity"][1]):
            f.write(arr.numpy().astype(np.float32).tobytes())
        f.write(lim["has_limit"].numpy().astype(np.uint8).tobytes())

    # ---- func.bin: pose2mat / pose_inv / 4x4 matmul 함수 단위
    nf = 20000
    with open(os.path.join(a.out, "func.bin"), "wb") as f:
        f.write(struct.pack("<i", nf))
        for _ in range(nf):
            p1 = (rng.standard_normal(3) * 3).astype(np.float32)
            q1 = rand_quat(rng, 0.3)
            p2 = (rng.standard_normal(3) * 3).astype(np.float32)
            q2 = rand_quat(rng, 0.3)
            M1 = NT.pose2mat((p1, q1))
            M2 = NT.pose2mat((p2, q2))
            I1 = NT.pose_inv(M1)
            P = I1 @ M2
            for x in (p1, q1, p2, q2, M1, M2, I1, P):
                f.write(np.ascontiguousarray(x, np.float32).tobytes())

    # ---- 제어기: 평가기 r1pro.yaml 설정 (+ robot.py 기본값 병합 결과)
    base = HolonomicBaseJointController(
        control_freq=30, motor_type="velocity", control_limits=lim, dof_idx=th.tensor(BASE),
        command_input_limits=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        command_output_limits=[[-0.75, -0.75, -1.0], [0.75, 0.75, 1.0]], vel_kp=150, use_impedances=False)
    trunk = JointController(control_freq=30, motor_type="position", control_limits=lim, dof_idx=th.tensor(TRUNK),
                            command_input_limits=None, command_output_limits=None, pos_kp=150,
                            use_impedances=False, use_delta_commands=False)
    arms = [JointController(control_freq=30, motor_type="position", control_limits=lim, dof_idx=th.tensor(ARM[i]),
                            command_input_limits=None, command_output_limits=None, pos_kp=150,
                            use_impedances=False, use_delta_commands=False) for i in range(2)]
    grips = [MultiFingerGripperController(control_freq=30, motor_type="position", control_limits=lim,
                                          dof_idx=th.tensor(GRIP[i]), command_input_limits="default",
                                          command_output_limits="default", mode="smooth", limit_tolerance=0.001,
                                          inverted=False) for i in range(2)]
    order = [base, trunk, arms[0], grips[0], arms[1], grips[1]]  # raw_controller_order
    V = FakeView(a.members)
    V.install()
    for c in order:
        for p in V.paths:
            c.add_member(p)
    dims = [3, 4, 7, 1, 7, 1]
    lo = lim["position"][0].numpy()
    hi = lim["position"][1].numpy()
    with open(os.path.join(a.out, "ctrl.bin"), "wb") as f:
        f.write(struct.pack("<ii", a.members, a.steps))
        for step in range(a.steps):
            upd = np.zeros(a.members, np.uint8)
            acts = np.zeros((a.members, 23), np.float32)
            for m, p in enumerate(V.paths):
                V.base[p] = ((rng.standard_normal(3) * 5).astype(np.float32), rand_quat(rng, 0.05))
                V.root[p] = ((rng.standard_normal(3) * 5).astype(np.float32), rand_quat(rng, 0.01))
                V.q[m] = (lo + (hi - lo) * rng.random(N_DOF)).astype(np.float32)
                V.q[m, :6] = (rng.standard_normal(6)).astype(np.float32)
                # 처음 몇 스텝은 일부 판에 행동을 안 줘서 no-op 목표 경로를 탄다
                if step < 3 and m % 2 == 1:
                    continue
                if step >= 3 and rng.random() < 0.1:
                    continue  # 행동 없는 스텝: 이전 목표 유지
                upd[m] = 1
                act = np.zeros(23, np.float32)
                act[0:3] = rng.uniform(-1.4, 1.4, 3)
                seg = [(3, TRUNK), (7, ARM[0]), (15, ARM[1])]
                for off, dof in seg:
                    for k, d in enumerate(dof):
                        span = hi[d] - lo[d]
                        act[off + k] = lo[d] - 0.2 * span + 1.4 * span * rng.random()
                        if rng.random() < 0.03:
                            act[off + k] = rng.choice([lo[d], hi[d]])
                act[14] = rng.uniform(-1.3, 1.3)
                act[22] = rng.uniform(-1.3, 1.3)
                for k in range(23):
                    act[k] = special(rng, act[k])
                acts[m] = act
                idx = 0
                for c, dim in zip(order, dims):
                    c.update_goal(m, act[idx: idx + dim])  # ControllerView.update_goal: cb.from_torch(th) = numpy float32
                    idx += dim
            V.pos_t[:] = np.nan
            V.vel_t[:] = np.nan
            for c in order:  # ControllerView.step_all
                c.step()
            for m, p in enumerate(V.paths):
                bp, bq = V.base[p]
                rp, rq = V.root[p]
                f.write(struct.pack("<B", upd[m]))
                for x in (acts[m], bp, bq, rp, rq, V.q[m], V.pos_t[m], V.vel_t[m]):
                    f.write(np.ascontiguousarray(x, np.float32).tobytes())

    # ---- rz 감기 (robot.py:767~775 그대로, torch float32)
    xs = np.concatenate([rng.uniform(-20, 20, 30000), rng.uniform(-3.3, 3.3, 30000),
                         np.array([math.pi, -math.pi, np.float32(math.pi), -np.float32(math.pi),
                                   np.nextafter(np.float32(math.pi), np.float32(10)), 3 * math.pi, 0.0, -0.0])]
                        ).astype(np.float32)
    with open(os.path.join(a.out, "wrap.bin"), "wb") as f:
        f.write(struct.pack("<i", len(xs)))
        for x in xs:
            j_pos = th.tensor([x], dtype=th.float32)
            wrapped = 0
            if j_pos < -math.pi or j_pos > math.pi:
                j_pos = wrap_angle(j_pos)
                wrapped = 1
            f.write(struct.pack("<fBf", float(x), wrapped, float(j_pos[0])))
    print("done", a.members, a.steps)


if __name__ == "__main__":
    main()
