"""core/omni/obs.h (C++ 관측 가공) vs 공식 trace: radio500_h 501 스텝 proprio 61·cam_rel_poses 21 비트 대조.
엔진(engine_core, 우리 PhysX 비계 = 공식과 물리 비트 동일)에서 관절값·링크 자세를 읽어 C++ 로 가공한다. 파이썬은 시험 틀일 뿐.
    WSL: bash /mnt/c/behavior-2026/src/engine/tests/omni/run_obs.sh
"""
import ctypes as C
import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/c/behavior-2026/src/engine/eval")
from engine_core import EngineCore  # noqa: E402
from obs_engine import RobotView  # noqa: E402

L = C.CDLL(sys.argv[1])
F = C.POINTER(C.c_float)
D = C.POINTER(C.c_double)
I = C.POINTER(C.c_int32)
L.obs_proprio.argtypes = [I, F, F, F, F, F, F]
L.obs_cam.argtypes = [D, C.c_int, F, F, F, F]


def fp(a):
    return a.ctypes.data_as(F)


rec = sys.argv[2] if len(sys.argv) > 2 else "/home/juyoung/engine-data/linux_official/radio500_h"
cams = json.load(open("/home/juyoung/engine-data/linux_official/camchain/scope.json"))[0]["robot"]["camera_chain"]
order = ["robot:left_realsense_link:Camera:0", "robot:right_realsense_link:Camera:0", "robot:zed_link:Camera:0"]
e = EngineCore(rec, with_s3=False)
rv = RobotView(rec)
idx = np.array(rv.base_control_idx + [rv.base_idx[5]] + rv.trunk_idx + rv.arm_idx["left"] + rv.arm_idx["right"] +
               rv.grip_idx["left"] + rv.grip_idx["right"], np.int32)
locals_ = {n: np.ascontiguousarray(np.array([c["local"] for c in cams[n]["chain"]], np.float64).reshape(-1)) for n in order}
acts = np.load(rec + "/actions.npz")["actions"][:, 0]
tr = np.load(rec + "/trace.npz", allow_pickle=True)
bad_p = np.zeros(61, int)
bad_c = np.zeros(21, int)
nbad = 0
for t in range(len(acts)):
    e.step(acts[t])
    q, v = e.joint_state()
    base = e.body_pose(rv.base_link)
    eL, eR = e.body_pose(rv.eef["left"]), e.body_pose(rv.eef["right"])
    out = np.zeros(61, np.float32)
    L.obs_proprio(idx.ctypes.data_as(I), fp(q), fp(v), fp(base), fp(eL), fp(eR), fp(out))
    cr = np.zeros(21, np.float32)
    for ci, n in enumerate(order):
        link = e.body_pose(cams[n]["link"])
        w7, r7 = np.zeros(7, np.float32), np.zeros(7, np.float32)
        L.obs_cam(locals_[n].ctypes.data_as(D), len(cams[n]["chain"]), fp(link), fp(base), fp(w7), fp(r7))
        cr[ci * 7:(ci + 1) * 7] = r7
    dp = out.astype(np.float64) != tr["obs::robot::proprio"][t, 0]
    dc = cr.astype(np.float64) != tr["obs::robot::cam_rel_poses"][t, 0]
    bad_p += dp
    bad_c += dc
    if (dp.any() or dc.any()):
        nbad += 1
        if nbad <= 2:
            print("스텝", t, "proprio 다른 칸", np.where(dp)[0], "cam 다른 칸", np.where(dc)[0])
print(f"스텝 {len(acts)}: 다른 스텝 {nbad}, proprio 칸별 {bad_p.tolist()}, cam_rel 칸별 {bad_c.tolist()}")
sys.exit(1 if nbad else 0)
