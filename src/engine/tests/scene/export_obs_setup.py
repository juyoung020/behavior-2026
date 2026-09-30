"""관측(proprio 61) 가공 입력 + 공식 기준값 (G4c, native_loop --obs 가 씀).

    python3 export_obs_setup.py <기록 폴더>    # scope.json · s1_setup.txt · trace.npz 가 있는 곳

<기록 폴더>/obs_setup.txt : base_link / eef_left / eef_right 경로, idx 26 개 (base_control 3, yaw dof, trunk 4, arm 2×7, grip 2×2 — tests/omni/obs_capi.cpp 와 같은 차례)
<기록 폴더>/obs_ref.bin   : int32 T, int32 61, float32 [T][61] 공식 obs::robot::proprio (스텝 t 행동 뒤)
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval"))
from obs_engine import RobotView  # noqa: E402


def main():
    rec = sys.argv[1]
    rv = RobotView(rec)
    idx = list(rv.base_control_idx) + [rv.base_idx[5]] + list(rv.trunk_idx) + list(rv.arm_idx["left"]) + list(rv.arm_idx["right"]) + \
        list(rv.grip_idx["left"]) + list(rv.grip_idx["right"])
    assert len(idx) == 26, len(idx)
    with open(os.path.join(rec, "obs_setup.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"base_link {rv.base_link}\n")
        f.write(f"eef_left {rv.eef['left']}\n")
        f.write(f"eef_right {rv.eef['right']}\n")
        f.write("idx " + " ".join(str(int(i)) for i in idx) + "\n")
    tr = np.load(os.path.join(rec, "trace.npz"), allow_pickle=True)
    p = np.asarray(tr["obs::robot::proprio"])[:, 0].astype(np.float32)
    with open(os.path.join(rec, "obs_ref.bin"), "wb") as f:
        f.write(struct.pack("<ii", p.shape[0], p.shape[1]))
        f.write(np.ascontiguousarray(p).tobytes())
    print(f"obs_setup.txt, obs_ref.bin: 스텝 {p.shape[0]} × {p.shape[1]}")


if __name__ == "__main__":
    main()
