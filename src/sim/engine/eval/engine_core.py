"""엔진 C 입구(libengine_capi.so, src/sim/engine/replay/engine_capi.cpp)의 얇은 ctypes 포장. 파이썬 쪽 백엔드(backend_engine.py)가 쓴다.

    from engine_core import EngineCore
    e = EngineCore(rec_dir)            # 에피소드 시작 직전까지 장면 추출물 재생
    e.step(action_23)                  # 평가 스텝 하나 (서브스텝 4)
    q, v = e.joint_state()             # 텐서 API dof 순서, float32
    p7 = e.body_pose("/World/scene_0/...")
"""
from __future__ import annotations

import ctypes as C
import glob
import json
import os

import numpy as np

DEFAULT_LIB = os.path.expanduser("~/engine-build/replay-checked/libengine_capi.so")


def _ovd_and_offset(rec_dir):
    """가장 큰 OVD = 에피소드가 든 PhysX 인스턴스. 앞 몫 = meta.json post_step_count - 그 OVD 의 simulate 수 (문서 15.1)."""
    ovds = sorted(glob.glob(os.path.join(rec_dir, "*_rec.ovd")), key=os.path.getsize)
    if not ovds:
        raise FileNotFoundError(f"OVD 없음: {rec_dir}")
    big = ovds[-1]
    off_file = os.path.join(rec_dir, "side_offset.txt")
    if os.path.exists(off_file):
        return big, int(open(off_file).read().split()[0])
    return big, 16


class EngineCore:
    def __init__(self, rec_dir, lib=None, threads=1, with_s3=True):
        self.lib = C.CDLL(lib or os.environ.get("ENGINE_CAPI", DEFAULT_LIB))
        L = self.lib
        L.ee_open.restype = C.c_void_p
        L.ee_open.argtypes = [C.c_char_p, C.c_char_p, C.c_uint64, C.c_int, C.c_int]
        for f in ("ee_n_dof", "ee_action_dim", "ee_substeps"):
            getattr(L, f).restype = C.c_int
            getattr(L, f).argtypes = [C.c_void_p]
        L.ee_sims.restype = C.c_uint64
        L.ee_sims.argtypes = [C.c_void_p]
        L.ee_step.argtypes = [C.c_void_p, C.POINTER(C.c_float)]
        L.ee_joint_state.restype = C.c_int
        L.ee_joint_state.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.POINTER(C.c_float)]
        L.ee_body_pose.restype = C.c_int
        L.ee_body_pose.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_float)]
        L.ee_body_vel.restype = C.c_int
        L.ee_body_vel.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_float)]
        L.ee_goal.restype = C.c_int
        L.ee_goal.argtypes = [C.c_void_p, C.c_char_p, C.c_int]
        L.ee_report.argtypes = [C.c_void_p]
        L.ee_close.argtypes = [C.c_void_p]
        ovd, off = _ovd_and_offset(rec_dir)
        self.h = L.ee_open(rec_dir.encode(), ovd.encode(), off, threads, 1 if with_s3 else 0)
        if not self.h:
            raise RuntimeError(f"엔진을 못 엶: {rec_dir}")
        self.n_dof = L.ee_n_dof(self.h)
        self.action_dim = L.ee_action_dim(self.h)
        self.substeps = L.ee_substeps(self.h)

    def step(self, action):
        a = np.ascontiguousarray(np.asarray(action, np.float32).reshape(-1))
        assert a.size == self.action_dim, (a.size, self.action_dim)
        self.lib.ee_step(self.h, a.ctypes.data_as(C.POINTER(C.c_float)))

    def joint_state(self):
        q = np.zeros(self.n_dof, np.float32)
        v = np.zeros(self.n_dof, np.float32)
        n = self.lib.ee_joint_state(self.h, q.ctypes.data_as(C.POINTER(C.c_float)), v.ctypes.data_as(C.POINTER(C.c_float)))
        if n != self.n_dof:
            raise RuntimeError("로봇 관절을 못 묶음")
        return q, v

    def body_pose(self, path):
        p = np.zeros(7, np.float32)
        if not self.lib.ee_body_pose(self.h, path.encode(), p.ctypes.data_as(C.POINTER(C.c_float))):
            raise KeyError(path)
        return p

    def body_vel(self, path):
        v = np.zeros(6, np.float32)
        if not self.lib.ee_body_vel(self.h, path.encode(), v.ctypes.data_as(C.POINTER(C.c_float))):
            raise KeyError(path)
        return v

    def goal(self):
        b = C.create_string_buffer(256)
        self.lib.ee_goal(self.h, b, 256)
        return b.value.decode()

    @property
    def sims(self):
        return int(self.lib.ee_sims(self.h))

    def report(self):
        self.lib.ee_report(self.h)

    def close(self):
        if self.h:
            self.lib.ee_close(self.h)
            self.h = None
