"""렌더 모듈 C 입구(librender_capi.so, src/sim/engine/eval/render_capi.cpp)의 ctypes 포장 + 엔진 몸체 자세 연결.

    rc = RenderCore(rsc_dir, frame_file, meta_json, engine_core)   # 기준 prim 경로 -> 엔진 몸체
    img = rc.render_all(cams_pose, res, frame)                     # {역할: (H, W, 3) u8}
"""
from __future__ import annotations

import ctypes as C
import json
import os

import numpy as np

DEFAULT_LIB = os.path.expanduser("~/engine-build/render-capi/librender_capi.so")
ROLES = ["head", "left_wrist", "right_wrist"]  # rsc 카메라 순서 (14.3)


class RenderCore:
    def __init__(self, rsc_dir, frame_file, meta_json, engine, spp=-1, threads=0, lib=None):
        self.lib = C.CDLL(lib or os.environ.get("ENGINE_RENDER_CAPI", DEFAULT_LIB))
        L = self.lib
        L.rr_open.restype = C.c_void_p
        L.rr_open.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_int]
        L.rr_n_anchor.restype = C.c_int
        L.rr_n_anchor.argtypes = [C.c_void_p]
        L.rr_set_anchor_pose.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_float)]
        L.rr_build.argtypes = [C.c_void_p]
        L.rr_render.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_int, C.c_int, C.c_int,
                                C.POINTER(C.c_uint8), C.POINTER(C.c_float)]
        L.rr_close.argtypes = [C.c_void_p]
        self.h = L.rr_open(rsc_dir.encode(), frame_file.encode(), int(spp), int(threads))
        if not self.h:
            raise RuntimeError(f"렌더 장면을 못 엶: {rsc_dir}")
        self.engine = engine
        mj = json.load(open(meta_json, encoding="utf-8"))
        # export/meta.json(anchor_paths) 또는 렌더 덤프 scene.json(anchors[].path) 둘 다 받는다
        paths = mj["anchor_paths"] if "anchor_paths" in mj else [a["path"] if isinstance(a, dict) else a for a in mj["anchors"]]
        n = L.rr_n_anchor(self.h)
        # 기준 prim 중 엔진(PhysX) 몸체인 것만 매 스텝 바꾼다. 나머지(조명 칸 포함)는 기준 프레임 값
        self.bound = []
        for a, pth in enumerate(paths[:n]):
            try:
                engine.body_pose(pth)
                self.bound.append((a, pth))
            except KeyError:
                pass
        self.n_anchor = n

    def set_poses(self):
        for a, pth in self.bound:
            p = self.engine.body_pose(pth)
            q = np.ascontiguousarray(p[3:], np.float32)
            t = np.ascontiguousarray(p[:3], np.float32)
            self.lib.rr_set_anchor_pose(self.h, a, q.ctypes.data_as(C.POINTER(C.c_float)), t.ctypes.data_as(C.POINTER(C.c_float)))
        self.lib.rr_build(self.h)

    def render(self, cam_index, pos, quat, w, hgt, frame):
        rgb = np.zeros((hgt, w, 3), np.uint8)
        q = np.ascontiguousarray(np.asarray(quat, np.float32))
        p = np.ascontiguousarray(np.asarray(pos, np.float32))
        self.lib.rr_render(self.h, int(cam_index), q.ctypes.data_as(C.POINTER(C.c_float)), p.ctypes.data_as(C.POINTER(C.c_float)),
                           int(w), int(hgt), int(frame), rgb.ctypes.data_as(C.POINTER(C.c_uint8)), None)
        return rgb

    def close(self):
        if self.h:
            self.lib.rr_close(self.h)
            self.h = None
