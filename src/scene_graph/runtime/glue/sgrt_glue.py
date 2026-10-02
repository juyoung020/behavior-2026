"""Thin evaluator-process glue for libsgrt.so (src/scene_graph/runtime/include/sgrt.h): hands the observation tensors'
pointers to the C++/CUDA runtime (YOLOE detection -> scenemap object map -> Spark-DSG save). No computation here.

    mem = SceneMemory(task_name, out_dir)        # once per process
    mem.step(obs)                                # every evaluator step, before the policy acts

The head RGB stays on the GPU (ovdet reads device memory); depth is copied to host only on keyframe steps.
"""
import ctypes
import os
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # repo root (src/scene_graph/runtime/glue -> ../../../..)
LIB = os.environ.get("SGRT_LIB", str(pathlib.Path.home() / "sgrt_build/libsgrt.so"))
ENGINE = os.environ.get("SGRT_ENGINE", str(pathlib.Path.home() / "ovdet_models/x86_sm120/yoloe-11l-all.plan"))
PROMPTS = ROOT / "src/scene_graph/ovdet/config/task_prompts.txt"
HEAD_K = (306.0, 306.0, 360.0, 360.0)  # omnigibson.eval.utils.eval_utils.CAMERA_INTRINSICS["R1Pro"]["head"] (720x720)


class _Cfg(ctypes.Structure):
    _fields_ = [("engine", ctypes.c_char_p), ("names", ctypes.c_char_p), ("out_dir", ctypes.c_char_p),
                ("kf_every", ctypes.c_int32), ("save_s", ctypes.c_double), ("conf_th", ctypes.c_float)]


def task_prompt_names(task: str) -> list[str]:
    """Task BDDL objects + scene structures, the ovdet prompt table (config/task_prompts.txt)."""
    lines = dict(l.split(":", 1) for l in PROMPTS.read_text(encoding="utf-8").splitlines() if ":" in l and not l.startswith("#"))
    names = [n.strip() for n in lines.get(task, "").split(",") if n.strip()]
    return names + [n.strip() for n in lines["_scene"].split(",") if n.strip() and n.strip() not in names]


def _ptr(t):
    """(pointer, on_device, row_stride_bytes, pix_stride_bytes) of an HxWxC uint8 tensor/array."""
    if hasattr(t, "data_ptr"):
        es = t.element_size()
        return t.data_ptr(), int(t.is_cuda), t.stride(0) * es, t.stride(1) * es
    a = np.ascontiguousarray(t)
    return a.ctypes.data, 0, a.strides[0], a.strides[1]


class SceneMemory:
    def __init__(self, task: str, out_dir: str, kf_every: int = 6, save_s: float = 1.0, robot: str = "robot"):
        self.L = ctypes.CDLL(LIB)
        L = self.L
        L.sgrt_create.restype = ctypes.c_void_p
        L.sgrt_create.argtypes = [ctypes.POINTER(_Cfg), ctypes.c_char_p, ctypes.c_size_t]
        L.sgrt_begin.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t]
        L.sgrt_want_image.argtypes = [ctypes.c_void_p]
        L.sgrt_step.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32,
                                ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
                                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        L.sgrt_save.argtypes = [ctypes.c_void_p]
        L.sgrt_stats.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 5
        L.sgrt_destroy.argtypes = [ctypes.c_void_p]
        cfg = _Cfg(ENGINE.encode(), (ENGINE + ".names.txt").encode(), out_dir.encode(), kf_every, save_s, 0.25)
        err = ctypes.create_string_buffer(512)
        self.h = L.sgrt_create(ctypes.byref(cfg), err, 512)
        if not self.h:
            raise RuntimeError(f"sgrt_create: {err.value.decode()}")
        names = task_prompt_names(task)
        arr = (ctypes.c_char_p * len(names))(*[n.encode() for n in names])
        L.sgrt_begin(self.h, arr, len(names), err, 512)
        if err.value:
            print(f"[sgrt] prompt: {err.value.decode()}", flush=True)
        self.keys = (f"{robot}::proprio", f"{robot}::{robot}:zed_link:Camera:0::rgb", f"{robot}::{robot}:zed_link:Camera:0::depth_linear")
        self.t = 0
        print(f"[sgrt] {task}: {len(names)} prompt names, out {out_dir}", flush=True)

    def step(self, obs: dict):
        kp, kr, kd = self.keys
        if kp not in obs:  # weights' robot name differs from the evaluator's (e.g. robot_r1): find by suffix
            kp = next(k for k in obs if k.endswith("::proprio"))
            r = kp.split("::")[0]
            kr, kd = f"{r}::{r}:zed_link:Camera:0::rgb", f"{r}::{r}:zed_link:Camera:0::depth_linear"
            self.keys = (kp, kr, kd)
        p = obs[kp]
        p = p[0] if p.ndim == 2 else p
        prop = np.ascontiguousarray((p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)), np.float32)
        stamp = self.t / 30.0
        rgb = depth = None
        rp, dev, rs, ps, w, h = None, 0, 0, 0, 0, 0
        want = self.L.sgrt_want_image(self.h)
        if want and not (kr in obs and kd in obs) and not getattr(self, "_warned", False):
            self._warned = True
            print(f"[sgrt] no head RGB-D in obs ({kr}, {kd}); keys: {sorted(obs)} — use RGBDFullResWrapper", flush=True)
        if want and kr in obs and kd in obs:
            rgb, depth = obs[kr], obs[kd]
            rgb = rgb[0] if rgb.ndim == 4 else rgb
            depth = depth[0] if depth.ndim == 3 else depth
            if hasattr(rgb, "contiguous"):
                rgb = rgb.contiguous()
            rp, dev, rs, ps = _ptr(rgb)
            h, w = int(rgb.shape[0]), int(rgb.shape[1])
            depth = np.ascontiguousarray((depth.detach().cpu().numpy() if hasattr(depth, "detach") else np.asarray(depth)), np.float32)
        self.L.sgrt_step(self.h, stamp, prop.ctypes.data, prop.size, rp, dev, rs, ps, w, h,
                         depth.ctypes.data if depth is not None else None, *HEAD_K)
        self.t += 1

    def stats(self):
        v = [ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32(), ctypes.c_float(), ctypes.c_float()]
        self.L.sgrt_stats(self.h, *[ctypes.byref(x) for x in v])
        return dict(keyframes=v[0].value, last_dets=v[1].value, objects=v[2].value, det_ms=v[3].value, save_ms=v[4].value)

    def close(self):
        if self.h:
            self.L.sgrt_save(self.h)
            self.L.sgrt_destroy(self.h)
            self.h = None
