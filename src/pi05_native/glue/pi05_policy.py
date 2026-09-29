"""Evaluator-side glue: the native pi0.5 engine as the policy object of the official LocalPolicy.

Only this thin layer is Python. It hands buffer pointers to the C++/CUDA engine (`_pi05native`):
  - CPU torch tensors -> `.numpy()` (zero copy) -> buffer protocol
  - CUDA torch tensors -> `__cuda_array_interface__` device pointer (zero copy, engine copies on the GPU)
The engine does the openpi input transforms, the whole model (one CUDA graph), the output transforms and the
B1KPolicyWrapper receding-horizon buffering (execute 16 of the 32 actions, then re-infer).
The action goes back to the evaluator as `torch.from_numpy` of our float32 buffer (the evaluator API needs a
tensor; no copy).

Per-step log (CSV): step, env slot, whether a chunk was inferred, engine times, whether each camera frame was
entirely black (all RGB zero), and the wall time between consecutive act() calls (= simulator step time).
"""
from __future__ import annotations

import os
import time

import numpy as np

import _pi05native as N


def _as_image(x):
    """torch/numpy [H, W, C] uint8 -> buffer or device-pointer tuple for the engine."""
    cai = getattr(x, "__cuda_array_interface__", None)
    if cai is not None:
        h, w, c = cai["shape"]
        strides = cai.get("strides") or (w * c, c, 1)
        return (int(cai["data"][0]), int(h), int(w), int(strides[0]), int(strides[1])), None
    arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    return arr, arr


class Pi05NativePolicy:
    def __init__(self, weights: str, prompt: str, replan_every: int = 16, device: int = 0, log_path: str | None = None,
                 seed: int = 0):
        t0 = time.perf_counter()
        self.eng = N.create(weights, device)
        self.info = N.info(self.eng)
        N.seed(self.eng, seed)
        self.prompt = prompt
        self.replan = replan_every
        self.cams = list(self.info["cam_keys"])
        self.prop_key = f"{self.info['robot_name']}::proprio"
        self.ad = self.info["action_dim"]
        self.step = 0
        self.last_t = None
        self.log_path = log_path
        self.rows = []
        self.load_s = time.perf_counter() - t0
        print(f"[pi05_native] engine ready in {self.load_s:.1f}s: weights {self.info['weight_bytes'] / 2**30:.2f} GiB, "
              f"activations {self.info['activation_bytes'] / 2**30:.2f} GiB, prompt={prompt!r}", flush=True)

    def act(self, obs: dict):
        import torch  # the evaluator's return type; used only to wrap our output buffer

        now = time.perf_counter()
        step_ms = (now - self.last_t) * 1e3 if self.last_t is not None else float("nan")
        prop = obs[self.prop_key]
        batched = prop.ndim == 2
        n_env = prop.shape[0] if batched else 1
        out = np.empty((n_env, self.ad), np.float32)
        for b in range(n_env):
            views, black = [], []
            for key in self.cams:
                im = obs[key][b] if batched else obs[key]
                v, host = _as_image(im)
                views.append(v)
                black.append(int(host is not None and not host[..., :3].any()))
            p = prop[b] if batched else prop
            p = (p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)).astype(np.float32, copy=False)
            new, t = N.act(self.eng, b, views[0], views[1], views[2], np.ascontiguousarray(p), self.prompt,
                           self.replan, out[b])
            self.rows.append((self.step, b, new, t["gpu_ms"], t["total_ms"], t["tokens"], *black, step_ms))
        self.step += 1
        if self.log_path and self.step % 200 == 0:
            self.flush()
        self.last_t = time.perf_counter()
        res = torch.from_numpy(out)
        return res if batched else res[0]

    def reset(self):
        N.reset(self.eng, -1)
        self.last_t = None

    def flush(self):
        if not self.log_path:
            return
        new_file = not os.path.exists(self.log_path)
        with open(self.log_path, "a") as f:
            if new_file:
                f.write("step,slot,new_chunk,gpu_ms,total_ms,tokens,black_head,black_left,black_right,step_ms\n")
            for r in self.rows:
                f.write(",".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + "\n")
        self.rows = []
