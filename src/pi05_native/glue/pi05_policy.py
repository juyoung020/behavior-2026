"""Evaluator-side glue: the native pi0.5 engine as the policy object of the official LocalPolicy.

Only this thin layer is Python. It hands buffer pointers to the C++/CUDA engine (`_pi05native`):
  - CPU torch tensors -> `.numpy()` (zero copy) -> buffer protocol
  - CUDA torch tensors -> `__cuda_array_interface__` device pointer (zero copy, engine copies on the GPU)
The engine does the input transforms, the whole model (one CUDA graph), the output transforms and the policy wrapper:
  - openpi pi05 weights: B1KPolicyWrapper receding horizon (execute 16 of 32, re-infer)
  - PiBehavior weights (2025 1st place): stage voting, 26 -> 20 cubic compression, soft inpainting, correction rules;
    the task id comes from obs["task_id"], the stage can be fixed from outside with set_stage() (planner hook).
The action goes back to the evaluator as `torch.from_numpy` of our float32 buffer (the evaluator API needs a
tensor; no copy).

Per-step log (CSV): step, env slot, whether a chunk was inferred, engine times, stage (PiBehavior), whether each
camera frame was entirely black (all RGB zero), and the wall time between consecutive act() calls (= simulator step).
"""
from __future__ import annotations

import json
import os
import pathlib
import time

import numpy as np

import _pi05native as N

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repo root (src/pi05_native/glue -> ../../..)
DATA = ROOT / "data/pi05_native"


def _as_image(x):
    """torch/numpy [H, W, C] uint8 -> buffer or device-pointer tuple for the engine."""
    cai = getattr(x, "__cuda_array_interface__", None)
    if cai is not None:
        h, w, c = cai["shape"]
        strides = cai.get("strides") or (w * c, c, 1)
        return (int(cai["data"][0]), int(h), int(w), int(strides[0]), int(strides[1])), None
    arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    return arr, arr


def pb2025_weights_for_task(task_id: int) -> str:
    """2025 1st place: which of the 4 submission checkpoints serves a task (refs/behavior-1k-solution/
    task_checkpoint_mapping.json), exported as data/pi05_native/pb2025_ckpt<N>.pi05w."""
    m = json.loads((HERE.parents[2] / "refs/behavior-1k-solution/task_checkpoint_mapping.json").read_text())
    for name, v in m["checkpoints"].items():
        if task_id in v["tasks"]:
            return str(DATA / f"pb2025_{name.replace('checkpoint_', 'ckpt')}.pi05w")
    raise KeyError(f"task {task_id} has no 2025 checkpoint")


class Pi05NativePolicy:
    def __init__(self, weights, prompt: str = "", replan_every: int = 16, device: int = 0, log_path: str | None = None,
                 seed: int = 0, stage: int | None = None):
        """weights: path, or a callable task_id -> path (engine created at the first act, when the task is known)."""
        self.weights = weights
        self.prompt = prompt
        self.replan = replan_every
        self.device = device
        self.seed = seed
        self.fixed_stage = stage
        self.eng = None
        self.step = 0
        self.last_t = None
        self.log_path = log_path
        self.rows = []
        if not callable(weights):
            self._create(weights)

    def _create(self, path: str):
        t0 = time.perf_counter()
        self.eng = N.create(path, self.device)
        self.info = N.info(self.eng)
        N.seed(self.eng, self.seed)
        self.kind = self.info.get("model_kind", 0)  # older builds have no model_kind (pi05 only)
        self.cams = list(self.info["cam_keys"])
        self.prop_key = f"{self.info['robot_name']}::proprio"
        self.ad = self.info["action_dim"]
        self.load_s = time.perf_counter() - t0
        steps = self.info.get("num_steps", 10)
        print(f"[pi05_native] {path}: {'PiBehavior' if self.kind else 'pi05'} engine ready in {self.load_s:.1f}s, "
              f"weights {self.info['weight_bytes'] / 2**30:.2f} GiB, activations "
              f"{self.info['activation_bytes'] / 2**30:.2f} GiB, {steps} flow steps", flush=True)

    def set_stage(self, slot: int, stage: int, fixed: bool = True):
        """PiBehavior: fix the stage from outside (planner); fixed=False returns to the model's own voting."""
        self.fixed_stage = stage if fixed else None
        if self.eng is not None:
            N.set_stage(self.eng, slot, stage, 1 if fixed else 0)

    def act(self, obs: dict):
        import torch  # the evaluator's return type; used only to wrap our output buffer

        now = time.perf_counter()
        step_ms = (now - self.last_t) * 1e3 if self.last_t is not None else float("nan")
        task = None
        if "task_id" in obs:
            t = obs["task_id"]
            t = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            task = t.reshape(-1).astype(np.int64)
        if self.eng is None:
            self._create(self.weights(int(task[0])))
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
            stage = -1
            if self.kind == 1:
                N.set_task(self.eng, b, int(task[b if task.size > b else 0]))
                if self.fixed_stage is not None:
                    N.set_stage(self.eng, b, int(self.fixed_stage), 1)
            new, t = N.act(self.eng, b, views[0], views[1], views[2], np.ascontiguousarray(p), self.prompt,
                           self.replan, out[b])
            if self.kind == 1:
                stage = N.get_stage(self.eng, b)[0]
            self.rows.append((self.step, b, new, t["gpu_ms"], t["total_ms"], stage, *black, step_ms))
        self.step += 1
        if self.log_path and self.step % 200 == 0:
            self.flush()
        self.last_t = time.perf_counter()
        res = torch.from_numpy(out)
        return res if batched else res[0]

    def reset(self):
        if self.eng is not None:
            N.reset(self.eng, -1)
        self.last_t = None

    def flush(self):
        if not self.log_path or not self.rows:
            return
        new_file = not os.path.exists(self.log_path)
        with open(self.log_path, "a") as f:
            if new_file:
                f.write("step,slot,new_chunk,gpu_ms,total_ms,stage,black_head,black_left,black_right,step_ms\n")
            for r in self.rows:
                f.write(",".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + "\n")
        self.rows = []


class _EvalPolicy:
    """Official-policy-shaped wrapper (forward(obs) -> (num_envs, action_dim), reset, set_action_dim) for
    tools/eval_instrumented.py --native-policy=pi05_policy:make_policy (tools/exp_run.ps1 policy "native")."""

    def __init__(self, inner: Pi05NativePolicy):
        self.inner = inner

    def set_action_dim(self, n):
        assert n == 23, f"robot action dim {n} != 23"

    def forward(self, obs, *args, **kwargs):
        out = self.inner.act(obs)
        return out if out.ndim == 2 else out[None]

    def act(self, obs):
        return self.inner.act(obs)

    def reset(self):
        self.inner.flush()
        self.inner.reset()


def task_prompt(task: str) -> str:
    reg = json.loads((HERE / "b1k_tasks.json").read_text(encoding="utf-8"))  # openpi TASK_REGISTRY["b1k"]
    if task in reg:
        return reg[task]
    meta = ROOT / "data/2026-challenge-demos/meta/tasks.jsonl"
    for line in meta.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("task_name") == task:
            return d["task"]
    raise KeyError(task)


def make_policy(cfg):
    """Factory for --native-policy. Settings come from environment variables (exp_run passes none):
    PI05_MODEL = radio (openpi pi05, default) | pb2025 (1st place, checkpoint chosen by task id)
    PI05_WEIGHTS (explicit weight file), PI05_PROMPT, PI05_REPLAN (16), PI05_STAGE (PiBehavior: fixed stage),
    PI05_NATIVE_LOG."""
    import atexit

    task = cfg["task"]["name"]
    model = os.environ.get("PI05_MODEL", "radio")
    log = os.environ.get("PI05_NATIVE_LOG") or time.strftime(f"{ROOT}/logs/native_steps_{task}_%Y%m%d_%H%M%S.csv")
    stage = os.environ.get("PI05_STAGE")
    if os.environ.get("PI05_WEIGHTS"):
        weights = os.environ["PI05_WEIGHTS"]
    elif model == "pb2025":
        weights = pb2025_weights_for_task
    else:
        weights = str(DATA / "pi05_radio.pi05w")
    prompt = os.environ.get("PI05_PROMPT") or (task_prompt(task) if model != "pb2025" else "")
    inner = Pi05NativePolicy(weights, prompt, replan_every=int(os.environ.get("PI05_REPLAN", "16")), log_path=log,
                             stage=int(stage) if stage else None)
    atexit.register(inner.flush)
    print(f"[pi05_native] model {model}, per-step log: {log}", flush=True)
    return _EvalPolicy(inner)
