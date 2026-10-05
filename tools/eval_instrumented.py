"""공식 평가기(omnigibson.eval.eval)를 한 줄도 고치지 않고 돌리면서 구간별 시간과 스텝별 기록을 남긴다.

    python C:\\behavior-2026\\tools\\eval_instrumented.py [--timing] [--trace] -- <omnigibson.eval.eval 인자 그대로>
    python C:\\behavior-2026\\tools\\eval_instrumented.py --report <timing.json> [--skip 20]

결과는 평가기 --output-dir 안에 쓴다.
    timing.json   --timing : 스텝마다 구간별 '자기 시간'(자식 구간을 뺀 시간), 호출 수, 서버 추론 시간, GPU 메모리
    trace.npz     --trace  : 스텝마다 행동·관측(proprio 값, 카메라 해시/평균)·로봇 pose/관절·과제 물체 pose·목표 조건·지표
    actions.npz   항상     : 정책이 돌려준 행동열 (tools\\replay_policy_server.py 로 같은 행동을 다시 먹일 때 씀)

원리
- 평가기 함수들을 실행 중에 '겉싸개'로 감싼다. 겉싸개는 perf_counter 로 들어간/나온 시각만 적고 인자·반환값은
  그대로 넘긴다. 공식 파일(BEHAVIOR-1K)은 건드리지 않는다.
- cProfile 을 쓰지 않는 이유: Kit(Isaac Sim) 이 carb.profiler 를 불러오면서 파이썬 프로파일 훅을 가져가
  그 뒤로는 아무것도 안 잡힌다 (2026-09-29 19:29 실행: 206 초 전체가 carb.profiler import 한 줄에 몰림).
- 구간 이름은 호출 경로다: step/apply_actions/env.step/sim.step/app.update/physx
- 물리(PhysX) = pre-physics 콜백이 끝난 시각 ~ post-physics 콜백이 시작한 시각 (서브스텝마다, 120 Hz 면 스텝당 4 번).
- app.update 의 자기 시간 = Kit 한 프레임에서 물리·콜백을 뺀 나머지 = RTX 렌더 제출 + fabric 동기화 + Kit 갱신.
  RTX 는 GPU 에서 비동기라, 렌더가 끝나기를 기다리는 시간 일부는 camera_read(GPU->CPU 읽기)에 잡힌다.
- 겉싸개 자체 비용은 호출당 약 1 us 다 (--report 가 호출 수 x 이 값을 따로 적는다).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from time import perf_counter

import numpy as np

IMAGE_STEPS = {0, 1, 2, 10, 50, 100, 101, 150, 200, 300, 400, 500}  # --trace 가 영상 원본을 남기는 스텝
WRAPPER_COST_S = 1.0e-6  # 겉싸개 한 번의 대략 비용 (보고용 추정, --report 가 따로 잰다)


# ----------------------------------------------------------------------------------------------------------------------
# 시간 재기
# ----------------------------------------------------------------------------------------------------------------------
class StepTimer:
    """호출 경로별 '자기 시간'을 스텝 단위로 모은다. 한 스텝 = _step_fn 시작 ~ 다음 _step_fn 시작."""

    def __init__(self):
        self.main = threading.get_ident()
        self.stack = []  # [path, t0, child_time]
        self.active = False
        self.excl = defaultdict(float)
        self.calls = defaultdict(int)
        self.meta = defaultdict(list)
        self.steps = []
        self.step_t0 = None
        self.pre_end = None
        self.offthread = defaultdict(float)
        self.oneoff = {}

    def _path(self, name):
        return f"{self.stack[-1][0]}/{name}" if self.stack else name

    def enter(self, name):
        if threading.get_ident() != self.main:
            return ("__off__", name, perf_counter())
        p = self._path(name)
        self.stack.append([p, perf_counter(), 0.0])
        return p

    def exit(self, tok):
        if isinstance(tok, tuple):
            self.offthread[tok[1]] += perf_counter() - tok[2]
            return
        p, t0, child = self.stack.pop()
        dt = perf_counter() - t0
        if self.stack:
            self.stack[-1][2] += dt
        if self.active:
            self.excl[p] += dt - child
            self.calls[p] += 1

    def synthetic(self, name, dt):
        p = self._path(name)
        if self.stack:
            self.stack[-1][2] += dt
        if self.active:
            self.excl[p] += dt
            self.calls[p] += 1

    def close_step(self):
        now = perf_counter()
        if self.active and self.step_t0 is not None:
            rec = {"wall": now - self.step_t0, "excl": dict(self.excl), "calls": dict(self.calls)}
            rec.update({k: v for k, v in self.meta.items()})
            self.steps.append(rec)
        self.excl, self.calls, self.meta = defaultdict(float), defaultdict(int), defaultdict(list)
        self.step_t0 = now

    def new_step(self):
        self.close_step()
        self.active = True


T = StepTimer()


def _owner(cls, attr):
    for k in cls.__mro__:
        if attr in k.__dict__:
            return k
    raise AttributeError(f"{cls.__name__}.{attr} 없음")


def wrap(cls, attr, label, before=None, after=None):
    """cls.attr(정의된 클래스에서) 를 시간 재는 겉싸개로 바꾼다. label 은 문자열 또는 (args)->문자열."""
    owner = _owner(cls, attr)
    orig = owner.__dict__[attr]
    if getattr(orig, "__fe_wrapped__", False):
        return
    kind = None
    if isinstance(orig, staticmethod):
        kind, fn = staticmethod, orig.__func__
    elif isinstance(orig, classmethod):
        kind, fn = classmethod, orig.__func__
    else:
        fn = orig

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        if before is not None:
            before(a, kw)
        tok = T.enter(label(a) if callable(label) else label)
        try:
            ret = fn(*a, **kw)
        finally:
            T.exit(tok)
        if after is not None:
            after(a, kw, ret)
        return ret

    wrapper.__fe_wrapped__ = True
    setattr(owner, attr, kind(wrapper) if kind else wrapper)


def wrap_module_fn(mod, attr, label):
    fn = getattr(mod, attr)

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        tok = T.enter(label)
        try:
            return fn(*a, **kw)
        finally:
            T.exit(tok)

    setattr(mod, attr, wrapper)


class _TimedPacker:
    def __init__(self, inner):
        self._inner = inner

    def pack(self, obj):
        tok = T.enter("pack")
        try:
            return self._inner.pack(obj)
        finally:
            T.exit(tok)


class _TimedConn:
    def __init__(self, inner):
        self._inner = inner

    def send(self, data):
        T.meta["req_bytes"].append(len(data))
        tok = T.enter("ws.send")
        try:
            return self._inner.send(data)
        finally:
            T.exit(tok)

    def recv(self, *a, **kw):
        tok = T.enter("ws.recv")
        try:
            return self._inner.recv(*a, **kw)
        finally:
            T.exit(tok)

    def __getattr__(self, k):
        return getattr(self._inner, k)


# ----------------------------------------------------------------------------------------------------------------------
# GPU 메모리 (전체 사용량, WDDM 이라 프로세스별 값은 안 나온다)
# ----------------------------------------------------------------------------------------------------------------------
def _gpu_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return -1


class GpuMonitor(threading.Thread):
    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        self.period, self.samples, self._halt = period, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            self.samples.append((time.time(), _gpu_used_mib()))
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set()


# ----------------------------------------------------------------------------------------------------------------------
# 스텝별 기록 (--trace)
# ----------------------------------------------------------------------------------------------------------------------
def _np(v):
    import torch as th

    if isinstance(v, th.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _digest(arr: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(arr).tobytes(), digest_size=16).hexdigest()


class TraceRecorder:
    """_apply_actions 가 끝난 직후(= 정책이 다음에 받을 관측이 정해진 시점)에 읽기만 한다."""

    def __init__(self, full):
        self.full = full
        self.rows = []  # 스텝마다 {key: [N 개 값]}
        self.static = {}
        self.images = {}  # 몇 스텝만 영상 원본 (렌더 잡음 크기 확인용): "스텝|env|key" -> uint8

    def on_apply(self, ev, actions, active, ret):
        terminated, truncated, info = ret
        n = ev.num_envs
        row = {"action": [_np(actions[i]).astype(np.float32) for i in range(n)],
               "active": [i in active for i in range(n)]}
        if self.full:
            from omnigibson.utils.bddl_utils import is_system_bddl_inst

            row["terminated"] = [bool(terminated[i]) for i in range(n)]
            row["truncated"] = [bool(truncated[i]) for i in range(n)]
            for i in range(n):
                st = ev.instance_eval_states[i]
                if i not in active:
                    continue
                for k, v in (st.obs or {}).items():
                    a = _np(v)
                    if a.dtype == np.uint8 or a.size > 4096:
                        row.setdefault(f"obs_hash::{k}", [None] * n)[i] = _digest(a)
                        if len(self.rows) in IMAGE_STEPS:
                            self.images[f"{len(self.rows)}|{i}|{k}"] = a.copy()
                        ch = a.reshape(-1, a.shape[-1]) if a.ndim >= 2 else a.reshape(-1, 1)
                        row.setdefault(f"obs_mean::{k}", [None] * n)[i] = ch.astype(np.float64).mean(axis=0)
                    else:
                        row.setdefault(f"obs::{k}", [None] * n)[i] = a.astype(np.float64).reshape(-1)
                robot = st.env_accessor.robot
                pos, quat = robot.get_position_orientation()
                row.setdefault("robot_pose", [None] * n)[i] = np.concatenate([_np(pos), _np(quat)]).astype(np.float64)
                row.setdefault("robot_qpos", [None] * n)[i] = _np(robot.get_joint_positions()).astype(np.float64)
                row.setdefault("robot_qvel", [None] * n)[i] = _np(robot.get_joint_velocities()).astype(np.float64)
                for name, ent in st.env_accessor.object_scope.items():
                    if ent is None or is_system_bddl_inst(name) or not hasattr(ent, "get_position_orientation"):
                        continue
                    try:
                        p, q = ent.get_position_orientation()
                        val = np.concatenate([_np(p), _np(q)]).astype(np.float64)
                    except Exception:
                        val = np.full(7, np.nan)
                    row.setdefault(f"obj::{name}", [None] * n)[i] = val
                gs = info[i].get("done", {}).get("goal_status", {}) if isinstance(info[i], dict) else {}
                row.setdefault("goal_satisfied", [None] * n)[i] = json.dumps(sorted(gs.get("satisfied", [])))
                am = [m for m in st.metrics if type(m).__name__ == "AgentMetric"]
                if am and hasattr(am[0], "delta_agent_distance"):
                    d = am[0].delta_agent_distance
                    row.setdefault("agent_delta", [None] * n)[i] = np.array([v[-1] if v else np.nan for v in d.values()])
        self.rows.append(row)

    def on_run_start(self, ev):
        if not self.full:
            return
        offs = []
        for st in ev.instance_eval_states:
            try:
                offs.append(_np(st.env_accessor.scene._scene_prim.get_position_orientation()[0]).astype(np.float64))
            except Exception:
                offs.append(np.full(3, np.nan))
        self.static["scene_offset"] = np.stack(offs)

    def save(self, path, actions_path):
        if not self.rows:
            return
        n = len(self.rows[0]["action"])
        acts = np.stack([np.stack(r["action"]) for r in self.rows])  # (T, N, A)
        np.savez_compressed(actions_path, actions=acts)
        if not self.full:
            return
        keys = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        out = {}
        for k in keys:
            vals = [r.get(k, [None] * n) for r in self.rows]
            sample = next((v for row in vals for v in row if v is not None), None)
            if isinstance(sample, str):
                out[k] = np.array([[v if v is not None else "" for v in row] for row in vals], dtype=object).astype(str)
            elif isinstance(sample, (bool, np.bool_)):
                out[k] = np.array([[bool(v) if v is not None else False for v in row] for row in vals])
            else:
                shape = np.asarray(sample).shape
                arr = np.full((len(vals), n) + shape, np.nan)
                for t, row in enumerate(vals):
                    for i, v in enumerate(row):
                        if v is not None and np.asarray(v).shape == shape:
                            arr[t, i] = v
                out[k] = arr
        out.update({f"static::{k}": v for k, v in self.static.items()})
        np.savez_compressed(path, **out)
        if self.images:
            np.savez_compressed(path.replace(".npz", "_images.npz"), **self.images)


# ----------------------------------------------------------------------------------------------------------------------
# 설치
# ----------------------------------------------------------------------------------------------------------------------
def install_deep_hooks(og, Sim, Robot, VisionSensor):
    """--deep: 컨트롤러 콜백·물체 상태 캐시·proprio 안쪽을 더 잘게 (호출이 많아 겉싸개 비용이 조금 붙는다)."""
    import warp as wp
    from omnigibson.controllers.controller_view import ControllerView
    from omnigibson.object_states.tensorized_state import TensorizedState
    from omnigibson.utils import usd_utils as U

    wrap(ControllerView, "step_all", "controllers.step_all")
    for g in ControllerView._controller_groups.values():
        wrap(type(g), "step", lambda a: f"ctrl.{type(a[0]).__name__}")
    wrap(Robot, "post_step", "robot.post_step(grasp)")
    wrap(U.ControllableObjectViewAPI, "flush_control", "flush_control")
    wrap(U.ControllableObjectViewAPI, "post_physics_step", "ctrl_view.post_physics_step")
    wrap(U.RigidContactAPIImpl, "read_from_physx", "contacts.read_from_physx")
    wrap(U.RigidContactAPIImpl, "update", "contacts.update")
    wrap(U.RigidBodyViewAPI, "read_from_physx", "rigid.read_from_physx")
    wrap(U.RigidBodyViewAPI, "update", "rigid.update")
    wrap(U.ArticulatedObjectViewAPI, "read_from_physx", "artic.read_from_physx")
    wrap(Sim, "_capture_warp_graph", "warp_graph")
    for st in list(og.sim.object_state_types_requiring_update):
        if isinstance(st, type) and issubclass(st, TensorizedState):
            for m in ("pre_update", "post_update"):
                if m in st.__dict__:
                    wrap(st, m, f"ts.{m}.{st.__name__}")
    for m in ("pre_update", "post_update"):
        wrap(TensorizedState, m, lambda a, m=m: f"ts.{m}.{a[0].__name__ if isinstance(a[0], type) else type(a[0]).__name__}")
    wrap_module_fn(wp, "synchronize_stream", "wp.synchronize_stream")
    # 아래는 정의된 클래스(부모)에서 감싸므로 로봇·카메라·물체 호출이 모두 잡힌다 -- 이름에 정의 클래스를 적는다
    for cls, m in ((Robot, "get_joint_positions"), (Robot, "get_joint_velocities"), (Robot, "get_joint_efforts"),
                   (Robot, "get_position_orientation"), (Robot, "get_eef_position"), (Robot, "get_eef_orientation"),
                   (VisionSensor, "get_position_orientation")):
        try:
            wrap(cls, m, f"{_owner(cls, m).__name__}.{m}")
        except AttributeError:
            pass


class BlackFrameDiag:
    """--black-diag / --black-guard: 정책에 들어가는 카메라 영상이 검은지(RGBA 전부 0) 스텝마다 본다.

    - 공식 경로는 annotator.get_data(device="cpu") (호스트 복사본, sensors/vision_sensor.py:308).
      --black-diag 는 같은 annotator 의 GPU 버퍼(device="cuda")도 읽어, 두 경로의 검정 여부와 픽셀 차이를 적는다
      (어느 단계에서 검어지는지 가르기용 -- GPU 버퍼 읽기가 렌더 그래프를 건드릴 수 있어 진단 전용).
    - --black-guard warn|abort: 정책에 들어간 프레임이 검으면 경고(warn) 또는 그 자리에서 중단(abort).
    """

    def __init__(self, diag: bool, guard: str):
        self.diag, self.guard = diag, guard
        self.rows = defaultdict(list)  # 센서 -> [(스텝, host 검정, cuda 검정, 둘 다 안 검을 때 최대|차|)]
        self.n_black = defaultdict(int)
        self.n_total = defaultdict(int)
        self.times = defaultdict(list)  # 센서 -> [(스텝, 검정, 프레임 기준 시각, 시뮬레이터 시각)]

    def after_get_obs(self, sensor, obs):
        if "rgb" not in obs:
            return
        host = obs["rgb"]
        hb = bool(host.max() == 0)
        name = sensor.name
        self.n_total[name] += 1
        self.n_black[name] += hb
        step = len(T.steps)
        if self.diag:
            # 시간 동기화 확인: 이 스텝에 받은 annotator 데이터가 몇 시각에 그린 프레임인지(ReferenceTime annotator) vs 시뮬레이터 시각
            ref = None
            try:
                import omnigibson as og
                import omni.replicator.core as rep

                if not hasattr(sensor, "_fe_reftime"):
                    with og.sim.editing_usd():  # render var 추가 = USD 편집 (OmniGibson 의 USD 편집 감시)
                        sensor._fe_reftime = rep.AnnotatorRegistry.get_annotator("ReferenceTime")
                        sensor._fe_reftime.attach([sensor.render_product])
                rt = sensor._fe_reftime.get_data()
                num, den = rt.get("referenceTimeNumerator"), rt.get("referenceTimeDenominator")
                ref = (float(num) / float(den)) if den else None
                self.times[name].append((step, hb, ref, float(og.sim.current_time)))
            except Exception as e:
                self.times[name].append((step, hb, None, f"{type(e).__name__}: {e}"[:60]))
            try:
                import warp as wp

                raw = sensor._annotators["rgb"].get_data(device="cuda")
                raw = raw["data"] if isinstance(raw, dict) else raw
                dev = wp.to_torch(raw).cpu()
                cb = bool(dev.max() == 0)
                d = -1 if (hb or cb) else int((dev.to(dtype=host.dtype) - host.cpu()).abs().max())
            except Exception as e:  # 진단 실패는 기록만
                cb, d = None, f"{type(e).__name__}: {e}"[:80]
            self.rows[name].append((step, hb, cb, d))
        if hb and self.guard:
            msg = f"[black-guard] 스텝 {step} 카메라 {name}: 정책에 들어갈 RGB 가 전부 0 (누적 {self.n_black[name]}/{self.n_total[name]})"
            print(msg, flush=True)
            if self.guard == "abort":
                raise RuntimeError(msg)

    def summary(self):
        out = {"black": dict(self.n_black), "total": dict(self.n_total), "settings": getattr(self, "settings", {})}
        if self.diag:
            out["diag"] = {k: v for k, v in self.rows.items()}
            out["times"] = {k: v for k, v in self.times.items()}
        return out


BLACK = None
SET_SETTINGS = []  # --set KEY=VALUE (진단용)
RENDER_ITERS = 0  # --render-iters=N (진단용)
KIT_ARGS = []  # --kit-arg=--/app/vulkan=false 등 Kit 시작 인자 (진단용)
DUMP_SETTINGS = ""  # --dump-settings=PATH : 판 시작 때 carb 설정 트리 전체를 JSON 으로 (진단용, 읽기만)
WATCH_SETTINGS = ["/app/vulkan", "/app/gatherRenderResults", "/app/settings/fabricDefaultStageFrameHistoryCount",
                  "/exts/omni.kit.renderer.core/present/enabled", "/renderer/multiGpu/enabled", "/renderer/activeGpu",
                  "/rtx-transient/dlssg/enabled", "/rtx/post/dlss/execMode", "/rtx/post/aa/op", "/rtx/rendermode",
                  "/app/asyncRendering", "/app/asyncRenderingLowLatency", "/omni/replicator/asyncRendering",
                  "/app/hydraEngine/waitIdle", "/app/renderer/waitIdle", "/rtx/pathtracing/dlss/enabled",
                  "/rtx-transient/dlssg/mode", "/rtx/dlssg/enabled"]


def install(timing: bool, trace: TraceRecorder, out_dir: str, gpu: GpuMonitor, deep: bool = False):
    import omnigibson as og
    from omnigibson.eval import evaluator as E
    from omnigibson.eval import policies as P
    from omnigibson.eval.utils import network_utils as NU

    Ev = E.BatchedEvaluator
    state = {"sim_hooks": False, "saved": False}

    def install_sim_hooks():
        if state["sim_hooks"] or not timing:
            return
        state["sim_hooks"] = True
        from omnigibson.envs.env_base import Environment
        from omnigibson.metrics.metric_base import MetricBase
        from omnigibson.object_states.update_state_mixin import UpdateStateMixin
        from omnigibson.objects.usd_object import USDObject
        from omnigibson.reward_functions.potential_reward import PotentialReward
        from omnigibson.robots.robot import Robot
        from omnigibson.sensors.vision_sensor import VisionSensor
        from omnigibson.systems.system_base import BaseSystem
        from omnigibson.tasks.task_base import BaseTask
        from omnigibson.termination_conditions.predicate_goal import PredicateGoal
        from omnigibson.transition_rules import TransitionRuleAPI
        from omnigibson.eval.utils.light_utils import LightToggleSynchronizer

        Sim = type(og.sim)
        wrap(Environment, "step", "env.step")
        wrap(Environment, "get_obs", "get_obs")
        wrap(Robot, "apply_action", "apply_action")
        wrap(Robot, "get_proprioception", "proprio")
        wrap(VisionSensor, "_get_obs", "camera_read")
        wrap(Sim, "step", "sim.step")
        wrap(Sim, "render", "sim.render")
        wrap(type(og.sim._sim_context), "step", "app.update")

        def pre_after(a, kw, ret):
            if threading.get_ident() == T.main:
                T.pre_end = perf_counter()
            else:
                T.oneoff["physx_callbacks_offthread"] = True

        def post_before(a, kw):
            if T.pre_end is not None and threading.get_ident() == T.main:
                T.synthetic("physx", perf_counter() - T.pre_end)
                T.pre_end = None

        wrap(Sim, "_on_pre_physics_step", "pre_physics(controllers)", after=pre_after)
        wrap(Sim, "_on_post_physics_step", "post_physics(contacts)", before=post_before)
        wrap(Sim, "_non_physics_step", "non_physics")
        wrap(Sim, "_refresh_state_caches", "state_caches(warp)")
        wrap(BaseSystem, "update", "systems.update")
        wrap(UpdateStateMixin, "update", "object_state.update")
        wrap(USDObject, "update_visuals", "update_visuals")
        wrap(TransitionRuleAPI, "step", "transition_rules")
        wrap(BaseTask, "step", "task.step")
        wrap(PredicateGoal, "_step", "bddl_goal(termination)")
        wrap(PotentialReward, "_step", "bddl_goal(reward)")
        wrap(MetricBase, "step", lambda a: f"metric.{type(a[0]).__name__}")
        wrap(LightToggleSynchronizer, "sync_from_current_state", "light_sync")
        if deep:
            install_deep_hooks(og, Sim, Robot, VisionSensor)

    # --- 평가기 층 ---
    orig_run = Ev.run

    @functools.wraps(orig_run)
    def run(self, *a, **kw):  # 경로(stack)에 넣지 않는다 -- 스텝 경로가 'step/...' 로 시작하게
        install_sim_hooks()
        install_black_hook()
        return orig_run(self, *a, **kw)

    def install_black_hook():
        if BLACK is None or state.get("black_hook"):
            return
        state["black_hook"] = True
        import carb

        cs = carb.settings.get_settings()
        for k, v in SET_SETTINGS:  # 진단 실험용 --set (공식 실행에는 쓰지 않는다)
            cur = cs.get(k)
            val = {"true": True, "false": False}.get(v.lower(), v)
            if not isinstance(val, bool):
                try:
                    val = int(val)
                except ValueError:
                    pass
            cs.set(k, val)
            print(f"[black] 설정 {k}: {cur!r} -> {cs.get(k)!r}", flush=True)
        BLACK.settings = {k: repr(cs.get(k)) for k in WATCH_SETTINGS}
        print(f"[black] 렌더 설정: {BLACK.settings}", flush=True)
        if DUMP_SETTINGS:
            # 진단: carb 설정 트리 전체를 평평하게("/a/b": 값) 적는다(읽기만). 판이 시작하는 시점 = Kit·장면·렌더 설정이 다 들어간 뒤
            flat = {}

            def walk(node, path):
                if isinstance(node, dict):
                    for kk, vv in node.items():
                        walk(vv, f"{path}/{kk}")
                else:
                    flat[path or "/"] = node if isinstance(node, (bool, int, float, str, type(None))) else repr(node)

            try:
                walk(cs.get("/"), "")
            except Exception as e:  # 진단 실패는 기록만
                flat = {"__error__": repr(e)}
            with open(DUMP_SETTINGS, "w", encoding="utf-8") as f:
                json.dump(flat, f, ensure_ascii=False, indent=0, sort_keys=True, default=repr)
            print(f"[black] carb 설정 {len(flat)} 개를 {DUMP_SETTINGS} 에 적음", flush=True)
        from omnigibson.sensors.vision_sensor import VisionSensor

        inner = VisionSensor._get_obs  # 시간 계측 겉싸개가 있으면 그것까지 포함해 한 겹 더

        @functools.wraps(inner)
        def get_obs_black(self, *a, **kw):
            ret = inner(self, *a, **kw)
            if T.active:
                BLACK.after_get_obs(self, ret[0])
            return ret

        VisionSensor._get_obs = get_obs_black

    Ev.run = run

    orig_load_batch = Ev.load_batch

    @functools.wraps(orig_load_batch)
    def load_batch(self, *a, **kw):
        t0 = perf_counter()
        r = orig_load_batch(self, *a, **kw)
        T.oneoff["load_batch_s"] = T.oneoff.get("load_batch_s", 0.0) + perf_counter() - t0
        if gpu is not None:
            T.oneoff["gpu_mib_after_load"] = _gpu_used_mib()
        trace.on_run_start(self)
        return r

    Ev.load_batch = load_batch

    orig_init = Ev.__init__

    @functools.wraps(orig_init)
    def ev_init(self, *a, **kw):
        t0 = perf_counter()
        orig_init(self, *a, **kw)
        T.oneoff["evaluator_init_s"] = perf_counter() - t0
        T.oneoff["num_envs"] = self.num_envs

    Ev.__init__ = ev_init

    wrap(Ev, "_step_fn", "step", before=lambda a, kw: T.new_step())
    if RENDER_ITERS:
        # 진단용: 공식 _apply_actions 는 env.step(actions, n_render_iterations=1) 로 부른다(eval/evaluator.py:383).
        # 여기서만 N 으로 바꿔 렌더를 더 돌린다 -- 물리·판정은 렌더 횟수와 무관해야 한다(trace_compare 로 확인).
        from omnigibson.envs.env_base import Environment

        orig_env_step = Environment.step

        @functools.wraps(orig_env_step)
        def env_step(self, action, n_render_iterations=1):
            return orig_env_step(self, action, n_render_iterations=RENDER_ITERS)

        Environment.step = env_step
    if timing:
        wrap(Ev, "_batch_obs", "batch_obs")
        wrap(Ev, "_preprocess_obs", "preprocess_obs")
        wrap(Ev, "_write_video", "video")
        wrap_module_fn(E, "write_video", "encode")
        wrap(P.WebsocketPolicy, "forward", "policy")
        wrap(P.LocalPolicy, "forward", "policy")
        wrap(NU.WebsocketClientPolicy, "act", "client.act")

        orig_unpackb = NU.unpackb

        def unpackb(*a, **kw):
            tok = T.enter("unpack")
            try:
                d = orig_unpackb(*a, **kw)
            finally:
                T.exit(tok)
            if isinstance(d, dict) and "server_timing" in d:
                st = d["server_timing"]
                T.meta["server_infer_ms"].append(float(st.get("infer_ms", np.nan)))
                if "prev_total_ms" in st:
                    T.meta["server_prev_total_ms"].append(float(st["prev_total_ms"]))
            return d

        NU.unpackb = unpackb

        orig_ci = NU.WebsocketClientPolicy.__init__

        def ci(self, *a, **kw):
            orig_ci(self, *a, **kw)
            self._packer = _TimedPacker(self._packer)

        NU.WebsocketClientPolicy.__init__ = ci

        orig_wait = NU.WebsocketClientPolicy._wait_for_server

        def wait(self):
            conn, md = orig_wait(self)
            return _TimedConn(conn), md

        NU.WebsocketClientPolicy._wait_for_server = wait

    wrap(Ev, "_apply_actions", "apply_actions",
         after=lambda a, kw, ret: trace.on_apply(a[0], a[1], set(a[2]), ret))

    # --- og.shutdown 이 프로세스를 바로 끝내므로 __exit__ 직전에 저장 ---
    orig_exit = Ev.__exit__

    def save_all():
        if state["saved"]:
            return
        state["saved"] = True
        T.close_step()
        T.active = False
        trace.save(os.path.join(out_dir, "trace.npz"), os.path.join(out_dir, "actions.npz"))
        if BLACK is not None:
            with open(os.path.join(out_dir, "black_frames.json"), "w", encoding="utf-8") as f:
                json.dump(BLACK.summary(), f)
            print(f"[black] 검은 프레임/전체: {dict(BLACK.n_black)} / {dict(BLACK.n_total)}", flush=True)
        if timing:
            if gpu is not None:
                gpu.stop()
            res = {
                "argv": sys.argv,
                "oneoff": T.oneoff,
                "offthread_s": dict(T.offthread),
                "gpu_samples_mib": gpu.samples if gpu is not None else [],
                "wrapper_cost_s": measure_wrapper_cost(),
                "og_profilers": _og_profilers(og),
                "steps": T.steps,
            }
            path = os.path.join(out_dir, "timing.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(res, f)
            print(report(path), flush=True)

    def ev_exit(self, *a):
        try:
            save_all()
        except Exception as e:  # 기록 실패가 평가 종료를 막지 않게
            print(f"[eval_instrumented] 저장 실패: {e!r}", flush=True)
        return orig_exit(self, *a)

    Ev.__exit__ = ev_exit


def install_instance_seq(indices, eval_args, trace):
    """--instances-seq=0,1,2 : 한 프로세스에서 인스턴스를 차례로 (장면 로딩 한 번).

    공식 main 은 --num-envs 1 과 첫 인스턴스 하나로 돈다. main 이 부르는 첫 evaluator.run() 을 가로채,
    공식 run() 을 인스턴스마다 한 번씩 부른다(공식 --num-rollouts 가 같은 evaluator 로 run() 을 거듭 부르는 것과 같은 경로).
    결과는 인스턴스마다 <output-dir>\\i<번호>\\ (json\\, videos\\, trace.npz, actions.npz) -- 새 프로세스로 돈 판과 같은 모양.
    새 프로세스 결과와 비트 동일인지 확인하기 전까지는 개발용이다(tools/exp_run.py --reuse).
    """
    from omnigibson.eval import evaluator as E
    from omnigibson.eval.evaluator import resolve_instance_ids

    task = eval_args[eval_args.index("--task-name") + 1]
    mode = eval_args[eval_args.index("--mode") + 1] if "--mode" in eval_args else "public_test"
    out_dir = eval_args[eval_args.index("--output-dir") + 1]
    ids = resolve_instance_ids(task, indices, mode=mode)
    Ev = E.BatchedEvaluator
    inner_run = Ev.run  # 계측 겉싸개(install)까지 포함
    done = {"flag": False}

    @functools.wraps(inner_run)
    def seq_run(self, instances_to_run, write_video=False, video_path=None, metrics_dir=None, rollout_id=0, video_fps=30):
        if done["flag"]:  # --num-rollouts > 1 이면 두 번째부터는 공식 그대로
            return inner_run(self, instances_to_run, write_video=write_video, video_path=video_path,
                             metrics_dir=metrics_dir, rollout_id=rollout_id, video_fps=video_fps)
        done["flag"] = True
        merged = {}
        for k, (ix, iid) in enumerate(zip(indices, ids)):
            d = os.path.join(out_dir, f"i{ix}")
            os.makedirs(os.path.join(d, "json"), exist_ok=True)
            vd = os.path.join(d, "videos")
            if write_video:
                os.makedirs(vd, exist_ok=True)
            print(f"[instances-seq] 시작 인덱스 {ix} 인스턴스 {iid} ({k + 1}/{len(ids)}) t={time.time():.3f}", flush=True)
            r = inner_run(self, [int(iid)], write_video=write_video, video_path=vd, metrics_dir=os.path.join(d, "json"),
                          rollout_id=rollout_id, video_fps=video_fps)
            merged.update(r)
            print(f"[instances-seq] 끝 인덱스 {ix} 인스턴스 {iid} t={time.time():.3f}", flush=True)
            if trace.rows:
                trace.save(os.path.join(d, "trace.npz"), os.path.join(d, "actions.npz"))
                trace.rows, trace.images = [], {}
            if BLACK is not None:
                with open(os.path.join(d, "black_frames.json"), "w", encoding="utf-8") as f:
                    json.dump(BLACK.summary(), f)
                BLACK.n_black.clear(), BLACK.n_total.clear(), BLACK.rows.clear(), BLACK.times.clear()
        return merged

    Ev.run = seq_run


def install_native_policy(spec):
    """--native-policy=모듈:함수 : 평가기 프로세스 안에서 도는 정책을 공식 load_policy 자리에 넣는다
    (모듈은 sys.path 에서 찾는다 — PYTHONPATH 로 준다).

    함수(cfg) 는 공식 정책과 같은 모양의 객체를 돌려줘야 한다: forward(obs=배치 관측) -> (num_envs, action_dim) 텐서, reset(),
    (있으면) set_action_dim(n). 평가기 쪽 코드는 그대로다 -- 정책은 우리 제출물이지 평가기가 아니다.
    """
    import importlib

    from omnigibson.eval import evaluator as E

    mod, fn = spec.split(":", 1)
    factory = getattr(importlib.import_module(mod), fn)
    held = {}

    def load_policy(self):
        policy = factory(self.cfg)
        if hasattr(policy, "set_action_dim"):
            policy.set_action_dim(self.instance_eval_states[0].env_accessor.robot.action_dim)
        held["p"] = policy
        print(f"[native-policy] {spec} 를 평가기 프로세스 안 정책으로 씀", flush=True)
        return policy

    E.BatchedEvaluator.load_policy = load_policy

    # 정책 쪽 기록(예: 정책의 스텝 CSV)을 og.shutdown 전에 내보낸다
    prev_exit = E.BatchedEvaluator.__exit__

    def ev_exit(self, *a):
        inner = getattr(held.get("p"), "policy", None) or held.get("p")
        if inner is not None and hasattr(inner, "flush"):
            try:
                inner.flush()
            except Exception as e:
                print(f"[native-policy] flush 실패: {e!r}", flush=True)
        return prev_exit(self, *a)

    E.BatchedEvaluator.__exit__ = ev_exit


def _og_profilers(og):
    out = {}
    for k in ("_step_profiler", "_pre_physics_step_profiler", "_post_physics_step_profiler", "_non_physics_step_profiler"):
        p = getattr(og.sim, k, None)
        if p is not None:
            out[k] = {"total_s": p.total_time, "calls": p.call_count}
    return out


def measure_wrapper_cost(n=200000):
    class _C:
        def f(self):
            return None

    c = _C()
    t0 = perf_counter()
    for _ in range(n):
        c.f()
    base = perf_counter() - t0
    was = T.active
    T.active = False
    wrap(_C, "f", "__bench__")
    t0 = perf_counter()
    for _ in range(n):
        c.f()
    wrapped = perf_counter() - t0
    T.active = was
    return max(wrapped - base, 0.0) / n


# ----------------------------------------------------------------------------------------------------------------------
# 보고서
# ----------------------------------------------------------------------------------------------------------------------
def report(path, skip=20):
    d = json.load(open(path, encoding="utf-8"))
    steps = d["steps"][skip:]
    if not steps:
        return "스텝 없음"
    n = len(steps)
    wall = np.array([s["wall"] for s in steps])
    paths = sorted({p for s in steps for p in s["excl"]})
    rows = []
    for p in paths:
        v = np.array([s["excl"].get(p, 0.0) for s in steps])
        c = np.array([s["calls"].get(p, 0) for s in steps])
        rows.append((v.mean(), p, np.percentile(v, 95), c.mean()))
    rows.sort(reverse=True)
    infer = [max(s.get("server_infer_ms", [0.0]) or [0.0]) for s in steps]
    inf_mask = np.array([x > 5.0 for x in infer])
    lines = [
        f"# {path}",
        f"스텝 {n} (앞 {skip} 스텝 제외), 스텝당 벽시계 평균 {wall.mean() * 1e3:.1f} ms (중앙 {np.median(wall) * 1e3:.1f}, "
        f"p95 {np.percentile(wall, 95) * 1e3:.1f}) = {1.0 / wall.mean():.2f} 스텝/초",
        f"한 번만: {json.dumps(d.get('oneoff', {}), ensure_ascii=False)}",
        f"겉싸개 비용 {d.get('wrapper_cost_s', 0) * 1e6:.2f} us/호출",
        "",
        "| 구간(호출 경로) | 평균 ms/스텝 | 비중 | p95 ms | 호출/스텝 |",
        "|---|---:|---:|---:|---:|",
    ]
    total_attr = 0.0
    for m, p, p95, c in rows:
        total_attr += m
        lines.append(f"| {p} | {m * 1e3:.2f} | {m / wall.mean() * 100:.1f}% | {p95 * 1e3:.2f} | {c:.1f} |")
    lines.append(f"| (어디에도 안 잡힌 시간: 루프·기록) | {(wall.mean() - total_attr) * 1e3:.2f} | "
                 f"{(wall.mean() - total_attr) / wall.mean() * 100:.1f}% | | |")
    if inf_mask.any():
        pol = np.array([sum(v for k, v in s["excl"].items() if k.startswith("step/policy")) for s in steps])
        inf_ms = np.array(infer)
        lines += [
            "",
            f"정책 추론 스텝 {inf_mask.sum()} / {n}: 정책 왕복 평균 {pol[inf_mask].mean() * 1e3:.1f} ms "
            f"(서버 추론 {inf_ms[inf_mask].mean():.1f} ms), 나머지 스텝 {pol[~inf_mask].mean() * 1e3:.1f} ms "
            f"(서버 추론 {inf_ms[~inf_mask].mean():.2f} ms)",
        ]
    g = [m for _, m in d.get("gpu_samples_mib", []) if m > 0]
    if g:
        lines.append(f"GPU 사용량(전체) 최대 {max(g)} MiB, 로드 후 {d['oneoff'].get('gpu_mib_after_load')} MiB, "
                     f"시작 전 {d['oneoff'].get('gpu_mib_before')} MiB")
    return "\n".join(lines)


# 설계 문서의 병목 항목 <- 호출 경로. (이름, 맨 끝 구간 이름들, 경로 어디든 들어 있으면 되는 말들) 을 위에서부터 처음 맞는 것.
GROUPS = [
    ("통신: 웹소켓 왕복(서버 계산 뺌)", ("ws.recv", "ws.send", "pack", "unpack", "client.act"), ()),
    ("통신: 관측 묶기·정책 겉층", ("batch_obs", "policy"), ()),
    ("GPU 대기(렌더 끝나길 기다림)", ("wp.synchronize_stream",), ()),
    ("물리 PhysX", ("physx",), ()),
    ("보조 잡기(robot.post_step)", ("robot.post_step(grasp)",), ()),
    ("컨트롤러(pre-physics 콜백)", (), ("pre_physics(controllers)",)),
    ("접촉 읽기(post-physics 콜백)", (), ("post_physics(contacts)",)),
    ("렌더 제출·Kit 갱신(app.update 자기 시간)", ("app.update",), ()),
    ("물체 상태 캐시(warp, CPU 쪽)", (), ("state_caches(warp)",)),
    ("물체 상태 update·시스템·전이 규칙", ("object_state.update", "systems.update", "transition_rules"), ()),
    ("update_visuals(물체마다)", ("update_visuals",), ()),
    ("sim.step 나머지(non_physics 자기 시간 등)", ("non_physics", "sim.step", "sim.render"), ()),
    ("카메라 읽기 GPU->CPU", ("camera_read",), ()),
    ("proprio 계산", (), ("/proprio",)),
    ("BDDL 목표 판정(종료+보상)", ("bddl_goal(termination)", "bddl_goal(reward)"), ()),
    ("task.step 나머지(info 복사 등)", ("task.step",), ()),
    ("관측 전처리(카메라 상대 pose)", (), ("preprocess_obs",)),
    ("지표(metric)", (), ("metric.",)),
    ("행동 적용(컨트롤러 목표 설정)", ("apply_action",), ("/apply_action/",)),
    ("영상 저장(크기조정+x264)", (), ("video",)),
    ("조명 동기화(조명 과제만)", ("light_sync",), ()),
    ("그 밖의 평가기 파이썬(env.step·루프)", (), ("",)),
]


def _group(p):
    leaf = p.rsplit("/", 1)[-1]
    for name, leaves, subs in GROUPS:
        if leaf in leaves or any(x in p for x in subs):
            return name
    return GROUPS[-1][0]


def summarize(path, skip=20):
    """호출 경로를 설계 문서의 병목 항목으로 묶어 스텝당 ms·비중을 낸다. 서버 추론 시간은 통신에서 빼서 따로."""
    d = json.load(open(path, encoding="utf-8"))
    steps = d["steps"][skip:]
    n = len(steps)
    wall = np.array([s["wall"] for s in steps])
    tot = defaultdict(float)
    for s in steps:
        for p, v in s["excl"].items():
            tot[_group(p)] += v
    infer = sum(sum(s.get("server_infer_ms", []) or []) for s in steps) / 1e3
    tot["통신: 웹소켓 왕복(서버 계산 뺌)"] -= infer
    tot["정책 추론(서버 계산)"] = infer
    tot["어디에도 안 잡힘(루프·기록)"] = wall.sum() - sum(v for k, v in tot.items())
    rows = sorted(((v / n * 1e3, k) for k, v in tot.items()), reverse=True)
    out = [f"# {os.path.basename(os.path.dirname(os.path.abspath(path)))}: 스텝 {n}, 평균 {wall.mean() * 1e3:.1f} ms/스텝 "
           f"({1 / wall.mean():.2f} 스텝/초, 환경 {d['oneoff'].get('num_envs')})", "",
           "| 병목 항목 | ms/스텝 | 비중 |", "|---|---:|---:|"]
    out += [f"| {k} | {v:.2f} | {v / (wall.mean() * 1e3) * 100:.1f}% |" for v, k in rows if abs(v) >= 0.005]
    return "\n".join(out)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--summary":
        for f in argv[1:]:
            print(summarize(f))
        return
    if argv and argv[0] == "--report":
        skip = 20
        files = [a for a in argv[1:] if not a.startswith("--")]
        if "--skip" in argv:
            skip = int(argv[argv.index("--skip") + 1])
            files = [f for f in files if f != str(skip)]
        for f in files:
            print(report(f, skip))
        return
    ours, eval_args = (argv[: argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else ([], argv)
    timing, full_trace = "--timing" in ours, "--trace" in ours
    seq = next((o.split("=", 1)[1] for o in ours if o.startswith("--instances-seq=")), "")
    native = next((o.split("=", 1)[1] for o in ours if o.startswith("--native-policy=")), "")
    if seq:
        # 장면 재사용: 공식 main 에는 첫 인스턴스 하나(--num-envs 1)만 넘기고, 공식 evaluator.run() 을 인스턴스마다 다시 부른다
        # (공식 --num-rollouts 가 같은 evaluator 로 run() 을 거듭 부르는 것과 같은 경로). 결과는 인스턴스마다 하위 폴더에.
        idx = [int(x) for x in seq.split(",") if x.strip()]
        i = eval_args.index("--instance-indices")
        j = i + 1
        while j < len(eval_args) and not eval_args[j].startswith("--"):
            j += 1
        eval_args = eval_args[: i + 1] + [str(idx[0])] + eval_args[j:]
        if "--num-envs" in eval_args:
            eval_args[eval_args.index("--num-envs") + 1] = "1"
    global BLACK
    guard = ""
    for g in ("warn", "abort"):
        if f"--black-guard={g}" in ours:
            guard = g
    global RENDER_ITERS, DUMP_SETTINGS
    for o in ours:
        if o.startswith("--dump-settings="):
            DUMP_SETTINGS = o.split("=", 1)[1]
        if o.startswith("--kit-arg="):
            KIT_ARGS.append(o[len("--kit-arg="):])
        if o.startswith("--render-iters="):
            RENDER_ITERS = int(o.split("=", 1)[1])
        if o.startswith("--set="):
            k, v = o[len("--set="):].split("=", 1)
            SET_SETTINGS.append((k, v))
    if "--black-diag" in ours or guard or SET_SETTINGS or KIT_ARGS or DUMP_SETTINGS:
        BLACK = BlackFrameDiag(diag="--black-diag" in ours, guard=guard)
    out_dir = eval_args[eval_args.index("--output-dir") + 1] if "--output-dir" in eval_args else "/tmp/b1k_eval"
    os.makedirs(out_dir, exist_ok=True)
    gpu = None
    if timing:
        T.oneoff["gpu_mib_before"] = _gpu_used_mib()
        gpu = GpuMonitor()
        gpu.start()
    trace = TraceRecorder(full_trace)
    install(timing, trace, out_dir, gpu, deep="--deep" in ours)
    if seq:
        install_instance_seq(idx, eval_args, trace)
    if native:
        install_native_policy(native)
    if "--reset-user" in KIT_ARGS:
        # Kit 사용자 설정을 지우는 인자 -- 복원을 보장할 수 없어 받지 않는다(다른 Isaac Sim 환경에 영향 가능).
        KIT_ARGS.remove("--reset-user")
        print("[kit-arg] --reset-user 는 거부했다(사용자 설정 삭제 위험). 이 실행은 설정 그대로다.", flush=True)
    if KIT_ARGS:
        # 진단용: OmniGibson 은 Kit 을 띄울 때 sys.argv 를 넘긴다(simulator.py _launch_app 가 앞뒤로 저장·복원).
        # 평가기 인자 파싱이 끝난 뒤, Kit 을 띄우는 그 순간에만 인자를 덧붙인다.
        import omnigibson.simulator as S

        orig_launch = S._launch_app

        def launch_with_args(*a, **kw):
            # _launch_app 은 sys.argv 를 스크립트 이름만 남기고 지운다 -> SimulationApp 설정의 extra_args 로 넣는다.
            import isaacsim

            orig_init = isaacsim.SimulationApp.__init__

            def init(self, launch_config=None, *ia, **ikw):
                cfg = dict(launch_config or {})
                cfg["extra_args"] = list(cfg.get("extra_args", [])) + KIT_ARGS
                print(f"[kit-arg] extra_args={cfg['extra_args']}", flush=True)
                return orig_init(self, cfg, *ia, **ikw)

            isaacsim.SimulationApp.__init__ = init
            try:
                return orig_launch(*a, **kw)
            finally:
                isaacsim.SimulationApp.__init__ = orig_init

        S._launch_app = launch_with_args
    from omnigibson.eval import eval as ev

    sys.argv = ["omnigibson.eval.eval"] + eval_args
    ev.main()


if __name__ == "__main__":
    main()
