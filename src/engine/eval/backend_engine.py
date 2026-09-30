"""--backend engine v0 (S4, docs/엔진_자체구현.md 15절): 포팅 평가기의 시뮬레이터 자리를 이 엔진으로.

물리·제어기·판정 = 엔진(libengine_capi.so: PhysX 비계 + core/omni 제어기·판정). 관측 가공 = obs_engine.py(원본 식·원본 usdrt.Gf).
에피소드 앞(인스턴스 불러오기·가라앉히기·복원)은 행동과 무관하므로 그 인스턴스의 공식 기록을 장면 추출물로 재생한다(v0).

판 폴더(장면 추출물) = --scene-root 아래 <과제>/<인스턴스> 또는 환경변수 ENGINE_REC_DIR (한 판).
카메라 사슬(scope.json camera_chain)이 그 폴더에 없으면 ENGINE_CAM_CHAIN(같은 로봇의 scope.json) 에서 가져온다.
v0 한계: 판 1 개, 영상 관측은 0 영상(렌더 모듈 전), 판정 접촉 행렬은 기록(s3_*)이 있으면 그것, 없으면 접촉 없음(켜짐 안 됨).
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch as th

from engine_core import EngineCore
from facade import EngineCamera, EngineEnv, EngineRobot, EngineScene, EngineSim, EngineTask
from obs_engine import RobotView, base_pose, camera_world_gf, load_usdrt_gf, proprio

log = logging.getLogger("backend_engine")


class _Cam(EngineCamera):
    def __init__(self, env, name):
        self.env, self.name = env, name

    def get_position_orientation(self):
        return camera_world_gf(self.env.core, self.env.cams[self.name], self.env.Gf)


class _Robot(EngineRobot):
    def __init__(self, env, rv, cfg):
        self.env, self.rv = env, rv
        self.name = rv.name
        self.model = cfg.get("model", "r1pro")
        self.arm_names = ["left", "right"]
        self.action_dim = 23
        self.sensors = {n: _Cam(env, n) for n in rv.sensors}

    def get_position_orientation(self):
        return base_pose(self.env.core, self.rv)

    def get_eef_position(self, arm="default"):
        arm = "left" if arm == "default" else arm
        return th.as_tensor(self.env.core.body_pose(self.rv.eef[arm])[:3].copy())

    def get_joint_positions(self):
        return th.as_tensor(self.env.core.joint_state()[0])

    def get_joint_velocities(self):
        return th.as_tensor(self.env.core.joint_state()[1])


class _Obj:
    """BDDL 물체: get_position_orientation = 뿌리 링크(base_link) 세계 자세 (trace obj:: 와 비트 동일 확인)."""

    def __init__(self, env, path):
        self.env, self.path = env, path

    def get_position_orientation(self):
        p = self.env.core.body_pose(self.path)
        return th.as_tensor(p[:3].copy()), th.as_tensor(p[3:].copy())


class _ScenePrim:
    def get_position_orientation(self):
        return th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])


class _Sim(EngineSim):
    def get_rendering_dt(self):
        return 1.0 / 30.0


class _Task(EngineTask):
    def __init__(self, env, name):
        self.env = env
        self.activity_name = name
        self.object_scopes = [dict()]
        self.success = th.zeros(1, dtype=th.bool)

    def get_goal_option_satisfaction(self, env_idx):
        # radio 같은 목표 하나짜리: 선택지 하나 = [충족 여부] (TaskMetric q_score)
        return [[self.env.goal_list() != []]]


class EngineEnvV0(EngineEnv):
    def __init__(self, task_name, robot_cfg, robot_eval_cfg, camera_spec, max_steps, num_envs, scene_root=None, **_):
        assert num_envs == 1, "엔진 v0 은 판 1 개"
        self.num_envs = 1
        self.task_name = task_name
        self.robot_cfg = dict(robot_cfg, eval=robot_eval_cfg)
        self.max_steps = int(max_steps)
        self.scene_root = scene_root
        self.res = camera_spec["resolution"]
        self.role_of = {v: k for k, v in (robot_eval_cfg.get("camera_sensor_names") or {}).items()}
        self.sim = _Sim()
        self.task = _Task(self, task_name)
        self.core = None
        self.Gf = load_usdrt_gf()
        if self.Gf is None:
            raise SystemExit("usdrt.Gf 를 못 불러옴: src/engine/eval/run_ported_engine.sh 로 실행 (LD_LIBRARY_PATH)")
        self.steps = 0
        s = EngineScene()
        s.robots = []
        s._scene_prim = _ScenePrim()  # trace static::scene_offset (판 0 은 원점)
        self.scenes = [s]
        # 평가기 __init__ 이 판을 불러오기 전에 로봇 이름·센서를 본다 -> 판 폴더의 scope.json 에서 미리 만든다 (물리는 load_instances 에서)
        rec = os.environ.get("ENGINE_REC_DIR")
        if rec:
            self._bind_static(rec)

    def _bind_static(self, rec):
        self.rv = RobotView(rec)
        sc = json.load(open(os.path.join(rec, "scope.json")))[0]
        cams = sc["robot"].get("camera_chain")
        if not cams:
            alt = os.environ.get("ENGINE_CAM_CHAIN")
            if not alt:
                raise SystemExit("scope.json 에 camera_chain 이 없다: ENGINE_CAM_CHAIN=<같은 로봇 scope.json>")
            cams = json.load(open(alt))[0]["robot"]["camera_chain"]
        self.cams = cams
        self.scenes[0].robots = [_Robot(self, self.rv, self.robot_cfg)]
        scope = {}
        for bn, o in sc["objects"].items():
            scope[bn] = self.scenes[0].robots[0] if o.get("cls") == "Robot" else _Obj(self, o["prim_path"] + "/base_link")
        self.task.object_scopes = [scope]

    # ---- 판 불러오기 ----
    def _rec_dir(self, instance_id):
        d = os.environ.get("ENGINE_REC_DIR")
        if d:
            return d
        return os.path.join(self.scene_root or "", self.task_name, str(instance_id))

    def apply_eval_settings(self, base_link_mass, head_sensor, head_aperture):
        pass  # 바닥 질량 250·머리 조리개: 장면 추출물(공식 기록)에 이미 들어 있다

    def load_instances(self, env_idx_to_instance, mode):
        rec = self._rec_dir(env_idx_to_instance[0])
        if self.core is not None:
            self.core.close()
        self.core = EngineCore(rec)
        self._bind_static(rec)
        self.task.success = th.zeros(1, dtype=th.bool)
        self.steps = 0

    # ---- 관측 ----
    def goal_list(self):
        g = self.core.goal() if self.core is not None else "[]"
        return json.loads(g)

    def _obs(self):
        T = _transform_utils()
        pr, _ = proprio(self.core, self.rv, T)
        o = {"proprio": pr}
        for sname in self.rv.sensors:
            h, w = self.res.get(self.role_of.get(sname, ""), (224, 224))
            o[sname] = {"rgb": th.zeros((h, w, 4), dtype=th.uint8)}
        return {self.rv.name: o}

    def reset(self, env_indices=None):
        self.steps = 0
        return [self._obs()], [{}]

    def get_obs(self, env_indices=None):
        return [self._obs()], [{}]

    def step(self, actions, n_render_iterations=1):
        a = actions[0] if hasattr(actions, "__len__") and len(actions) == 1 else actions
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        self.core.step(np.asarray(a, np.float32).reshape(-1))
        # Timeout(termination_conditions/timeout.py:23): episode_steps >= max_steps, episode_steps 는 이 스텝 전 값
        # (공식은 max_steps=500 에서 501 스텝을 돈다)
        before = self.steps
        self.steps += 1
        sat = self.goal_list()
        success = len(sat) > 0
        self.task.success = th.tensor([success])
        term = th.tensor([success])
        trunc = th.tensor([before >= self.max_steps])
        info = [{"done": {"goal_status": {"satisfied": sat, "unsatisfied": [i for i in [0] if i not in sat]}}}]
        return [self._obs()], th.zeros(1), term, trunc, info

    def close(self):
        if self.core is not None:
            self.core.report()
            self.core.close()
            self.core = None


_T = None


def _transform_utils():
    global _T
    if _T is None:
        import omnigibson.utils.transform_utils as T

        _T = T
    return _T
