"""물리 없는 가짜 엔진 (포팅 평가기 겉껍데기 시험용).

로봇·물체는 제자리, 행동은 받기만 한다. 목표는 늘 미충족, max_steps 에서 truncated.
확인하는 것: 공식 인자 -> 판 묶음 루프 -> 정책 웹소켓 -> 지표 -> 결과 JSON 경로가 엔진 겉모습(facade.py)만으로 끝까지 도는지.
"""
from __future__ import annotations

import torch as th

from facade import EngineCamera, EngineEnv, EngineRobot, EngineScene, EngineSim, EngineTask


class _Cam(EngineCamera):
    def __init__(self, pos):
        self.pos = pos

    def get_position_orientation(self):
        return self.pos.clone(), th.tensor([0.0, 0.0, 0.0, 1.0])


class _Robot(EngineRobot):
    def __init__(self, cfg, action_dim=23):
        self.name = cfg["name"]
        self.model = cfg["model"]
        self.arm_names = ["left", "right"]
        self.action_dim = action_dim
        self.pos = th.tensor(cfg.get("position", [0.0, 0.0, 0.0]), dtype=th.float32)
        self.quat = th.tensor(cfg.get("orientation", [0.0, 0.0, 0.0, 1.0]), dtype=th.float32)
        cams = (cfg.get("eval") or {}).get("camera_sensor_names", {})
        self.sensors = {s: _Cam(self.pos + th.tensor([0.0, 0.0, 1.0])) for s in cams.values()}
        self.cam_shapes = {}

    def get_position_orientation(self):
        return self.pos.clone(), self.quat.clone()

    def get_eef_position(self, arm):
        return self.pos + th.tensor([0.3, 0.2 if arm == "left" else -0.2, 1.0])

    def get_joint_positions(self):
        return th.zeros(self.action_dim)

    def get_joint_velocities(self):
        return th.zeros(self.action_dim)


class _Sim(EngineSim):
    def get_rendering_dt(self):
        return 1.0 / 30.0


class _Task(EngineTask):
    def __init__(self, name, n):
        self.activity_name = name
        self.object_scopes = [dict() for _ in range(n)]
        self.success = th.zeros(n, dtype=th.bool)

    def get_goal_option_satisfaction(self, env_idx):
        return [[False]]


class DummyEnv(EngineEnv):
    def __init__(self, task_name, robot_cfg, robot_eval_cfg, camera_spec, max_steps, num_envs, **_):
        self.num_envs = num_envs
        cfg = dict(robot_cfg, eval=robot_eval_cfg)
        self.scenes = []
        for _ in range(num_envs):
            s = EngineScene()
            s.robots = [_Robot(cfg)]
            self.scenes.append(s)
        self.task = _Task(task_name, num_envs)
        self.sim = _Sim()
        self.max_steps = max_steps
        self.steps = th.zeros(num_envs, dtype=th.long)
        self.res = camera_spec["resolution"]
        self.role_of = {v: k for k, v in (robot_eval_cfg.get("camera_sensor_names") or {}).items()}

    def apply_eval_settings(self, base_link_mass, head_sensor, head_aperture):
        pass

    def load_instances(self, env_idx_to_instance, mode):
        pass

    def _obs(self, i):
        r = self.scenes[i].robots[0]
        o = {"proprio": th.zeros(8)}
        for sname in r.sensors:
            h, w = self.res.get(self.role_of.get(sname, ""), (224, 224))
            o[sname] = {"rgb": th.zeros((h, w, 4), dtype=th.uint8)}  # 키 = 센서 이름 (펼치면 "<로봇>::<센서>::rgb")
        return {r.name: o}

    def reset(self, env_indices=None):
        idx = range(self.num_envs) if env_indices is None else [int(i) for i in env_indices]
        for i in idx:
            self.steps[i] = 0
        return [self._obs(i) for i in idx], [{} for _ in idx]

    def get_obs(self, env_indices=None):
        idx = range(self.num_envs) if env_indices is None else [int(i) for i in env_indices]
        return [self._obs(i) for i in idx], [{} for _ in idx]

    def step(self, actions, n_render_iterations=1):
        self.steps += 1
        term = th.zeros(self.num_envs, dtype=th.bool)
        trunc = self.steps >= self.max_steps
        info = [{"done": {"goal_status": {"satisfied": [], "unsatisfied": [0]}}} for _ in range(self.num_envs)]
        return [self._obs(i) for i in range(self.num_envs)], th.zeros(self.num_envs), term, trunc, info
