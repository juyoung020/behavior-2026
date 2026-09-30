"""포팅 평가기가 엔진에 요구하는 겉모습 — OmniGibson 과 같은 이름·뜻 (docs/엔진_자체구현.md 10.2).

공식 BatchedEvaluator(omnigibson/eval/evaluator.py)가 시뮬레이터에 닿는 곳만 이 겉모습으로 바꾼다.
판 묶음 루프(evaluate_instances_batched), 정책 웹소켓(WebsocketPolicy), 지표(AgentMetric·TaskMetric), 결과 JSON 은
공식 코드를 그대로 부른다. 그래서 엔진이 이 겉모습을 공식과 같은 값으로 채우면 결과 JSON 도 같아진다.

엔진(C++/CUDA)의 파이썬 바인딩이 아래 클래스를 채운다. 값은 torch 텐서(CPU), 형·모양·좌표계는 OmniGibson 과 같다
(위치 float32 (3,), 쿼터니언 xyzw float32 (4,), 세계 좌표). 공식에서 누가 부르는지를 줄마다 적었다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


class EngineCamera:
    """robot.sensors 의 값. 공식 _preprocess_obs(evaluator.py:576)가 cam_rel_poses 계산에 자세만 읽는다."""

    def get_position_orientation(self) -> Tuple[Any, Any]:
        raise NotImplementedError


class EngineRobot:
    name: str  # 예 "robot_r1" (관측 키 앞머리, get_robot_camera_names)
    model: str  # "r1pro"
    arm_names: List[str]  # ["left", "right"] (AgentMetric)
    action_dim: int  # load_policy·_step_fn
    sensors: Dict[str, EngineCamera]  # 이름 -> 카메라 (_validate_robot_eval_config 는 이름만 본다)

    def get_position_orientation(self) -> Tuple[Any, Any]:  # 바닥 링크 세계 자세 (AgentMetric, _preprocess_obs, trace)
        raise NotImplementedError

    def get_eef_position(self, arm: str) -> Any:  # 손끝 세계 위치 (AgentMetric)
        raise NotImplementedError

    def get_joint_positions(self) -> Any:  # trace robot_qpos
        raise NotImplementedError

    def get_joint_velocities(self) -> Any:  # trace robot_qvel
        raise NotImplementedError


class EngineScene:
    robots: List[EngineRobot]  # 판마다 로봇 하나 (InstanceEnvAccessor.robot)


class EngineTask:
    activity_name: str
    object_scopes: List[Dict[str, Any]]  # 판마다 {BDDL 이름: 물체(get_position_orientation) 또는 None} (trace obj::)
    success: Any  # (num_envs,) bool (InstanceEnvAccessor.success -> 결과 JSON success, q_score)

    def get_goal_option_satisfaction(self, env_idx: int) -> List[List[bool]]:  # TaskMetric q_score (처음·끝)
        raise NotImplementedError


class EngineSim:
    """og.sim 자리. 공식 코드가 og.sim 에서 부르는 것 중 평가 경로에 남는 두 개."""

    def get_rendering_dt(self) -> float:  # TaskMetric.reset -> time.simulator_time
        raise NotImplementedError

    def update_handles(self) -> None:  # BatchedEvaluator.__init__ 끝
        pass


class EngineEnv:
    """og.Environment 자리 (판 N 개를 한 번에)."""

    num_envs: int
    scenes: List[EngineScene]
    task: EngineTask
    sim: EngineSim

    def apply_eval_settings(self, base_link_mass: float, head_sensor: str | None, head_aperture: float) -> None:
        """공식 _apply_robot_eval_settings(evaluator.py:340): R1/R1Pro 바닥 링크 질량 250, 머리 카메라 수평 조리개 40."""
        raise NotImplementedError

    def load_instances(self, env_idx_to_instance: Dict[int, int], mode: str) -> None:
        """공식 _load_instance_state(:445) + _settle_and_finalize(:513) 와 같은 물리:
        로봇 기본 자세로 되돌림 -> 인스턴스 TRO 상태(로봇 바닥 자세·물체 상태) 적용 -> 물리 25 서브스텝 동안 과제 물체 멈춤 유지
        -> 그 상태를 판의 '처음 상태'로 저장. 뒤따르는 reset(env_indices) 가 이 처음 상태로 되돌린다."""
        raise NotImplementedError

    def reset(self, env_indices: Sequence[int] | None = None) -> Tuple[List[dict], List[dict]]:
        """og.Environment.reset 과 같은 관측 구조: 판마다 {로봇이름: {"proprio": .., 센서이름: {"rgb": ..}}}."""
        raise NotImplementedError

    def get_obs(self, env_indices: Sequence[int] | None = None) -> Tuple[List[dict], List[dict]]:
        raise NotImplementedError

    def step(self, actions: Any, n_render_iterations: int = 1) -> Tuple[List[dict], Any, Any, Any, List[dict]]:
        """(관측 목록, 보상, terminated(num_envs,), truncated(num_envs,), info 목록).
        terminated = 과제 성공 등 종료 조건, truncated = 에피소드 스텝 >= max_steps (Timeout, termination_conditions/timeout.py:20).
        info[i]["done"]["goal_status"] = {"satisfied": [...], "unsatisfied": [...]} (trace goal_satisfied)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


def load_backend(name: str, **kw) -> EngineEnv:
    """--backend 이름 -> EngineEnv.
    engine : 이 엔진(층 2 CUDA)의 파이썬 바인딩 모듈 engine_eval (빌드되면 PYTHONPATH 로)
    dummy  : 물리 없는 가짜 (평가기 겉껍데기 시험용 — 정책 통신·판 묶음·지표·JSON 경로만 확인)"""
    if name == "dummy":
        from backend_dummy import DummyEnv

        return DummyEnv(**kw)
    if name == "engine":
        # v0: PhysX 비계 + core/omni (libengine_capi.so, ctypes). 모듈이 층 2 로 합쳐지면 같은 겉모습의 CUDA 바인딩으로 바꾼다.
        from backend_engine import EngineEnvV0

        return EngineEnvV0(**kw)
    raise SystemExit(f"모르는 backend: {name}")
