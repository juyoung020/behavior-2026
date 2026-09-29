"""에이전트 상태 — LLM 입력으로 들어가는 "지금 어디까지 왔나"."""
from dataclasses import dataclass, field


@dataclass
class RobotPose:
    """이번 판 출발점 기준 로봇 위치(base_qvel 적분). 시뮬레이터 정답 위치는 평가 때 금지."""
    x: float = 0.0  # m, 출발 방향 앞 +
    y: float = 0.0  # m, 왼쪽 +
    yaw: float = 0.0  # rad, 반시계 +


@dataclass
class AgentState:
    task_name: str  # 예: "turning_on_radio"
    task_prompt: str  # 과제 문장
    goals_left: list[str] = field(default_factory=list)  # BDDL 목표 조건 중 아직 안 채운 것
    stage: int = 0  # 지금 단계 번호
    steps_left: int = 0  # 제한시간까지 남은 시뮬레이터 스텝
    pose: RobotPose = field(default_factory=RobotPose)
    holding: dict[str, str | None] = field(default_factory=lambda: {"left": None, "right": None})
    last_result: str = ""  # "done" / "failed: …" / "timeout"
    retries: int = 0  # 지금 단계 재시도 횟수
