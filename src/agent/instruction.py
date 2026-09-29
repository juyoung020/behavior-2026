"""지시 계약: LLM 이 내는 JSON → π0.5 가 받는 문장 한두 줄.

형식은 plan.md 4.1 에서 비교 중이라 바꿔 끼울 수 있게 둔다. π0.5 문장 입력은 토큰 200개(로봇 상태 포함)라 짧게.
좌표는 로봇 기준: 앞 +x, 왼쪽 +y, 반시계 +yaw. 단위 m·도.
"""
from dataclasses import dataclass, field

FORMATS = ("task", "subtask", "purpose", "metric")


@dataclass
class Instruction:
    skill: str  # 시연 주석 어휘 그대로: "move to", "pick up from", "place on", "press" …
    target: str  # 대상 물체(그래프 노드 이름). back / the other 는 LLM 이 이미 구체 물체로 풀어서 넣는다
    support: str | None = None  # 받침·목적지 물체 (pick up from / place on …)
    purpose: str = ""  # 목적: "라디오를 켜기 위해 라디오 앞으로 간다" 같은 한 줄(영어)
    expected: str = ""  # 예상 행동: "walk to the coffee table and face the radio"
    goal: dict = field(default_factory=dict)  # 숫자 명령용 {"x": m, "y": m, "yaw": deg}
    budget_steps: int = 300  # 이 단계에 줄 최대 스텝


def _metric(g: dict) -> str:
    parts = []
    if g.get("x"):
        parts.append(f"go {'forward' if g['x'] > 0 else 'backward'} {abs(g['x']):.1f} m")
    if g.get("y"):
        parts.append(f"{abs(g['y']):.1f} m to the {'left' if g['y'] > 0 else 'right'}")
    if g.get("yaw"):
        parts.append(f"turn {'left' if g['yaw'] > 0 else 'right'} {abs(g['yaw']):.0f} degrees")
    return ", ".join(parts)


def render(ins: Instruction, fmt: str, task_prompt: str = "") -> str:
    """Instruction → π0.5 prompt 문자열."""
    obj = ins.target.rsplit("_", 1)[0].replace("_", " ")  # radio_89 → radio
    sup = ins.support.rsplit("_", 1)[0].replace("_", " ") if ins.support else ""
    step = f"{ins.skill} {obj}" + (f" {sup}" if sup else "")
    if fmt == "task":  # ① 과제 문장만 (기본 체크포인트가 학습한 형태)
        return task_prompt
    if fmt == "subtask":  # ② Comet 기술/하위과제 문장 형태
        return step
    if fmt == "purpose":  # ③ 사용자 제안: 목적 / 예상 행동
        return f"Purpose: {ins.purpose}. Expected action: {ins.expected or step}."
    if fmt == "metric":  # ④ 숫자 명령
        m = _metric(ins.goal)
        return f"{step}: {m}" if m else step
    raise ValueError(f"fmt 는 {FORMATS} 중 하나: {fmt}")
