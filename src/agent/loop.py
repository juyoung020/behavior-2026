"""계획기 반복문: 목표를 다 채우거나 스텝이 떨어질 때까지 LLM 이 도구를 부르며 단계를 진행한다."""
import json
from pathlib import Path

from .llm import LLM
from .memory import Memory
from .state import AgentState
from .tools import SCHEMAS, Toolbox

SYSTEM = (Path(__file__).parent / "prompts" / "system.md").read_text(encoding="utf-8")


def run(state: AgentState, llm: LLM, toolbox: Toolbox, max_turns: int = 200) -> AgentState:
    memory: Memory = toolbox.memory
    for _ in range(max_turns):
        if not state.goals_left or state.steps_left <= 0:
            break
        user = (
            f"Task: {state.task_prompt}\n"
            f"Goals left: {json.dumps(state.goals_left)}\n"
            f"Stage: {state.stage}, steps left: {state.steps_left}, last result: {state.last_result}\n"
            f"History:\n{memory.context()}"
        )
        msg = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], tools=SCHEMAS)
        for call in msg.get("tool_calls") or []:
            fn = call["function"]
            result = toolbox.call(fn["name"], json.loads(fn.get("arguments") or "{}"))
            state.last_result = json.dumps(result, ensure_ascii=False)[:500]
    return state
