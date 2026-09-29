"""LLM 이 부르는 도구. 정의는 OpenAI 호환 tools 스키마, 실행은 Toolbox 메서드.

graph_query 는 meridian 통합 뒤, issue_command / check_done 은 π0.5 다리가 생긴 뒤 연결한다.
"""
from .memory import Memory
from .state import AgentState

SCHEMAS = [
    {"type": "function", "function": {
        "name": "graph_query",
        "description": "Find objects in the scene graph by text. Returns id, category, position relative to the robot, room, whether already handled, original position.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "robot_state",
        "description": "Current robot pose (odometry from episode start), what each hand holds, steps left.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "issue_command",
        "description": "Send one step to the low-level policy and run it until done or budget.",
        "parameters": {"type": "object", "properties": {
            "skill": {"type": "string"}, "target": {"type": "string"}, "support": {"type": "string"},
            "purpose": {"type": "string"}, "expected": {"type": "string"},
            "goal": {"type": "object"}, "budget_steps": {"type": "integer"}},
            "required": ["skill", "target"]}}},
    {"type": "function", "function": {
        "name": "check_done",
        "description": "Check whether the current step succeeded, using scene graph changes and the head camera.",
        "parameters": {"type": "object", "properties": {"expect": {"type": "string"}}, "required": ["expect"]}}},
    {"type": "function", "function": {
        "name": "goal_status",
        "description": "Which BDDL goal conditions are still unsatisfied (as far as the agent can observe).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "summarize",
        "description": "Replace the old step log with a short summary to keep context small.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


class Toolbox:
    def __init__(self, state: AgentState, memory: Memory, graph=None, executor=None):
        self.state, self.memory = state, memory
        self.graph = graph  # meridian Graphcore 조회 객체 (통합 뒤)
        self.executor = executor  # π0.5 다리: 지시를 넣고 실행 결과를 돌려줌 (만든 뒤)

    def call(self, name: str, args: dict):
        return getattr(self, name)(**args)

    def graph_query(self, text: str):
        if self.graph is None:
            raise NotImplementedError("meridian 통합 뒤 연결")
        return self.graph.query(text)

    def robot_state(self):
        s = self.state
        return {"pose": vars(s.pose), "holding": s.holding, "steps_left": s.steps_left}

    def issue_command(self, **kwargs):
        if self.executor is None:
            raise NotImplementedError("π0.5 다리 만든 뒤 연결")
        return self.executor.run(kwargs)

    def check_done(self, expect: str):
        if self.executor is None:
            raise NotImplementedError("π0.5 다리 만든 뒤 연결")
        return self.executor.check(expect)

    def goal_status(self):
        return {"goals_left": self.state.goals_left}

    def summarize(self, summary: str):
        self.memory.summary = summary
        old = self.memory.to_summarize()
        self.memory.records = self.memory.records[len(old):]
        return {"ok": True, "dropped": len(old)}
