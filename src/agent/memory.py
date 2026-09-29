"""한 일 기억. 공간 기억(물체가 어디 있나, 처음 자리)은 씬그래프가 맡는다.

단계가 끝날 때마다 기록을 남기고, LLM 에는 최근 몇 단계만 그대로 + 나머지는 요약 몇 줄로 넣는다
(시연 기준 과제당 단계 중앙 17, 최대 74 — docs/π05_인지연결_설계.md 2.6).
"""
from dataclasses import dataclass, field


@dataclass
class StepRecord:
    stage: int
    prompt: str  # π0.5 에 넣은 문장
    result: str  # "done" / "failed: …" / "timeout"
    steps_used: int
    note: str = ""  # check_done 이 본 것 (예: "radio lifted 0.2 m")


@dataclass
class Memory:
    records: list[StepRecord] = field(default_factory=list)
    summary: str = ""  # 오래된 기록의 요약 (LLM 이 summarize 도구로 갱신)
    keep_recent: int = 4

    def add(self, rec: StepRecord) -> None:
        self.records.append(rec)

    def context(self) -> str:
        """LLM 입력용 문자열: 요약 + 최근 기록."""
        recent = self.records[-self.keep_recent:]
        lines = [f"[summary] {self.summary}"] if self.summary else []
        lines += [f"[{r.stage}] {r.prompt} -> {r.result} ({r.steps_used} steps) {r.note}".rstrip() for r in recent]
        return "\n".join(lines)

    def to_summarize(self) -> list[StepRecord]:
        """요약에 넘길 오래된 기록 (최근 keep_recent 개 제외)."""
        return self.records[: -self.keep_recent] if len(self.records) > self.keep_recent else []
