# agent — 상위 계획기 (Qwen3.5-9B 양자화, GPU)

`plan.md` 1~2절의 **판단** 층. 긴 계획·기억·단계 추적을 맡고, π0.5 에는 지금 단계 지시 한 줄만 넘긴다.
π0.5 는 과거를 기억하지 않으므로 "무엇을 했고 무엇이 남았나"는 전부 여기서 들고 있는다.

## 흐름

```
과제 지시 + BDDL 목표 ─┐
씬그래프(meridian) ────┼─▶ loop.py: 다음 단계 고르기 → issue_command → 실행 감시 → check_done
로봇 상태(오도메트리) ─┘                     │            ▲                       │
                                             ▼            │            다음 / 재시도 / 재계획
                                   instruction.py → π0.5 문장        memory.py 요약
```

- 매 스텝 부르지 않는다. 단계가 끝났거나 스텝 한도를 넘었을 때만 LLM 을 부른다.
- LLM 이 생각하는 동안 시뮬레이터 시간은 멈춰 있다(점수는 시뮬레이터 스텝 기준).

## 파일

| 파일 | 하는 일 |
|---|---|
| `state.py` | 에이전트 상태: 과제, 남은 목표 조건, 지금 단계, 남은 스텝, 로봇 위치(추정), 손에 든 것, 직전 결과 |
| `instruction.py` | 지시 계약(JSON) → π0.5 문장. 형식 4가지(과제 문장 / Comet 하위과제 / 목적·예상행동 / 숫자 명령)를 바꿔 끼울 수 있게 — `plan.md` 4.1 비교용 |
| `memory.py` | 단계 기록과 요약. 공간 기억은 씬그래프가 맡고 여기는 "한 일" 기억만 |
| `tools.py` | LLM 이 부르는 도구 정의(JSON 스키마)와 실행: `graph_query` · `robot_state` · `issue_command` · `check_done` · `goal_status` · `summarize` |
| `llm.py` | 로컬 LLM 서버(OpenAI 호환, 예: llama.cpp CUDA) 호출 |
| `loop.py` | 반복문: 목표를 다 채우거나 스텝이 떨어질 때까지 |
| `prompts/system.md` | 시스템 프롬프트 초안 |

## 아직 연결 안 된 것

- `graph_query`: meridian Graphcore 조회 — `src/meridian/`, `docs/meridian_통합설계.md` 통합 뒤.
- `issue_command` / 실행 감시: π0.5 서버와 평가기 사이 다리 — 아직 없음.
- 오늘 "있는 그대로" 실행은 2위 Comet 의 계획기 반복문(`refs/openpi-comet/src/openpi/shared/client.py:273-314`)을 그대로 쓴다. 여기 코드는 그 다음 단계.
