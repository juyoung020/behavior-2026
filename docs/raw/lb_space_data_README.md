# 리더보드 Space 파일 data/README.md

> 원본: https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard/blob/main/data/README.md
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
# Public Data Schema

All files are newline-delimited JSON. Empty files are valid.

`2025_results.jsonl` is the archived final leaderboard from the 1st BEHAVIOR
Challenge. The other files contain live 2026 challenge data.

## `submissions.jsonl`

Public metadata for submitted systems. Do not include private artifacts,
credentials, unpublished model weights, private logs, or local file paths.

Required fields:

- `submission_id`: stable unique ID, e.g. `teamname-method-2026-06-22`
- `team`: team or organization name
- `method`: system name
- `submitted_at`: ISO date or timestamp
- `status`: `submitted`, `running`, `failed`, or `verified`

Optional fields:

- `affiliation`
- `contact`
- `paper_url`
- `code_url`
- `model_url`
- `notes`

Example:

```json
{"submission_id":"example-team-baseline-2026-06-22","team":"Example Team","method":"Baseline","submitted_at":"2026-06-22","status":"verified","paper_url":"","code_url":"","model_url":""}
```

## `results.jsonl`

Aggregate verified leaderboard results.

Required fields:

- `submission_id`: matches `submissions.jsonl`
- `team`
- `method`
- `evaluated_at`: ISO date or timestamp
- `score`: primary ranking score, higher is better
- `success_rate`
- `q_score`
- `time_score`
- `efficiency_score`
- `num_tasks`
- `num_episodes`
- `verified`: boolean

Optional fields:

- `rank_note`
- `report_url`
- `video_url`
- `logs_url`

Example:

```json
{"submission_id":"example-team-baseline-2026-06-22","team":"Example Team","method":"Baseline","evaluated_at":"2026-06-22","score":0.0,"success_rate":0.0,"q_score":0.0,"time_score":0.0,"efficiency_score":0.0,"num_tasks":50,"num_episodes":500,"verified":true}
```

## `per_task_results.jsonl`

Optional task-level results.

Required fields:

- `submission_id`
- `task`
- `score`
- `success_rate`
- `q_score`
- `time_score`
- `efficiency_score`
- `num_episodes`

Optional fields:

- `mean_steps`
- `mean_time`

Example:

```json
{"submission_id":"example-team-baseline-2026-06-22","task":"turning_on_radio","score":0.0,"success_rate":0.0,"q_score":0.0,"time_score":0.0,"efficiency_score":0.0,"num_episodes":10}
```
