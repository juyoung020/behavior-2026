# 리더보드 Space 파일 README.md

> 원본: https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard/blob/main/README.md
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
---
title: BEHAVIOR-1K 2026 Challenge Leaderboard
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.35.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
---

# BEHAVIOR-1K 2026 Challenge Leaderboard

This Space hosts the public leaderboard and submission portal for the
BEHAVIOR-1K 2026 Challenge, plus the archived 2025 challenge leaderboard.
The submission portal is available at
`https://behavior-1k-2026-challenge-leaderboard.hf.space/submit`.

Evaluation is run manually by the challenge organizers on the official
OmniGibson setup. Public submission metadata and verified results are stored in
the `data/` directory in this repository.

Portal responses are written as JSON files to the private Hugging Face dataset
configured by `SUBMISSION_DATASET_REPO`. The Space must also define
`HF_WRITE_TOKEN` as a secret with write access to that dataset.
Responses for this challenge are stored under `2026/submissions/` so the same
dataset can be reused for future challenge years.

## Updating Results

Append one JSON object per line to:

- `data/submissions.jsonl` for public submission metadata
- `data/results.jsonl` for aggregate verified scores
- `data/per_task_results.jsonl` for optional task-level breakdowns

Refresh self-reported aggregate scores from the private submission artifacts:

```bash
HF_TOKEN=<private-dataset-read-token> python scripts/extract_self_reported_scores.py
```

This writes `data/self_reported_results.jsonl`. Scores are averaged over all
100 challenge tasks, with unsubmitted tasks contributing zero. Review the
extracted task and episode counts, then commit that file with the leaderboard
update.

Then commit and push the Space repository.
