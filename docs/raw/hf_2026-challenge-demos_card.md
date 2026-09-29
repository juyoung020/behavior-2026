# HuggingFace behavior-1k/2026-challenge-demos 데이터셋 카드·파일 구성

> 원본: https://huggingface.co/datasets/behavior-1k/2026-challenge-demos
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
## 데이터셋 카드 (README.md 원문)

---
license: mit
pretty_name: BEHAVIOR-1K 2026 Challenge Demos
tags:
- lerobot
- lerobot-v3
- behavior-1k
- omnigibson
- robotics
- imitation-learning
- embodied-ai
- manipulation
---

# BEHAVIOR-1K 2026 Challenge Demos

This dataset contains BEHAVIOR-1K 2026 challenge demonstration trajectories in LeRobotDataset v3 format.

## Dataset Statistics

- Tasks: 100
- Episodes: 20,000
- Frames: 210,916,774
- Size: approximately 3.0 TB
- Data shards: 955 Parquet files
- Video files: 17,093 MP4 files
- Video features: 6

## Format

The repository follows the LeRobotDataset v3 layout:

- `meta/info.json`: dataset schema and path templates
- `meta/stats.json`: feature statistics
- `meta/tasks.parquet`: task metadata
- `meta/episodes/`: per-episode metadata shards
- `data/`: frame-level Parquet shards
- `videos/`: encoded video shards

## Loading

The full dataset is large. For a quick smoke test, load a single episode without videos:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "behavior-1k/2026-challenge-demos",
    episodes=[0],
    download_videos=False,
)
```

To load the dataset at the published LeRobot v3 revision explicitly:

```python
dataset = LeRobotDataset(
    "behavior-1k/2026-challenge-demos",
    revision="v3.0",
)
```

## Versioning

The `v3.0` Hub tag is intended to match the LeRobot v3.0 codebase-compatible dataset revision.

## License

This dataset is released under the MIT License. See `LICENSE` for the full text.

## HuggingFace API 메타데이터

- lastModified: 2026-08-05T07:46:42.000Z
- gated: False · private: False
- 파일 수: 38157
- 총 크기(API usedStorage, 바이트): 3411677004450

| 파일 패턴 | 개수 | 합계 크기(GB) |
|---|---|---|
| `.gitattributes` | 1 | 0.00 |
| `LICENSE` | 1 | 0.00 |
| `README.md` | 1 | 0.00 |
| `annotations/skill_summary.csv` | 1 | 0.00 |
| `annotations/skill_type_summary.csv` | 1 | 0.00 |
| `annotations/task-N/episode_N.json` | 20000 | 0.21 |
| `data/chunk-N/file-N.parquet` | 955 | 74.56 |
| `meta/episodes/chunk-N/file-N.parquet` | 100 | 0.01 |
| `meta/info.json` | 1 | 0.00 |
| `meta/stats.json` | 1 | 0.00 |
| `meta/tasks.jsonl` | 1 | 0.00 |
| `meta/tasks.parquet` | 1 | 0.00 |
| `videos/observation.depth_linear.left_realsense_link_camera_N/chunk-N/file-N.mpN` | 2009 | 369.44 |
| `videos/observation.depth_linear.right_realsense_link_camera_N/chunk-N/file-N.mpN` | 1910 | 358.83 |
| `videos/observation.depth_linear.zed_link_camera_N/chunk-N/file-N.mpN` | 7888 | 1449.96 |
| `videos/observation.rgb.left_realsense_link_camera_N/chunk-N/file-N.mpN` | 1117 | 215.85 |
| `videos/observation.rgb.right_realsense_link_camera_N/chunk-N/file-N.mpN` | 1119 | 216.42 |
| `videos/observation.rgb.zed_link_camera_N/chunk-N/file-N.mpN` | 3050 | 569.99 |
