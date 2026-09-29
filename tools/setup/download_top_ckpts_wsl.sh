#!/usr/bin/env bash
# 2025 상위 팀 공개 체크포인트(추론용만)를 WSL ~/checkpoints 에 받는다. 안 A = 1위 제출 4개 + Comet pt50, 약 63 GB.
# 근거·크기: docs/2025상위팀_깃허브.md 1.2절. 끊겨도 다시 돌리면 이어받는다.
# 실행 (Windows): wsl -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/tools/setup/download_top_ckpts_wsl.sh
set -euo pipefail
LOG=/mnt/c/behavior-2026/logs/download_top_ckpts_$(date +%Y%m%d_%H%M).log
~/openpi/.venv/bin/python - <<'PY' 2>&1 | tee "$LOG"
import os
from huggingface_hub import snapshot_download
base = os.path.expanduser("~/checkpoints")
jobs = [
    ("IliaLarchenko/behavior_submission", "behavior_submission", dict(ignore_patterns=["replay.mp4"])),
    ("sunshk/openpi_comet", "openpi_comet", dict(allow_patterns=["pi05-b1kpt50-cs32/**"])),
]
for repo, name, kw in jobs:
    print(f"== {repo} -> {base}/{name}", flush=True)
    snapshot_download(repo, local_dir=f"{base}/{name}", max_workers=8, **kw)
    print(f"== {repo} 완료", flush=True)
print("=== DOWNLOAD DONE ===", flush=True)
PY
du -sh ~/checkpoints/* | tee -a "$LOG"
