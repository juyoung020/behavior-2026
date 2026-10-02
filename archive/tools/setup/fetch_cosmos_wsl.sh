#!/usr/bin/env bash
# GR00T N1.7 이 쓰는 게이트 백본 nvidia/Cosmos-Reason2-2B 를 WSL HuggingFace 캐시에 받는다.
# (juyoung02 계정 게이트 동의 완료 2026-09-29, 토큰은 ~/.cache/huggingface/token)
# 실행: wsl -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/tools/setup/fetch_cosmos_wsl.sh
set -euo pipefail
cd ~/openpi
~/.local/bin/uv run python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("nvidia/Cosmos-Reason2-2B", max_workers=4)
print("DONE", p)
PY
du -sh ~/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B
