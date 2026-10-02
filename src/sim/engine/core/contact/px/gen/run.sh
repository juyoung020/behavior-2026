#!/bin/bash
# 번역 다시 돌리기 (WSL): bash /mnt/c/behavior-2026/src/sim/engine/core/contact/px/gen/run.sh
set -e
D=$(cd "$(dirname "$0")" && pwd)
python3 "$D/translate.py" "${PHYSX_ROOT:-$HOME/engine-deps/physx-107.3-omni/physx}" "$D/.." "$D/jobs.txt"
