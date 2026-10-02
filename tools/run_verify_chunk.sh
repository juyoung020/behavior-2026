#!/usr/bin/env bash
# 행동 묶음 재생(src/sim/fasteval/pi05_chunk_server.py)이 공식 경로와 행동이 비트 단위로 같은지 확인 (시뮬레이터 없이, GPU 에 π0.5 만 올림).
# 실행: wsl -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/tools/run_verify_chunk.sh [추가 인자]
set -euo pipefail
CKPT=~/checkpoints/pi05_turning_on_radio/pi05_turn_on_the_radio
LOG=/mnt/c/behavior-2026/logs/verify_chunk_$(date +%Y%m%d_%H%M).log
cd ~/openpi
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
.venv/bin/python /mnt/c/behavior-2026/src/sim/fasteval/verify_chunk_equivalence.py \
    --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id turning_on_radio \
    --policy.config pi05_b1k --policy.dir "$CKPT" --control_mode receding_horizon --action_horizon 16 "$@" 2>&1 | tee "$LOG"
