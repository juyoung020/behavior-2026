#!/usr/bin/env bash
# π0.5 서버를 '행동 묶음 재생' 응답이 되는 판(src/fasteval/pi05_chunk_server.py)으로 띄운다. 인자·체크포인트는 run_pi05_server.sh 와 같다.
# 평가기는 run_eval_radio.ps1 -ChunkSize 16 (= 공식 --replay-action-chunk-size 16). 묶음 인자 없이 오면 공식 서버와 똑같이 동작한다.
# 실행: wsl -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/tools/run_pi05_chunk_server.sh [TASK]
set -euo pipefail
TASK=${1:-turning_on_radio}
CKPT=~/checkpoints/pi05_turning_on_radio/pi05_turn_on_the_radio
LOG=/mnt/c/behavior-2026/logs/pi05_chunk_server_$(date +%Y%m%d_%H%M).log
cd ~/openpi
echo "π0.5 묶음 서버: task=$TASK ckpt=$CKPT port=8000 (로그 $LOG)"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
.venv/bin/python /mnt/c/behavior-2026/src/fasteval/pi05_chunk_server.py \
    --robot b1k/R1Pro \
    --task "b1k/$TASK" \
    --repo-id "$TASK" \
    --policy.config pi05_b1k \
    --policy.dir "$CKPT" \
    --control_mode receding_horizon \
    --action_horizon 16 \
    --port 8000 2>&1 | tee "$LOG"
