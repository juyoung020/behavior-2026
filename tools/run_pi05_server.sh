#!/usr/bin/env bash
# π0.5 기본 제공 체크포인트(turning_on_radio)로 정책 서버를 띄운다 (공식 baselines 명령 그대로).
# 실행: wsl -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/tools/run_pi05_server.sh [TASK]
# 다른 점 하나: GPU 한 장(16GB)을 Windows 쪽 시뮬레이터와 나눠 써서 XLA 메모리 비율을 0.85 -> 0.5 로 낮췄다.
set -euo pipefail
TASK=${1:-turning_on_radio}
CKPT=~/checkpoints/pi05_turning_on_radio/pi05_turn_on_the_radio
LOG=/mnt/c/behavior-2026/logs/pi05_server_$(date +%Y%m%d_%H%M).log
export PATH=$HOME/.local/bin:$PATH
cd ~/openpi
source .venv/bin/activate
echo "π0.5 서버: task=$TASK ckpt=$CKPT port=8000 (로그 $LOG)"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
uv run scripts/b1k/serve_b1k.py \
    --robot b1k/R1Pro \
    --task "b1k/$TASK" \
    --repo-id "$TASK" \
    --policy.config pi05_b1k \
    --policy.dir "$CKPT" \
    --control_mode receding_horizon \
    --action_horizon 16 \
    --port 8000 2>&1 | tee "$LOG"
