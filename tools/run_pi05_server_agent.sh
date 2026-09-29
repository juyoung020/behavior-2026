#!/usr/bin/env bash
# π0.5 서버를 에이전트 지시 훅(tools/serve_b1k_agent.py)과 함께 띄운다. 평가기는 중계기(bagent relay, 포트 8000)에 붙고,
# 중계기가 이 서버(포트 8100)로 관측 + "__agent_prompt__" 를 넘긴다. 설계: docs/에이전트_설계.md 1.10, 7절.
# 실행: wsl -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/tools/run_pi05_server_agent.sh [TASK] [PORT]
# run_pi05_server.sh 와 다른 점: serve_b1k.py 대신 serve_b1k_agent.py(같은 인자), 기본 포트 8100.
# GPU 를 쓰므로 먼저 nvidia-smi 로 남은 메모리를 확인한다.
set -euo pipefail
TASK=${1:-turning_on_radio}
PORT=${2:-8100}
CKPT=~/checkpoints/pi05_turning_on_radio/pi05_turn_on_the_radio
LOG=/mnt/c/behavior-2026/logs/pi05_server_agent_$(date +%Y%m%d_%H%M).log
export PATH=$HOME/.local/bin:$PATH
cd ~/openpi
source .venv/bin/activate
echo "π0.5 서버(에이전트 훅): task=$TASK ckpt=$CKPT port=$PORT (로그 $LOG)"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
uv run /mnt/c/behavior-2026/tools/serve_b1k_agent.py \
    --robot b1k/R1Pro \
    --task "b1k/$TASK" \
    --repo-id "$TASK" \
    --policy.config pi05_b1k \
    --policy.dir "$CKPT" \
    --control_mode receding_horizon \
    --action_horizon 16 \
    --port "$PORT" 2>&1 | tee "$LOG"
