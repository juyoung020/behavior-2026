#!/usr/bin/env bash
# 시뮬레이터 없이 Comet 계획기 반복문(+ 선택: π0.5)을 시험한다. 앞서 녹화한 평가 영상 프레임을 2026 평가기 관측 모양으로 넣는다.
#   계획기 LLM 은 tools/comet_reasoner_env.sh (COMET_LLM=kau 기본).
#   MODE=nopolicy : π0.5 대신 0 행동(GPU 안 씀) — 계획기 응답 시간만 잰다
#   MODE=policy   : π0.5 pt50 을 GPU 에 올려 같이 — JAX GPU 메모리(현재·최대)도 찍는다
#   FGL=1(기본) 계획기 켬 | 0 계획기 끔(과제 문장 고정, π0.5 만 잴 때)
# 실행: wsl -d Ubuntu-22.04 -u juyoung -e bash /mnt/c/behavior-2026/tools/run_comet_offline_test.sh [MODE] [STEPS] [TAG] [FGL]
set -euo pipefail
MODE=${1:-nopolicy}
STEPS=${2:-160}
TAG=${3:-}
FGL=${4:-1}
STAMP=$(date +%Y%m%d_%H%M%S)_offline_${MODE}${TAG:+_$TAG}
. /mnt/c/behavior-2026/tools/comet_reasoner_env.sh
export COMET_TRACE_LOG=/mnt/c/behavior-2026/logs/comet_trace_$STAMP.jsonl
export COMET_TRACE_IMG_DIR=/mnt/c/behavior-2026/logs/comet_trace_${STAMP}_img
VIDEO=/mnt/c/behavior-2026/outputs/eval_turning_on_radio_20260929_1929/videos/turning_on_radio_301_0.mp4  # 머리 화면 301장 모두 정상
cd ~/openpi-comet
ARGS=(--steps "$STEPS" --video "$VIDEO" --fine_grained_level "$FGL")
if [ "$MODE" = nopolicy ]; then
  export JAX_PLATFORMS=cpu
  ARGS+=(--no-policy)
else
  export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
  ARGS+=(--policy_dir "$HOME/checkpoints/openpi_comet/pi05-b1kpt50-cs32")
fi
echo "기록: $COMET_TRACE_LOG"
.venv/bin/python -u scripts/offline_loop_test.py "${ARGS[@]}" 2>&1 | grep -v -E "^INFO:(httpx|policy)" | tee "/mnt/c/behavior-2026/logs/comet_offline_$STAMP.log"
rm -f client.log
