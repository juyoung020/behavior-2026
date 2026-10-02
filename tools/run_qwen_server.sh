#!/usr/bin/env bash
# Qwen3.5-9B 계획기 서버 (llama.cpp llama-server, OpenAI 호환 API, 이미지 입력 가능)
#   주소   : http://127.0.0.1:8081/v1  (같은 PC. 밖에서 붙으려면 QWEN_HOST=0.0.0.0)
#            POST /v1/chat/completions (image_url 에 data:image/png;base64,... 가능), GET /health
#   모델   : ~/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf (5.68 GB, unsloth, sha256 03b74727…)
#            ~/models/Qwen3.5-9B-GGUF/mmproj-F16.gguf       (0.92 GB, 이미지 인코더)
#   GPU    : 전부 GPU(-ngl 99, 이미지 인코더도 GPU). 필요 메모리 = 실행기록_스펙_있는그대로.md 3절 실측값
#   생각 모드 끔(enable_thinking=false), 샘플링은 Qwen3.5-9B 모델 카드의 non-thinking 권장값.
#   바이너리: ~/llama.cpp/build/bin/llama-server (CUDA 12.8, sm_120 빌드: tools/setup/setup_llamacpp_cuda.sh; 모델 받기: tools/setup/fetch_qwen35_gguf.sh)
# 실행: bash tools/run_qwen_server.sh [CTX] [PORT]
set -euo pipefail
CTX=${1:-8192}
PORT=${2:-8081}
MODEL_DIR=$HOME/models/Qwen3.5-9B-GGUF
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$REPO/logs"
LOG=$REPO/logs/qwen_server_$(date +%Y%m%d_%H%M%S).log
echo "Qwen3.5-9B 서버: port=$PORT ctx=$CTX (로그 $LOG)"
exec "$HOME/llama.cpp/build/bin/llama-server" \
    -m "$MODEL_DIR/Qwen3.5-9B-Q4_K_M.gguf" \
    --mmproj "$MODEL_DIR/mmproj-F16.gguf" \
    --alias qwen3.5-9b \
    -ngl 99 -c "$CTX" -np 1 \
    --jinja --chat-template-kwargs '{"enable_thinking": false}' \
    --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5 --seed 0 \
    --host "${QWEN_HOST:-127.0.0.1}" --port "$PORT" > "$LOG" 2>&1
