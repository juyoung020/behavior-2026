#!/usr/bin/env bash
# Comet 계획기(openpi-comet src/openpi/shared/client.py)가 부를 LLM 을 고른다. run_comet_server.sh 등이 source 한다.
#   COMET_LLM=kau   (기본) KAU API — https://agent.kau.ac.kr/v1 , 모델 qwen3.5-9b (vLLM, 문맥 16384). GPU 안 씀.
#                   키는 WSL ~/.config/behavior-2026/kau.env (chmod 600, 저장소 밖)의 KAU_BASE_URL·KAU_API_KEY.
#   COMET_LLM=local 로컬 llama.cpp — tools/run_qwen_server.sh (http://127.0.0.1:8081/v1, GPU ~6.7 GB)
# 키는 화면·로그·기록에 찍지 않는다(주소만 찍음).
# 두 서버 모두 Qwen3.5-9B 의 생각 모드를 끄고(원래 Comet 이 쓰던 Qwen3-VL-30B-A3B-Instruct 는 생각 모드 없음)
# 샘플링은 Qwen3.5-9B 모델 카드의 non-thinking 권장값으로 요청마다 보낸다(COMET_REASONER_EXTRA_BODY).
COMET_LLM=${COMET_LLM:-kau}
case "$COMET_LLM" in
  kau)
    set -a; . "$HOME/.config/behavior-2026/kau.env"; set +a
    export COMET_REASONER_BASE_URL="$KAU_BASE_URL"
    export COMET_REASONER_API_KEY="$KAU_API_KEY"
    unset KAU_API_KEY
    ;;
  local)
    export COMET_REASONER_BASE_URL=http://127.0.0.1:8081/v1
    export COMET_REASONER_API_KEY=local
    ;;
  *) echo "COMET_LLM 은 kau 또는 local" >&2; return 1 ;;
esac
export COMET_REASONER_MODEL=qwen3.5-9b
export COMET_REASONER_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}'
echo "계획기 LLM: $COMET_LLM $COMET_REASONER_BASE_URL model=$COMET_REASONER_MODEL"
