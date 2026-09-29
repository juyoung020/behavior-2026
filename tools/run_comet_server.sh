#!/usr/bin/env bash
# 2위 Comet π0.5(pt50) 정책 서버 + Comet 계획기 반복문(계획 → 점검 → 다음 하위과제 문장) → Qwen3.5-9B.
#   주소 : ws://0.0.0.0:8000 (평가기는 127.0.0.1:8000 으로 접속), GET /healthz
#   코드 : WSL ~/openpi-comet (커밋 4bb2aa7 + 로컬 브랜치 behavior-2026-run 의 수정, docs/실행기록_스펙_있는그대로.md)
#   가중치: WSL ~/checkpoints/openpi_comet/pi05-b1kpt50-cs32 (설정 pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k)
#   계획기: tools/comet_reasoner_env.sh — COMET_LLM=kau(기본, KAU API, GPU 안 씀) | local(tools/run_qwen_server.sh)
#           FGL=0 이면 계획기 끔(과제 문장 고정 = 원래 Comet 기본).
#   검은 프레임: 검출만 한다(관측은 받은 그대로). 스텝별 카메라 밝기·검은 수를 기록 → 섞이면 그 실행은 무효.
#   기록 : logs/comet_trace_<시각>.jsonl (계획기 요청·응답·시간, π0.5 prompt 변화·스텝·검은 프레임),
#          logs/comet_trace_<시각>_frames.csv (스텝별 카메라 밝기), logs/comet_trace_<시각>_img/ (π0.5 호출 때 머리 영상)
#   GPU  : XLA_PYTHON_CLIENT_PREALLOCATE=false 로 필요한 만큼만(원래 openpi 는 75% 를 미리 잡음).
# 실행: wsl -d Ubuntu-22.04 -u juyoung -e bash /mnt/c/behavior-2026/tools/run_comet_server.sh [FGL] [TAG]
#       (키를 읽으므로 bash -c 로 넘기지 말고 이 파일을 실행)
set -euo pipefail
FGL=${1:-1}
TAG=${2:-}
STAMP=$(date +%Y%m%d_%H%M%S)${TAG:+_$TAG}
export PATH=$HOME/.local/bin:$PATH
. /mnt/c/behavior-2026/tools/comet_reasoner_env.sh
export COMET_TRACE_LOG=/mnt/c/behavior-2026/logs/comet_trace_$STAMP.jsonl
export COMET_TRACE_IMG_DIR=/mnt/c/behavior-2026/logs/comet_trace_${STAMP}_img
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
LOG=/mnt/c/behavior-2026/logs/comet_server_$STAMP.log
cd ~/openpi-comet
echo "Comet 서버: FGL=$FGL trace=$COMET_TRACE_LOG (로그 $LOG)"
# uv run 이 아니라 .venv 의 python 을 직접 쓴다: uv run 은 실행 전에 잠금파일(uv.lock)대로 환경을 다시 맞추는데,
# 계획기 의존성(openai 1.x·json-repair·tenacity)은 잠금파일에 없어서 따로 넣었다.
exec .venv/bin/python scripts/serve_b1k.py \
    --task_name=turning_on_radio \
    --control_mode=receeding_horizon \
    --max_len=32 \
    --fine_grained_level="$FGL" \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k \
    --policy.dir="$HOME/checkpoints/openpi_comet/pi05-b1kpt50-cs32" > "$LOG" 2>&1
