#!/usr/bin/env bash
# GPU 잠금을 잡고(기다림) 렌더 기준 자료를 뜬 뒤 바로 푼다. 인자는 run_render_capture.sh 로 넘긴다.
# (리눅스판; 옛 Windows 판 archive/src/sim/engine/tests/render/capture/capture_with_lock.ps1 과 같은 선택지·동작.
#  잠금은 공용 대기열 tools/gpu_lock.sh(owner engine-render)로 잡는다 — 옛 판의 mkdir 되풀이 대신, 최대 3 시간 기다림)
#   bash src/sim/engine/tests/render/capture/capture_with_lock.sh --actions .../actions.npz --tag radio_rgbd
#       [--wrapper RGBD|Default] [--max-steps 500] [--steps 0,100] [--minutes 20] [--max-other-mib 3500]
# 검은 화면 (A)(다른 프로세스가 VRAM 약 5 GiB 넘게 잡으면 RTX 색이 빔)를 피하려고, 잠금을 잡은 뒤에도 GPU 전체 사용량이
# --max-other-mib(기본 3500) 아래일 때만 뜬다(30 초 간격 20 번까지 기다림).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../../.." && pwd)
ACTIONS=''; TAG=capture; WRAPPER=RGBD; MAX_STEPS=500; STEPS=''; MINUTES=20; MAX_OTHER=3500
while [ $# -gt 0 ]; do
  case $1 in
    --actions) ACTIONS=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --wrapper) WRAPPER=$2; shift 2 ;;
    --max-steps) MAX_STEPS=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --minutes) MINUTES=$2; shift 2 ;;
    --max-other-mib) MAX_OTHER=$2; shift 2 ;;
    -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "모르는 인자: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ACTIONS" ] || { echo "--actions 가 필요하다" >&2; exit 2; }
# shellcheck source=../../../../../../tools/gpu_lock.sh
source "$REPO/tools/gpu_lock.sh"
trap 'gpu_lock_cancel; gpu_lock_exit engine-render > /dev/null 2>&1 || true' EXIT
gpu_lock_enter engine-render "render 기준 자료 ($TAG, 공식 평가기 RTX $WRAPPER)" "$MINUTES" 9 180 || { echo "[gpu_lock] 3 시간 기다려도 못 잡음"; exit 2; }
OK=0
for _ in $(seq 20); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "[capture_with_lock] GPU 사용량 $USED MiB (기준 < $MAX_OTHER)"
  [ "$USED" -lt "$MAX_OTHER" ] && { OK=1; break; }
  sleep 30
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
[ $OK = 1 ] || { echo "[capture_with_lock] 다른 프로세스 VRAM 이 안 줄어 뜨지 않음"; exit 3; }
A=(--actions "$ACTIONS" --tag "$TAG" --wrapper "$WRAPPER" --max-steps "$MAX_STEPS")
[ -n "$STEPS" ] && A+=(--steps "$STEPS")
set +e
bash "$HERE/run_render_capture.sh" "${A[@]}"
CODE=$?
set -e
gpu_lock_exit engine-render > /dev/null || true
echo "[capture_with_lock] 잠금 풀림"
exit $CODE
