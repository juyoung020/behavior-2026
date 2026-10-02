#!/usr/bin/env bash
# render B GPU 시험 한 판: 공용 대기열(tools/gpu_lock.sh)에 줄 서서 잡고 -> 시험 스크립트 -> 바로 풂.
# (리눅스판; 옛 Windows 판 archive/src/sim/engine/tests/render/gpu_session.ps1 과 같은 선택지·동작)
#   bash src/sim/engine/tests/render/gpu_session.sh [--script <경로>] [--minutes 5] [--log logs/render_gpu_session.log]
# 기본 스크립트: src/sim/engine/tests/render/gpu_tests.sh (합성 층1=층2 + 뜬 장면 있으면 장면 층1=층2·처리량)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
SCRIPT=$HERE/gpu_tests.sh; MINUTES=5; LOG=$REPO/logs/render_gpu_session.log
while [ $# -gt 0 ]; do
  case $1 in
    --script) SCRIPT=$2; shift 2 ;;
    --minutes) MINUTES=$2; shift 2 ;;
    --log) LOG=$2; shift 2 ;;
    -h|--help) sed -n '2,5p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "모르는 인자: $1" >&2; exit 2 ;;
  esac
done
# shellcheck source=../../../../../tools/gpu_lock.sh
source "$REPO/tools/gpu_lock.sh"
trap 'gpu_lock_cancel' INT TERM
gpu_lock_enter render-B 'render B: 층1=층2 비트 시험·처리량 (수 분)' "$MINUTES" 3 240 || exit 2
mkdir -p "$(dirname "$LOG")"
set +e
bash "$SCRIPT" > "$LOG" 2>&1
CODE=$?
set -e
gpu_lock_exit render-B > /dev/null || true
echo "[gpu_session] 잠금 풀림" >> "$LOG"
exit $CODE
