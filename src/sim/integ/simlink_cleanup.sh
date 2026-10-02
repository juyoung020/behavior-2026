#!/usr/bin/env bash
# 통합 한 판 비정상 종료 때 simlink 쪽 정리: simlink_stack.sh 가 적어 둔 pid 만 끈다(다른 에이전트 프로세스는 안 건드림).
# (리눅스판; 옛 WSL 판 archive/src/sim/integ/wsl_cleanup.sh 와 같은 동작)
#   bash src/sim/integ/simlink_cleanup.sh <출력 폴더>
set -euo pipefail
OUT=${1:?출력 폴더}
[ -f "$OUT/stack_pids" ] || { echo "[cleanup] $OUT/stack_pids 없음"; exit 0; }
SL=''; STACK=''
# shellcheck disable=SC1091
. "$OUT/stack_pids"
[ -n "$SL" ] && kill -TERM "$SL" 2>/dev/null || true
sleep 5
for p in $SL $STACK; do kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null || true; done
echo "[cleanup] 끝"
