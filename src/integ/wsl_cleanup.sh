#!/bin/bash
# 통합 한 판 비정상 종료 때 WSL 쪽 정리: wsl_stack.sh 가 적어 둔 pid 만 끈다(다른 에이전트 프로세스는 안 건드림).
#   bash /mnt/c/behavior-2026/src/integ/wsl_cleanup.sh <출력 폴더(/mnt/c/...)>
OUT=${1:?출력 폴더}
[ -f "$OUT/wsl_pids" ] || { echo "[cleanup] $OUT/wsl_pids 없음"; exit 0; }
. "$OUT/wsl_pids"
[ -n "$SL" ] && kill -TERM $SL 2>/dev/null
sleep 5
for p in $SL $STACK; do kill -0 $p 2>/dev/null && kill -KILL $p 2>/dev/null; done
echo "[cleanup] 끝"
