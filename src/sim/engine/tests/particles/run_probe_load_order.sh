#!/usr/bin/env bash
# base 창 탐침 (probe_load_order.py): 텐서 쓰기·psi 가 PhysX 에 닿는 때. GPU 대기열로 잡는다.
#   bash /mnt/c/behavior-2026/src/sim/engine/tests/particles/run_probe_load_order.sh [과제=chopping_wood] [먼저 돌 스텝=60] [인스턴스=0]
# 결과: ~/engine-data/particles/probe_load/<과제>_<인스턴스>/probe.json
set -uo pipefail
ulimit -c 0
TASK=${1:-chopping_wood}
STEPS=${2:-60}
IDX=${3:-0}
OUT=~/engine-data/particles/probe_load/${TASK}_${IDX}
mkdir -p "$OUT"
FREE=$(df -BG /mnt/c | awk 'NR==2{gsub("G","",$4); print $4}')
if [ "$FREE" -lt 5 ]; then echo "C: 여유 ${FREE} GB — 5 GB 밑이라 멈춤"; exit 4; fi
BASE=~/behavior-linux
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_DATA_PATH=/mnt/c/behavior-2026/BEHAVIOR-1K/datasets
export OMNIGIBSON_APPDATA_PATH=$BASE/appdata
export PYTHONUTF8=1
source /mnt/c/behavior-2026/src/sim/engine/scripts/gpu_lock.sh
gpu_lock_acquire engine-particles "base 창 탐침 $TASK (약 4분)" 15 6 || exit 3
source $BASE/.venv/bin/activate
cd $BASE/BEHAVIOR-1K/OmniGibson
set +e
python /mnt/c/behavior-2026/src/sim/engine/tests/particles/probe_load_order.py --task "$TASK" --instance "$IDX" --settle "$STEPS" \
  --out "$OUT" > "$OUT/capture.log" 2>&1
CODE=$?
set -e
gpu_lock_release engine-particles
grep -E "^\[probe\]|Error|error" "$OUT/capture.log" | tail -20
echo "=== 끝 (exit $CODE): $OUT"
