#!/usr/bin/env bash
# 공식 평가기로 Covered 과제 한 판 기록 (capture_covered.py). GPU 대기열로 잡는다.
#   bash /mnt/c/behavior-2026/src/engine/tests/particles/run_capture_covered.sh <과제> [스텝=60] [인스턴스=0] [제거기 이름]
# 결과: ~/engine-data/particles/covered/<과제>_<인스턴스>/ (에셋 파생물 -> 저장소 밖)
set -uo pipefail
ulimit -c 0
TASK=${1:?task}
STEPS=${2:-60}
IDX=${3:-0}
REM=${4:-}
OUT=~/engine-data/particles/covered/${TASK}_${IDX}
mkdir -p "$OUT"
FREE=$(df -BG /mnt/c | awk 'NR==2{gsub("G","",$4); print $4}')
if [ "$FREE" -lt 5 ]; then echo "C: 여유 ${FREE} GB — 5 GB 밑이라 멈춤"; exit 4; fi
BASE=~/behavior-linux
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_DATA_PATH=/mnt/c/behavior-2026/BEHAVIOR-1K/datasets
export OMNIGIBSON_APPDATA_PATH=$BASE/appdata
export PYTHONUTF8=1
source /mnt/c/behavior-2026/src/engine/scripts/gpu_lock.sh
gpu_lock_acquire engine-particles "Covered 기록 $TASK $IDX (약 5분)" 15 6 || exit 3
source $BASE/.venv/bin/activate
cd $BASE/BEHAVIOR-1K/OmniGibson
set +e
python /mnt/c/behavior-2026/src/engine/tests/particles/capture_covered.py --task "$TASK" --instance "$IDX" --steps "$STEPS" \
  --out "$OUT" ${REM:+--remover "$REM"} > "$OUT/capture.log" 2>&1
CODE=$?
set -e
gpu_lock_release engine-particles
grep -E "^\[particles\]|Error|error" "$OUT/capture.log" | tail -20
echo "=== 끝 (exit $CODE): $OUT"
