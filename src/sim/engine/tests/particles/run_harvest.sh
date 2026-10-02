#!/usr/bin/env bash
# 새 행위자 틀 수확 (harvest_spawn.py): 공식 평가기를 리드의 physx_capture(OVD 기록)로 돌리며 판의 자를 것·다질 것을 모두 전이시킨다.
#   bash /mnt/c/behavior-2026/src/sim/engine/tests/particles/run_harvest.sh <과제> [모드=public_test] [인스턴스=0] [최대 스텝=1500]
# 결과: ~/engine-data/linux_official/harvest_<과제>_<모드>_<인스턴스>/ (OVD·곁기록·harvest_map.json). 리드의 g1_sc 틀 쓰기가 이 기록을 읽는다.
# GPU 대기열로 잡는다. 풀린 장면 USD(og.tempdir)는 physx_capture 가 자기 경로만 지운다.
set -uo pipefail
ulimit -c 0
TASK=${1:?task}
MODE=${2:-public_test}
IDX=${3:-0}
MAXSTEPS=${4:-1500}
FREE=$(df -BG /mnt/c | awk 'NR==2{gsub("G","",$4); print $4}')
if [ "$FREE" -lt 8 ]; then echo "C: 여유 ${FREE} GB — 8 GB 밑이라 멈춤"; exit 4; fi
ZA=~/engine-data/particles/zero_actions_$((MAXSTEPS + 2)).npz  # 영행동 (스텝 수만큼, 리드 것과 따로)
mkdir -p ~/engine-data/particles
[ -f "$ZA" ] || python3 -c "import numpy as np; np.savez('$ZA', actions=np.zeros(($MAXSTEPS + 2,1,23), np.float32))"
source /mnt/c/behavior-2026/src/sim/engine/scripts/gpu_lock.sh
gpu_lock_acquire engine-particles "틀 수확 $TASK $MODE $IDX" 30 6 || exit 3
set +e
# 새 이름(.tmp)에 뜬 뒤 끝나면 옮긴다 — 다른 작업자가 읽고 있을 수 있는 기록을 제자리에서 덮어쓰지 않음 (09-30 조정자 규칙)
TAG=${HARVEST_TAG:-harvest_${TASK}_${MODE}_${IDX}}  # HARVEST_TAG: 기존 기록(리드 틀이 붙은 폴더)을 건드리지 않고 따로 뜰 때
TASK_NAME=$TASK EVAL_MODE=$MODE INSTANCE_IDX=$IDX bash /mnt/c/behavior-2026/src/sim/engine/capture/run_capture_linux.sh "$ZA" "${TAG}.tmp" "$MAXSTEPS" \
  --script /mnt/c/behavior-2026/src/sim/engine/tests/particles/harvest_spawn.py
CODE=$?
set -e
gpu_lock_release engine-particles
OUT=~/engine-data/linux_official/$TAG
if [ "$CODE" = 0 ]; then
  [ -d "$OUT" ] && mv "$OUT" "$OUT.old.$$"
  mv "$OUT.tmp" "$OUT" && rm -rf "$OUT.old.$$"
else
  echo "기록 실패 — $OUT.tmp 를 남김 (옛 기록은 그대로)"
fi
grep -E "^\[harvest\]" "$OUT/eval.log" | tail -30
echo "=== 끝 (exit $CODE): $OUT  $(du -sh $OUT | cut -f1)"
