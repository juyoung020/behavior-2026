#!/usr/bin/env bash
# B9 탐침: 공식 평가기(chopping_wood, 영행동)를 리드의 physx_capture 로 OVD·곁기록과 함께 돌리며 probe_psi_ovd.py 조작을 한다.
#   bash /mnt/c/behavior-2026/src/engine/tests/particles/run_probe_psi_ovd.sh [과제=chopping_wood]
# 결과: ~/engine-data/linux_official/probe_psi_ovd_<과제>/ (조작 목록은 eval.log 의 [probe] 줄)
set -uo pipefail
ulimit -c 0
TASK=${1:-chopping_wood}
FREE=$(df -BG /mnt/c | awk 'NR==2{gsub("G","",$4); print $4}')
if [ "$FREE" -lt 8 ]; then echo "C: 여유 ${FREE} GB — 8 GB 밑이라 멈춤"; exit 4; fi
ZA=~/engine-data/particles/zero_actions_402.npz
[ -f "$ZA" ] || python3 -c "import numpy as np; np.savez('$ZA', actions=np.zeros((402,1,23), np.float32))"
source /mnt/c/behavior-2026/src/engine/scripts/gpu_lock.sh
gpu_lock_acquire engine-particles "B9 psi 탐침 $TASK (약 4분)" 15 6 || exit 3
set +e
TAG=probe_psi_ovd_${TASK}
rm -rf ~/engine-data/linux_official/$TAG
TASK_NAME=$TASK EVAL_MODE=public_test INSTANCE_IDX=0 bash /mnt/c/behavior-2026/src/engine/capture/run_capture_linux.sh "$ZA" "$TAG" 80 \
  --script /mnt/c/behavior-2026/src/engine/tests/particles/probe_psi_ovd.py > /dev/null 2>&1
CODE=$?
set -e
gpu_lock_release engine-particles
OUT=~/engine-data/linux_official/$TAG
grep -E "^\[probe\]|스크립트 실패" "$OUT/eval.log"
echo "=== 끝 (exit $CODE): $OUT  $(du -sh $OUT | cut -f1)"
