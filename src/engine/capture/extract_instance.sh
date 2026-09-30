#!/usr/bin/env bash
# 판 하나의 "장면 추출물" 뜨기 (인스턴스 불러오기 B안, docs/엔진_자체구현.md 15절): 공식 평가기로 불러오기 구간만 1 스텝, 0 행동.
# 그 뒤 이 판의 정책 실행은 몇 번이든 엔진(포팅 평가기 --backend engine)이 한다.
#   bash /mnt/c/behavior-2026/src/engine/capture/extract_instance.sh <모드 public_test|train> <인스턴스 번호> [과제=turning_on_radio]
# 결과: ~/engine-data/scenes/<과제>/<모드>_<번호>/ (OVD·곁기록·scope.json + export_s1/s3/ag 결과). GPU 대기열로 잡는다.
set -euo pipefail
ulimit -c 0
MODE=${1:?mode}
IDX=${2:?instance index}
TASK=${3:-turning_on_radio}
OUTROOT=~/engine-data/scenes/$TASK
TAG=${MODE}_${IDX}
mkdir -p "$OUTROOT"
ZA=~/engine-data/scenes/zero_actions_1.npz
[ -f "$ZA" ] || python3 -c "import numpy as np; np.savez('$ZA', actions=np.zeros((2,1,23), np.float32))"
source /mnt/c/behavior-2026/src/engine/scripts/gpu_lock.sh
gpu_lock_acquire "engine-lead" "장면 추출 $TASK $TAG (약 2분)" 10 6 || exit 3
set +e
TASK_NAME=$TASK EVAL_MODE=$MODE INSTANCE_IDX=$IDX bash /mnt/c/behavior-2026/src/engine/capture/run_capture_linux.sh "$ZA" "scene_${TASK}_${TAG}" 1 --record-toggle
CODE=$?
set -e
gpu_lock_release "engine-lead"
SRC=~/engine-data/linux_official/scene_${TASK}_${TAG}
DST=$OUTROOT/$TAG
rm -rf "$DST.tmp"
mv "$SRC" "$DST.tmp"
[ -d "$DST" ] && rm -rf "$DST"
mv "$DST.tmp" "$DST"
cd "$DST"
python3 /mnt/c/behavior-2026/src/engine/capture/export_s1.py . | tail -1
python3 /mnt/c/behavior-2026/src/engine/capture/export_s3.py . --bddl /mnt/c/behavior-2026/BEHAVIOR-1K/bddl3/bddl/activity_definitions/$TASK/problem0.bddl 2>&1 | tail -1 || true
python3 /mnt/c/behavior-2026/src/engine/capture/export_ag.py . | tail -1 || true
# 앞 몫(side offset) = post 수 - 가장 큰 OVD 의 simulate 수 (ovd_dump 는 simulate 를 프레임 2 개로 센다)
BIG=$(ls -S *_rec.ovd 2>/dev/null | head -1)
if [ -n "$BIG" ]; then
  FR=$(~/engine-build/replay-checked/ovd_dump "$BIG" 2>/dev/null | grep -o '문맥(장면)별: [^ ]*=[0-9]*' | head -1 | sed 's/.*=//')
  POST=$(python3 -c "import json;print(json.load(open('meta.json'))['post_step_count'])")
  [ -n "$FR" ] && echo $(( POST - FR / 2 )) > side_offset.txt
  # 줄이기: 엔진은 가장 큰 OVD 와 내보낸 파일만 쓴다 (앞선 PhysX 인스턴스 OVD·원본 npz·0 영상은 지움)
  for f in *_rec.ovd; do [ "$f" != "$BIG" ] && rm -f "$f"; done
  rm -f convex.npz trace_images.npz toggle.pkl
fi
[ $CODE -eq 0 ] && [ -f scope.json ] && [ -n "$BIG" ] && [ -f side_offset.txt ] && echo "$(date +%F_%T) $(du -sm . | cut -f1)MB" > extract_ok
echo "=== 장면 추출 끝 (capture exit $CODE): $DST ==="
