#!/usr/bin/env bash
# Linux 대회 환경(WSL2 Ubuntu 22.04 + pip isaacsim 5.1 + BEHAVIOR v3.9.3)에서 공식 평가기를 기록한 행동열로 재생하며
# 물리 층 기록(OVD·볼록 메시·곁기록·거르개 표 + trace.npz)을 뜬다. 윈도 run_capture.ps1 의 리눅스판.
#   bash /mnt/c/behavior-2026/src/engine/capture/run_capture_linux.sh <actions.npz 의 WSL 경로> <태그> [최대 스텝=500] [추가 Kit 설정...]
# 결과: ~/engine-data/linux_official/<태그>/ (에셋 파생물 -> 저장소 밖)
# 켜기 전에 윈도 쪽 Isaac Sim 이 없는지 확인할 것 (GPU 한 장 공유).
set -euo pipefail
ACTIONS=${1:?actions.npz}
TAG=${2:?tag}
MAXSTEPS=${3:-500}
shift 3 || true
EXTRA=("$@")
BASE=~/behavior-linux
OUT=~/engine-data/linux_official/$TAG
mkdir -p "$OUT"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_DATA_PATH=/mnt/c/behavior-2026/BEHAVIOR-1K/datasets
export OMNIGIBSON_APPDATA_PATH=$BASE/appdata
export PYTHONUTF8=1

PORT=8011
/home/juyoung/openpi/.venv/bin/python /mnt/c/behavior-2026/tools/replay_policy_server.py \
  --actions "$ACTIONS" --port $PORT --log "$OUT/server_log.npz" --once > "$OUT/replay_server.log" 2>&1 &
SRV=$!
for i in $(seq 60); do curl -sf http://127.0.0.1:$PORT/healthz >/dev/null && break; sleep 1; done

source $BASE/.venv/bin/activate
cd $BASE/BEHAVIOR-1K/OmniGibson
set +e
python /mnt/c/behavior-2026/src/engine/capture/physx_capture.py --dump-dir "$OUT" "${EXTRA[@]}" -- --trace -- \
  --task-name turning_on_radio --robot-config /mnt/c/behavior-2026/src/configs/r1pro_openpi.yaml \
  --env-wrapper omnigibson.eval.wrappers.DefaultWrapper --mode public_test \
  --host 127.0.0.1 --port $PORT --instance-indices 0 --num-envs 1 --max-steps "$MAXSTEPS" \
  --output-dir "$OUT" --headless 2>&1 | tee "$OUT/eval.log"
CODE=${PIPESTATUS[0]}
set -e
kill $SRV 2>/dev/null || true
python /mnt/c/behavior-2026/src/engine/capture/export_sidecar.py "$OUT" || true
echo "=== 기록 끝 (exit $CODE): $OUT ==="
