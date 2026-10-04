#!/bin/bash
# LIMO + OMX-F 로 장면 그래프 런타임(libsgrt, SGRT_ROBOT=limo_omx)을 시뮬에서 짧게 굴려 지도·자세를 정답과 맞춰 본다(에이전트·move_robot 없음).
#   run_limo_map.sh [task=turning_on_radio] [steps=900] [tag]
# 결과: outputs/limo_map_<ts>_<task>[_tag]/ (memory/ 지도·장면 그래프, poses.csv, summary.json, overlay.png, sim.log)
# 사전 조건: robot-agent src/robot/og 의 limo_omx 로봇이 OmniGibson 에 설치돼 있을 것(import_to_omnigibson.sh), VRAM 여유 ≥ 6 GB.
set -u
TASK=${1:-turning_on_radio}; STEPS=${2:-900}; TAG=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
OGDIR=${LIMO_OG_DIR:-$HOME/robot-agent/src/robot/og}
TS=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-$REPO/outputs/limo_map_${TS}_${TASK}${TAG:+_$TAG}}
GT=${GT_DIR:-$REPO/src/sim/explore/gt}
[ -d "$GT" ] || GT=$HOME/behavior-2026/src/sim/explore/gt   # gt/ 는 git 밖(gt_trav.py 로 만듦)
mkdir -p "$OUT"
while true; do
  fv=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, '{print $1-$2}')
  fr=$(free -g | awk '/Mem:/{print $7}')
  [ "$fv" -ge 6000 ] && [ "$fr" -ge 12 ] && break
  echo "[limo] waiting: VRAM free $fv MiB, RAM avail $fr GB"; sleep 30
done
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavior
export OMNI_KIT_ACCEPT_EULA=YES
export SGRT_ROBOT=${SGRT_ROBOT:-limo_omx}
export SGRT_POSE=${SGRT_POSE:-slam}
export SGRT_LIB=${SGRT_LIB:-$HOME/sgrt_build/libsgrt.so}
# YOLO26s-seg 는 보관됨(2026-10-05, ~/ovdet_models/archive) — FastSAM-s + SigLIP 2(objprob)로 옮길 때까지 보관 엔진
export SGRT_ENGINE=${SGRT_ENGINE:-$HOME/ovdet_models/archive/x86_sm120/yolo26s-seg.plan}
export LIMO_SHIM=${LIMO_SHIM:-$OGDIR/eval_with_limo.py}
if ! strings "$SGRT_LIB" | grep -q sgrt_set_robot; then echo "[limo] $SGRT_LIB 에 로봇 고르기가 없다(옛 빌드) — 다시 빌드할 것"; exit 1; fi
cd "$OUT"
python "$HERE/run_limo_map.py" --out "$OUT" --steps "$STEPS" --gt-dir "$GT" -- --task-name "$TASK" --mode public_test \
  --instance-indices 0 --num-envs 1 --max-steps $((STEPS + 60)) --headless \
  --robot-config "$OGDIR/limo_omx_eval.yaml" --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper > "$OUT/sim.log" 2>&1
echo "[limo] done rc $? -> $OUT"
[ -f "$OUT/summary.json" ] && python -c "import json,sys; s=json.load(open('$OUT/summary.json')); print(json.dumps({k: s[k] for k in ('steps','gt_path_m','gt_rot_deg','pose_diag','map')}, indent=1))"
