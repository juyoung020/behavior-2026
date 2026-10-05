#!/usr/bin/env bash
# 공식 평가기를 기록한 행동열로 재생하면서 물리 층 기록(OVD·볼록 메시·곁기록·거르개 표)을 뜬다. RTX 렌더 켬(영상 저장).
# (리눅스판; 옛 Windows 판 archive/src/sim/engine/capture/run_capture.ps1 과 같은 선택지·결과. 렌더 없는 옛 WSL 판은 run_capture_linux.sh)
#   bash src/sim/engine/capture/run_capture.sh --actions outputs/eval_turning_on_radio_20260929_195500_nf_a/actions.npz --tag radio_nfa
#       [--task turning_on_radio] [--instance 0] [--max-steps 500] [--dry-run]
# 결과: src/sim/engine/dumps/<tag>/  (에셋 파생물 -> git 제외)  + trace.npz 는 같은 폴더(평가기 --output-dir)
# 먼저 nvidia-smi 로 다른 시뮬레이터가 없는지 확인할 것 (GPU 한 장을 여러 작업이 씀). 재생 서버: REPLAY_PY(기본 conda behavior 환경 파이썬), 포트 8010.
set -euo pipefail
ENGINE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(cd "$ENGINE/../../.." && pwd)
ACTIONS=''; TASK=turning_on_radio; INSTANCE=0; MAX_STEPS=500; TAG=capture; DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --actions) ACTIONS=$2; shift 2 ;;
    --task) TASK=$2; shift 2 ;;
    --instance) INSTANCE=$2; shift 2 ;;
    --max-steps) MAX_STEPS=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "모르는 인자: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ACTIONS" ] || { echo "--actions 가 필요하다" >&2; exit 2; }
ACTIONS=$(realpath "$ACTIONS")
DUMP=$ENGINE/dumps/$TAG
SRV=("${REPLAY_PY:-$HOME/miniconda3/envs/behavior/bin/python}" "$REPO/tools/replay_policy_server.py" --actions "$ACTIONS" --port 8010
     --log "$DUMP/server_log.npz" --once)
EV=(python "$ENGINE/capture/physx_capture.py" --dump-dir "$DUMP" -- --trace --
    --task-name "$TASK" --robot-config "$REPO/src/sim/configs/r1pro_robot.yaml"
    --env-wrapper omnigibson.eval.wrappers.DefaultWrapper --mode public_test
    --host 127.0.0.1 --port 8010 --instance-indices "$INSTANCE" --num-envs 1 --max-steps "$MAX_STEPS"
    --output-dir "$DUMP" --write-video --headless)
if [ $DRY = 1 ]; then echo "(dry-run) 서버: ${SRV[*]}"; echo "(dry-run) 평가기: ${EV[*]}"; exit 0; fi
mkdir -p "$DUMP"
"${SRV[@]}" > "$DUMP/replay_server.out.log" 2> "$DUMP/replay_server.log" &
SRV_PID=$!
OK=0
for _ in $(seq 60); do curl -sf -m 2 http://127.0.0.1:8010/healthz > /dev/null 2>&1 && { OK=1; break; }; sleep 1; done
[ $OK = 1 ] || { echo "재생 서버가 안 떴다 ($DUMP/replay_server.log)"; kill $SRV_PID 2>/dev/null || true; exit 1; }
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 OMNI_KIT_ACCEPT_EULA=YES KMP_DUPLICATE_LIB_OK=TRUE
CONDA_BASE=${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u; conda activate behavior; set -u
cd "$REPO/BEHAVIOR-1K/OmniGibson"
set +e
"${EV[@]}" 2>&1 | tee "$DUMP/eval.log"
set -e
for _ in $(seq 30); do kill -0 $SRV_PID 2>/dev/null || break; sleep 1; done
kill $SRV_PID 2>/dev/null || true
python "$ENGINE/capture/export_sidecar.py" "$DUMP"
echo "=== 기록 끝: $DUMP ==="
