#!/usr/bin/env bash
# 렌더 모듈 기준 자료: 기록해 둔 행동열을 재생하며 공식 평가기(RGB-D 720/480 래퍼)에서 장면 기하·카메라·공식 영상을 뜬다.
# (리눅스판; 옛 Windows 판 archive/src/sim/engine/tests/render/capture/run_render_capture.ps1 과 같은 선택지·결과)
#   bash src/sim/engine/tests/render/capture/run_render_capture.sh --actions outputs/eval_turning_on_radio_20260929_195500_nf_a/actions.npz \
#       --tag radio_rgbd [--task turning_on_radio] [--instance 0] [--max-steps 500] [--wrapper RGBD|Default] [--steps 0,100] [--noise-renders 2] [--dry-run]
# 결과: src/sim/engine/dumps/render_<tag>/  (에셋 파생물 -> git 제외) + trace.npz(평가기 --output-dir 도 같은 폴더)
# GPU 잠금은 부르는 쪽이 잡는다(capture_with_lock.sh). 다른 프로세스가 VRAM 5 GiB 넘게 잡으면 RTX 색이 빈다(plan.md 3절 (A)).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE=$(cd "$HERE/../../.." && pwd)
REPO=$(cd "$ENGINE/../../.." && pwd)
ACTIONS=''; TASK=turning_on_radio; INSTANCE=0; MAX_STEPS=500; TAG=capture; WRAPPER=RGBD; STEPS=''; NOISE=2; DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --actions) ACTIONS=$2; shift 2 ;;
    --task) TASK=$2; shift 2 ;;
    --instance) INSTANCE=$2; shift 2 ;;
    --max-steps) MAX_STEPS=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --wrapper) WRAPPER=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --noise-renders) NOISE=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "모르는 인자: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ACTIONS" ] || { echo "--actions 가 필요하다" >&2; exit 2; }
case $WRAPPER in
  Default) WT=omnigibson.eval.wrappers.DefaultWrapper ;;
  RGBD) WT=omnigibson.eval.wrappers.RGBDFullResWrapper ;;
  *) echo "--wrapper 는 Default|RGBD" >&2; exit 2 ;;
esac
ACTIONS=$(realpath "$ACTIONS")
DUMP=$ENGINE/dumps/render_$TAG
OURS=(--dump-dir "$DUMP" --noise-renders "$NOISE")
[ -n "$STEPS" ] && OURS+=(--steps "$STEPS")
SRV=("${REPLAY_PY:-$HOME/miniconda3/envs/behavior/bin/python}" "$REPO/tools/replay_policy_server.py" --actions "$ACTIONS" --port 8010
     --log "$DUMP/server_log.npz" --once)
EV=(python "$HERE/render_capture.py" "${OURS[@]}" -- --trace --
    --task-name "$TASK" --robot-config "$REPO/src/sim/configs/r1pro_robot.yaml"
    --env-wrapper "$WT" --mode public_test
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
"${EV[@]}" > "$DUMP/eval.log" 2>&1
set -e
# 평가기가 연결 전에 죽으면 --once 재생 서버가 8010 에서 계속 기다린다 -> 30 초 기다린 뒤 이 실행의 서버만 끈다(pid)
for _ in $(seq 30); do kill -0 $SRV_PID 2>/dev/null || break; sleep 1; done
kill $SRV_PID 2>/dev/null || true
echo "=== 렌더 기준 자료 끝: $DUMP ==="
