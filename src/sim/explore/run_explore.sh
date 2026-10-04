#!/bin/bash
# 스킬 explore 한 판(시뮬): 평가기(run_explore.py, 지도·move_robot) + 에이전트(explore, LLM 또는 기준선).
#   run_explore.sh <policy llm|frontier> <task> [tag] [extra agent args...]
# 결과: outputs/explore_<ts>_<task>_<policy>[_tag]/ (memory/ 지도, decisions.jsonl, timeline.jsonl, summary.json, sim.log)
# 사전 조건: VRAM 여유 ≥ 9 GB, RAM 여유 ≥ 16 GB (아니면 기다린다). 키는 환경변수로만(kau.env).
# 로봇: 기본 R1 Pro. SGRT_ROBOT=limo_omx 면 우리 LIMO + OMX-F — 평가기를 $ROBOT_AGENT/src/robot/og/eval_with_limo.py 로
#   띄우고(--robot-config limo_omx_eval.yaml), move_robot 은 베이스만(팔 홈 자세·그리퍼 닫힘 유지, move_robot_limo.py),
#   정답 자세·물체 기록(SGRT_GT_LOG=<out>/gt_poses.csv, .objects.json)·poses.csv·pose_diag.json 기본 켬. 가까운 자르기는
#   robot-agent 391c04b 부터 eval_with_limo.py 가 0.05 m 로 둔다(옛 자산이면 LIMO_NEAR_CLIP, 기본 0.05 까지만 올림).
#   libmove_robot 몸 크기는 MOVE_ROBOT_FOOTPRINT=limo_omx(기본, 바꾸려면 rect:LxW·circle:R). R1 은 설정 안 함(원 0.37 그대로).
set -u
POL=$1; TASK=$2; TAG=${3:-}; shift 3 2>/dev/null || shift $#
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)            # behavior-2026
SUPER=${ROBOT_AGENT:-$(cd "$REPO/../.." && pwd)}   # robot-agent (behavior-2026 을 서브모듈 밖에서 돌리면 ROBOT_AGENT=~/robot-agent)
[ -x "$SUPER/src/agent/skills/explore/target/release/explore" ] || SUPER=$HOME/robot-agent
TS=$(date +%Y%m%d_%H%M%S)
OUT=$REPO/outputs/explore_${TS}_${TASK}_${POL}${TAG:+_$TAG}
ROBOT=${SGRT_ROBOT:-r1pro}
case "$ROBOT" in r1pro|limo_omx) ;; *) echo "[run] SGRT_ROBOT=$ROBOT 모름 (r1pro | limo_omx)"; exit 1;; esac
export MOVE_ROBOT_LIB=${MOVE_ROBOT_LIB:-$SUPER/src/agent/tools/move_robot/target/release/libmove_robot.so}   # 서브모듈 밖(클론)에서도
GT=$HERE/gt; [ -d "$GT" ] || GT=$HOME/behavior-2026/src/sim/explore/gt   # gt/ 는 git 밖(gt_trav.py 로 만듦)
ROBOT_ARGS=(); EVAL_ROBOT=()
if [ "$ROBOT" = limo_omx ]; then
  OGDIR=${LIMO_OG_DIR:-$SUPER/src/robot/og}
  [ -f "$OGDIR/eval_with_limo.py" ] || OGDIR=$HOME/robot-agent/src/robot/og
  ROBOT_ARGS=(--robot limo_omx --limo-shim "${LIMO_SHIM:-$OGDIR/eval_with_limo.py}")
  EVAL_ROBOT=(--robot-config "$OGDIR/limo_omx_eval.yaml")
  export SGRT_ROBOT=limo_omx SGRT_GT_LOG=${SGRT_GT_LOG:-$OUT/gt_poses.csv}
  export MOVE_ROBOT_FOOTPRINT=${MOVE_ROBOT_FOOTPRINT:-limo_omx}   # libmove_robot 몸통: LIMO 0.36 × 0.22 m 사각형(R1 은 원 0.37)
fi
PORT=${PORT:-8771}
MAXSTEPS=${MAXSTEPS:-27000}
mkdir -p "$OUT"
while true; do
  fv=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, '{print $1-$2}')
  fr=$(free -g | awk '/Mem:/{print $7}')
  [ "$fv" -ge 9000 ] && [ "$fr" -ge 16 ] && break
  echo "[run] waiting: VRAM free $fv MiB, RAM avail $fr GB"; sleep 30
done
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavior
export OMNI_KIT_ACCEPT_EULA=YES
export SGRT_POSE=${SGRT_POSE:-slam}   # 실제 로봇과 같게 slam(오도메트리 + 스캔 맞추기). 정답 자세 확인용은 SGRT_POSE=gt
export SGRT_LIB=${SGRT_LIB:-$HOME/sgrt_build_explore/libsgrt.so}
if [ -n "${SGRT_STREAM:-}" ] && ! strings "$SGRT_LIB" | grep -q SGRT_STREAM; then echo "[run] $SGRT_LIB 에 SGRT_STREAM 이 없다(옛 빌드) — 다시 빌드할 것"; exit 1; fi   # 뷰어가 조용히 비는 실수 방지
if [ "$ROBOT" = limo_omx ] && ! grep -aqF sgrt_set_robot "$SGRT_LIB"; then echo "[run] $SGRT_LIB 에 로봇 고르기(sgrt_set_robot)가 없다(옛 빌드) — 다시 빌드할 것"; exit 1; fi
cd "$OUT"
python "$HERE/run_explore.py" --listen 127.0.0.1:$PORT --out "$OUT" --gt-dir "$GT" "${ROBOT_ARGS[@]}" -- --task-name "$TASK" --mode public_test \
  --instance-indices 0 --num-envs 1 --max-steps $MAXSTEPS --headless "${EVAL_ROBOT[@]}" \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper > "$OUT/sim.log" 2>&1 &
SIM=$!
echo "[run] sim pid $SIM out $OUT"
# 평가기가 첫 관측을 받을 때까지(지도가 생김)
for i in $(seq 1 120); do
  grep -q "reference set\|Traceback" "$OUT/sim.log" 2>/dev/null && break
  kill -0 $SIM 2>/dev/null || break
  sleep 5
done
grep -q "Traceback" "$OUT/sim.log" && { echo "[run] sim failed"; tail -30 "$OUT/sim.log"; kill $SIM 2>/dev/null; exit 1; }
set -a; . ~/.config/behavior-2026/kau.env; set +a
"$SUPER/src/agent/skills/explore/target/release/explore" --policy "$POL" --addr 127.0.0.1:$PORT --out "$OUT" --task "$TASK" \
  --max-calls ${MAXCALLS:-80} --max-sim-s ${MAXSIM:-880} --max-wall-s ${MAXWALL:-3600} "$@" > "$OUT/agent.log" 2>&1
echo "[run] agent done: $(tail -c 300 $OUT/agent.log | tr '\n' ' ')"
kill -INT $SIM 2>/dev/null; sleep 20; kill $SIM 2>/dev/null; wait $SIM 2>/dev/null
echo "[run] finished $OUT"
