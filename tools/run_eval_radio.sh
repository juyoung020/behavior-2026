#!/usr/bin/env bash
# 공식 평가기(omnigibson.eval.eval)를 돌린다 (리눅스판; 옛 Windows 판 archive/tools/run_eval_radio.ps1 과 같은 선택지·동작).
# 정책 서버(포트 8000, 또는 --port 의 재생 서버)가 먼저 떠 있어야 한다(--policy local 이면 서버 없이 공식 0 행동 정책).
#   tools/run_eval_radio.sh [선택지]
#   --task 과제 (기본 turning_on_radio)       --instances '0,1' 처럼 여러 개 -> --num-envs 자동 (기본 --instance 0 하나)
#   --max-steps 0 = 공식 기본 제한시간(사람 평균 x 1.5). 제출용 결과는 반드시 0(기본)으로.
#   --wrapper Default(RGB 224) | RGBD(공식 RGB-D 720/480)
#   --gui 시뮬레이터 창.  --port 정책 서버 포트(재생 서버는 8010).  --tag 결과 폴더 이름 뒤에 붙일 말, --out-dir 결과 폴더 직접 지정
#   --timing / --trace : tools/eval_instrumented.py 로 감싸 구간별 시간(timing.json) / 스텝별 기록(trace.npz)을 결과 폴더에.
#   --chunk-size 16    : 공식 인자 --replay-action-chunk-size (서버가 action_chunk 를 줘야 함)
#   --black-guard warn|abort : 정책에 들어갈 카메라 영상이 전부 0(검은 화면)이면 경고/중단.  --black-diag : 호스트/GPU 경로 비교(진단)
#   --kit-set '/키=값' : 진단 실험용 Kit 설정 바꾸기 (여러 번 줄 수 있음, 공식 결과에는 쓰지 않는다)
#   --robot-config 경로 : 로봇 설정 (기본 src/sim/configs/r1pro_robot.yaml = 로봇 이름 robot).
#                      robot_r1 키를 쓰는 서버에는 공식 BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml 을 준다.
#                      none : --robot-config 를 아예 안 넘김(평가기 기본 = 공식 eval/r1pro.yaml, 수정 0 재현용)
#   --deep            : 컨트롤러 콜백·물체 상태 캐시 안쪽까지 잘게 (--timing 과 같이). 평가기 코드는 안 바꾼다.
#   --policy local    : 공식 평가기의 0 행동 정책(서버 없이, 평가기 점검용 공식 옵션)
#   --kit-arg '--/app/vulkan=false' : 진단용 Kit 시작 인자 (여러 번)
#   --render-iters N  : 진단용 -- 스텝마다 렌더를 N 번(공식은 1, eval/evaluator.py:383). 물리·판정은 안 바뀌어야 한다
#   --vk-nvidia-only  : 진단용 -- 이 실행에서만 Vulkan 이 NVIDIA ICD 만 보게 한다(VK_DRIVER_FILES). 시스템 설정은 안 바꾼다
#   --dry-run         : 명령만 찍고 안 돈다
# 결과: outputs/eval_<과제>_<시각>[_태그]/ , 로그: logs/<같은 이름>.log
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

TASK=turning_on_radio; INSTANCE=0; INSTANCES=''; MAX_STEPS=0; WRAPPER=Default; PORT=8000; TAG=''; OUT_DIR=''
CHUNK=0; GUI=0; TIMING=0; TRACE=0; DEEP=0; BLACK_DIAG=0; BLACK_GUARD=''; KIT_SET=(); KIT_ARG=()
ROBOT_CONFIG=$REPO/src/sim/configs/r1pro_robot.yaml; POLICY=websocket; RENDER_ITERS=0; VK_NV=0; DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --task) TASK=$2; shift 2 ;;
    --instance) INSTANCE=$2; shift 2 ;;
    --instances) INSTANCES=$2; shift 2 ;;
    --max-steps) MAX_STEPS=$2; shift 2 ;;
    --wrapper) WRAPPER=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --out-dir) OUT_DIR=$2; shift 2 ;;
    --chunk-size) CHUNK=$2; shift 2 ;;
    --gui) GUI=1; shift ;;
    --timing) TIMING=1; shift ;;
    --trace) TRACE=1; shift ;;
    --deep) DEEP=1; shift ;;
    --black-diag) BLACK_DIAG=1; shift ;;
    --black-guard) BLACK_GUARD=$2; shift 2 ;;
    --kit-set) KIT_SET+=("$2"); shift 2 ;;
    --kit-arg) KIT_ARG+=("$2"); shift 2 ;;
    --robot-config) ROBOT_CONFIG=$2; shift 2 ;;
    --policy) POLICY=$2; shift 2 ;;
    --render-iters) RENDER_ITERS=$2; shift 2 ;;
    --vk-nvidia-only) VK_NV=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "모르는 인자: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case $WRAPPER in
  Default) WRAPPER_TARGET=omnigibson.eval.wrappers.DefaultWrapper ;;
  RGBD) WRAPPER_TARGET=omnigibson.eval.wrappers.RGBDFullResWrapper ;;
  *) echo "--wrapper 는 Default|RGBD" >&2; exit 2 ;;
esac
case $BLACK_GUARD in ''|warn|abort) ;; *) echo "--black-guard 는 warn|abort" >&2; exit 2 ;; esac
case $POLICY in websocket|local) ;; *) echo "--policy 는 websocket|local" >&2; exit 2 ;; esac

HEADLESS=--headless; [ $GUI = 1 ] && HEADLESS=--no-headless
STEP_ARGS=()
[ "$MAX_STEPS" -gt 0 ] && STEP_ARGS+=(--max-steps "$MAX_STEPS")
[ "$CHUNK" -gt 1 ] && STEP_ARGS+=(--replay-action-chunk-size "$CHUNK")
if [ -n "$INSTANCES" ]; then read -r -a INST <<< "${INSTANCES//,/ }"; else INST=("$INSTANCE"); fi

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 OMNI_KIT_ACCEPT_EULA=YES
# torch/MKL 와 conda llvm-openmp 가 같이 올라와 'OMP: Error #15' 로 죽는 것의 공식 안내 우회책 (이 실행에만)
export KMP_DUPLICATE_LIB_OK=TRUE
if [ $VK_NV = 1 ]; then
  ICD=''
  for f in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json; do
    [ -f "$f" ] && { ICD=$f; break; }
  done
  export VK_DRIVER_FILES=$ICD VK_ICD_FILENAMES=$ICD
  echo "Vulkan ICD (이 실행만): ${ICD:-못 찾음}"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
NAME="eval_${TASK}_$STAMP${TAG:+_$TAG}"
[ -n "$OUT_DIR" ] && NAME=$(basename "$OUT_DIR")
OUT=${OUT_DIR:-$REPO/outputs/$NAME}
LOG=$REPO/logs/$NAME.log

RUNNER=(-m omnigibson.eval.eval)
if [ $TIMING = 1 ] || [ $TRACE = 1 ] || [ $BLACK_DIAG = 1 ] || [ -n "$BLACK_GUARD" ] || [ ${#KIT_SET[@]} -gt 0 ] \
   || [ "$RENDER_ITERS" -gt 0 ] || [ ${#KIT_ARG[@]} -gt 0 ]; then
  FLAGS=()
  [ $TIMING = 1 ] && FLAGS+=(--timing)
  [ $TRACE = 1 ] && FLAGS+=(--trace)
  [ $DEEP = 1 ] && FLAGS+=(--deep)
  [ $BLACK_DIAG = 1 ] && FLAGS+=(--black-diag)
  [ -n "$BLACK_GUARD" ] && FLAGS+=("--black-guard=$BLACK_GUARD")
  for kv in ${KIT_SET[@]+"${KIT_SET[@]}"}; do FLAGS+=("--set=$kv"); done
  [ "$RENDER_ITERS" -gt 0 ] && FLAGS+=("--render-iters=$RENDER_ITERS")
  for ka in ${KIT_ARG[@]+"${KIT_ARG[@]}"}; do FLAGS+=("--kit-arg=$ka"); done
  RUNNER=("$REPO/tools/eval_instrumented.py" ${FLAGS[@]+"${FLAGS[@]}"} --)
fi
ROBOT_ARGS=()
[ "$ROBOT_CONFIG" != none ] && ROBOT_ARGS=(--robot-config "$ROBOT_CONFIG")
CMD=(python "${RUNNER[@]}"
  --task-name "$TASK" ${ROBOT_ARGS[@]+"${ROBOT_ARGS[@]}"} --policy "$POLICY"
  --env-wrapper "$WRAPPER_TARGET"
  --mode public_test
  --host 127.0.0.1 --port "$PORT"
  --instance-indices "${INST[@]}" --num-envs "${#INST[@]}" ${STEP_ARGS[@]+"${STEP_ARGS[@]}"}
  --output-dir "$OUT" --write-video "$HEADLESS")
if [ $DRY = 1 ]; then
  echo "(dry-run) cd $REPO/BEHAVIOR-1K/OmniGibson && ${CMD[*]}"
  echo "(dry-run) 로그: $LOG"
  exit 0
fi

CONDA_BASE=${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u; conda activate behavior; set -u
mkdir -p "$OUT" "$REPO/logs"
cd "$REPO/BEHAVIOR-1K/OmniGibson"
set +e
"${CMD[@]}" 2>&1 | tee -a "$LOG"
CODE=${PIPESTATUS[0]}
set -e
echo "=== eval exit $CODE, 결과: $OUT ===" | tee -a "$LOG"
exit "$CODE"
