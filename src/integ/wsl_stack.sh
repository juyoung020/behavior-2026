#!/bin/bash
# WSL 쪽 통합 묶음: meridian 사슬(frontend → graphcore → scene_server) + simlink(계획기 + 관측 발행) 한 판.
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/src/integ/wsl_stack.sh \
#       <출력 폴더(/mnt/c/...)> [meridian 1|0] [llm kau|oracle|none] [simlink 인자 ...]
# 순서: (meridian 이면) launch → frontend 준비(/tracklet 발행자) 기다림 → VRAM 기록 → simlink(--once) → 준비 표시 파일
#       → 평가기(Windows)가 붙어 한 판 → 연결이 끝나면 simlink 종료 → 그래프 저장·요약 → meridian 끔 → 끝 표시 파일.
# 키는 ~/.config/behavior-2026/kau.env 에서 환경변수로만 읽는다(저장소·기록에 남기지 않음).
OUT=${1:?출력 폴더}; MER=${2:-1}; LLM=${3:-kau}; shift 3 2>/dev/null
mkdir -p "$OUT"
LOG="$OUT/wsl_stack.log"
exec > >(tee -a "$LOG") 2>&1
echo "[stack] $(date '+%F %T') out=$OUT meridian=$MER llm=$LLM args=$*"
source /opt/ros/humble/setup.bash
source ~/meridian_ws/install/setup.bash
# 통합 판은 자기 ROS 도메인(38)을 쓴다 — 다른 에이전트의 meridian 시험(37)과 토픽이 섞이지 않게
export ROS_DOMAIN_ID=${INTEG_ROS_DOMAIN_ID:-38}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/meridian_inductor}
export FASTRTPS_DEFAULT_PROFILES_FILE=$(ros2 pkg prefix meridian_scene)/share/meridian_scene/config/fastdds_shm.xml
SIMLINK=${SIMLINK:-$HOME/cargo-target/simlink/release/simlink}
HERE=/mnt/c/behavior-2026/src
rm -f "$OUT/wsl_ready" "$OUT/wsl_done"
if pgrep -x simlink >/dev/null; then echo "[stack] 다른 simlink 가 돌고 있다(pid $(pgrep -x simlink | tr '\n' ' ')) — 끝"; exit 1; fi
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
echo "gpu_used_mib_before=$(gpu)" > "$OUT/vram.txt"

LAUNCH=; REC=
if [ "$MER" = "1" ]; then
  pgrep -f meridian_pipeline.launch >/dev/null && echo "[stack] 참고: 다른 meridian launch 가 돌고 있다(도메인이 달라 토픽은 안 섞임, GPU 는 잠금이 가른다)"
  ros2 launch meridian_scene meridian_pipeline.launch.py frontend_venv:=$HOME/meridian_venv ${MERIDIAN_ARGS} \
      > "$OUT/meridian_launch.log" 2>&1 &
  LAUNCH=$!
  python3 "$HERE/meridian/commit_recorder.py" > "$OUT/commit_recorder.log" 2>&1 &
  REC=$!
  # frontend 는 torch.compile + CUDA graph 캡처로 수십 초 → /tracklet 발행자가 생기면 준비 끝
  for i in $(seq 1 300); do
    if ros2 topic info /tracklet 2>/dev/null | grep -q "Publisher count: [1-9]"; then break; fi
    sleep 1
  done
  sleep 3
  echo "[stack] meridian 준비 ($i s)"
  echo "gpu_used_mib_meridian_ready=$(gpu)" >> "$OUT/vram.txt"
fi

set -a
[ "$LLM" = "kau" ] && . ~/.config/behavior-2026/kau.env
set +a
PLAN=(--llm "$LLM")
[ "$LLM" = "none" ] && PLAN=(--no-planner)
GRAPH=none; [ "$MER" = "1" ] && GRAPH=meridian
"$SIMLINK" --listen 0.0.0.0:7801 "${PLAN[@]}" --graph $GRAPH --trace-dir "$OUT/trace" --once "$@" &
SL=$!
echo "LAUNCH=$LAUNCH REC=$REC SL=$SL STACK=$$" > "$OUT/wsl_pids"
for i in $(seq 1 50); do (echo > /dev/tcp/127.0.0.1/7801) 2>/dev/null && break; sleep 0.2; done
# nvidia-smi 표본(1 s) — 판 전체 VRAM
( while kill -0 $SL 2>/dev/null; do echo "$(date +%T),$(gpu)"; sleep 1; done ) > "$OUT/vram_timeline.csv" &
touch "$OUT/wsl_ready"
echo "[stack] simlink 준비 — 평가기를 기다림"
wait $SL
echo "[stack] simlink 끝 ($?)"
if [ "$MER" = "1" ]; then
  sleep 4
  ros2 service call /save_graph meridian_msgs/srv/SaveGraph "{path: '/tmp/integ_graph.sparkdsg'}" 2>&1 | tail -1
  cp /tmp/integ_graph.sparkdsg "$OUT/graph.sparkdsg" 2>/dev/null
  python3 "$HERE/meridian/graph_summary.py" "$OUT/graph.sparkdsg" > "$OUT/graph_summary.txt" 2>&1
  kill -TERM $REC 2>/dev/null; timeout 10 tail --pid=$REC -f /dev/null
  kill -TERM $LAUNCH 2>/dev/null; timeout 30 tail --pid=$LAUNCH -f /dev/null || kill -KILL $LAUNCH 2>/dev/null
fi
echo "gpu_used_mib_after=$(gpu)" >> "$OUT/vram.txt"
touch "$OUT/wsl_done"
echo "[stack] 끝 $(date '+%F %T')"
