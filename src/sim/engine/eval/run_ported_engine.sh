#!/usr/bin/env bash
# 포팅 평가기(--backend engine v0)를 Linux(WSL) 에서 공식과 같은 인자로 돌린다. Kit·GPU 없이 CPU 로만 돈다.
#   bash /mnt/c/behavior-2026/src/sim/engine/eval/run_ported_engine.sh <판 기록 폴더> <actions.npz> <출력 폴더> [최대 스텝=500] [모드=public_test] [인스턴스 번호=0]
# 판 기록 폴더 = 공식 기록(run_capture_linux.sh 결과, export_s1.py 까지 한 것). 에피소드 앞을 장면 추출물로 쓴다.
# 결과: <출력 폴더>/json/*.json (공식 결과 JSON 과 같은 꼴), trace.npz (tools/trace_compare.py 로 공식 trace 와 비교)
set -euo pipefail
ulimit -c 0
REC=${1:?판 기록 폴더}
ACTIONS=${2:?actions.npz}
OUT=${3:?출력 폴더}
MAXSTEPS=${4:-500}
MODE=${5:-public_test}
IDX=${6:-0}
BASE=~/behavior-linux
E=$BASE/.venv/lib/python3.11/site-packages/isaacsim/extscache
U=$(ls -d $E/omni.usd.libs-*)
D=$(ls -d $E/usdrt.scenegraph-*)
PYL=$(dirname "$(find ~/.local/share/uv/python -name 'libpython3.11.so.1.0' | head -1)")
export LD_LIBRARY_PATH=$PYL:$U/bin:$D/bin:${LD_LIBRARY_PATH:-}
export ENGINE_REC_DIR=$REC
export ENGINE_CAM_CHAIN=${ENGINE_CAM_CHAIN:-$REC/scope.json}
# 관측 영상(렌더 모듈 층 1): ENGINE_RENDER_RSC 를 주면 그린다 (없으면 0 영상). 예) radio 인스턴스 301:
#   ENGINE_RENDER_RSC=/mnt/c/behavior-2026/src/sim/engine/dumps/render_radio_rgbd/rsc ENGINE_RENDER_FRAME=$ENGINE_RENDER_RSC/frame_0000.rfr
#   ENGINE_RENDER_META=/mnt/c/behavior-2026/src/sim/engine/dumps/render_radio_rgbd/export/meta.json
export OMNIGIBSON_DATA_PATH=/mnt/c/behavior-2026/BEHAVIOR-1K/datasets
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUTF8=1
mkdir -p "$OUT"
PORT=8013
/home/juyoung/openpi/.venv/bin/python /mnt/c/behavior-2026/tools/replay_policy_server.py \
  --actions "$ACTIONS" --port $PORT --log "$OUT/server_log.npz" --once > "$OUT/replay_server.log" 2>&1 &
SRV=$!
for i in $(seq 60); do curl -sf http://127.0.0.1:$PORT/healthz >/dev/null && break; sleep 1; done
cd $BASE/BEHAVIOR-1K/OmniGibson
set +e
$BASE/.venv/bin/python /mnt/c/behavior-2026/src/sim/engine/eval/ported_eval.py --backend engine --instrument=--trace -- \
  --task-name turning_on_radio --robot-config /mnt/c/behavior-2026/src/sim/configs/r1pro_openpi.yaml \
  --env-wrapper omnigibson.eval.wrappers.DefaultWrapper --mode "$MODE" \
  --host 127.0.0.1 --port $PORT --instance-indices "$IDX" --num-envs 1 --max-steps "$MAXSTEPS" \
  --output-dir "$OUT" --headless 2>&1 | tee "$OUT/eval.log"
CODE=${PIPESTATUS[0]}
set -e
# 재생 서버는 --once 라 접속이 끊기면 server_log.npz 를 쓰고 스스로 끝난다 -> 쓰는 중에 죽이지 않게 기다린다
for i in $(seq 60); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
kill $SRV 2>/dev/null || true
echo "=== 포팅 평가기 끝 (exit $CODE): $OUT ==="
