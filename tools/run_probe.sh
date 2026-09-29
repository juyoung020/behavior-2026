#!/usr/bin/env bash
# 지시 형식 오프라인 실험 (시뮬레이터 없음, π0.5 만 GPU) — 한 명령:
#   준비(없으면) → prompt 생성(Rust src/probe) → 추론(GPU, tools/probe_infer.py) → 채점·표(Rust) → logs/probe_<모델>_<시각>.md
#   GPU 사용량은 nvidia-smi 로 1초마다 같이 기록(_gpu.csv).
# 실행: wsl -d Ubuntu-22.04 -u juyoung -e bash /mnt/c/behavior-2026/tools/run_probe.sh [pt50|radio] [EPISODES] [PER_STAGE]
#   pt50  = 2위 Comet pt50 (~/openpi-comet), radio = 공식 radio 체크포인트 (~/openpi, pi05_b1k)
#   시작 전에 Windows 에서 Isaac Sim(평가기) 이 돌고 있지 않은지 확인할 것 (다른 프로세스가 GPU 를 잡으면 시뮬레이터 화면이 검어짐).
# 설명·결과: docs/실험_지시형식_오프라인.md
set -euo pipefail
MODEL=${1:-pt50}
EPISODES=${2:-20}
PER_STAGE=${3:-2}
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH CARGO_TARGET_DIR=$HOME/cargo-target/probe
T=/mnt/c/behavior-2026/tools
D=/mnt/c/behavior-2026/data/probe
S=$D/radio_e${EPISODES}_p${PER_STAGE}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$D/${MODEL}_$STAMP
case "$MODEL" in
  pt50)  PY=$HOME/openpi-comet/.venv/bin/python; EXTRA=(--input_style comet
           --policy_config pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k --policy_dir "$HOME/checkpoints/openpi_comet/pi05-b1kpt50-cs32") ;;
  radio) PY=$HOME/openpi/.venv/bin/python; EXTRA=(--input_style openpi --repo_id turning_on_radio --robot b1k/R1Pro
           --policy_config pi05_b1k --policy_dir "$HOME/checkpoints/pi05_turning_on_radio/pi05_turn_on_the_radio") ;;
  *) echo "모델: pt50 | radio"; exit 2 ;;
esac
cd /mnt/c/behavior-2026/src/probe && cargo build --release -q
PROBE=$CARGO_TARGET_DIR/release/probe
mkdir -p "$D"
[ -f "$S.npz" ] || JAX_PLATFORMS=cpu "$HOME/openpi-comet/.venv/bin/python" -u "$T/probe_prep.py" --episodes "$EPISODES" --per_stage "$PER_STAGE" --out "$S"
"$PROBE" prompts "$S.json" "$S.prompts.jsonl"
/usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > "${OUT}_gpu.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
cd "$(dirname "$(dirname "$(dirname "$PY")")")"   # 정책 저장소 루트 (상대 경로 설정 파일용)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" -u "$T/probe_infer.py" \
    --samples "$S" --prompts "$S.prompts.jsonl" --out "$OUT" "${EXTRA[@]}"
kill $SMI 2>/dev/null || true
"$PROBE" report "$S.json" "$S.prompts.jsonl" "$OUT" "/mnt/c/behavior-2026/logs/probe_${MODEL}_$STAMP.md" "radio 과제 · 모델 ${MODEL} · $STAMP" > /dev/null
cp "${OUT}_meta.json" "/mnt/c/behavior-2026/logs/probe_${MODEL}_${STAMP}_meta.json"
cp "${OUT}_gpu.csv" "/mnt/c/behavior-2026/logs/probe_${MODEL}_${STAMP}_gpu.csv"
echo "결과: logs/probe_${MODEL}_$STAMP.md"
