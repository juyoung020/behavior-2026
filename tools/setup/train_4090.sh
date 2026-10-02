#!/usr/bin/env bash
# π0.5 native training on an RTX 4090 (Ubuntu 22.04/24.04) — docs/π05_네이티브엔진.md 15절.
# Written on the RTX 5070 Ti WSL machine; NOT run on a 4090 yet. Each step can be re-run.
#
#   bash train_4090.sh check            # driver / GPU / disk
#   bash train_4090.sh deps             # CUDA 12.8 toolkit, build tools, Rust, uv, ffmpeg (sudo)
#   bash train_4090.sh repos            # this repo + openpi (wensi-ai behavior branch) + fast-data patch, uv sync
#   bash train_4090.sh fetch            # data: from the old PC (SRC=user@host) or Hugging Face (task 0), checkpoint
#   bash train_4090.sh build            # engine (sm_89), trainer, fasttrain loader
#   bash train_4090.sh prepare          # colour LUT + check, fasttrain table, weight file, initial training state
#   bash train_4090.sh verify           # GEMM / loader checks (short)
#   MODE=expert|lora STEPS=30000 bash train_4090.sh train
#   bash train_4090.sh export STATE=...  # trained state -> openpi checkpoint -> engine weight file
set -euo pipefail
STEP=${1:-}
REPO=${REPO:-$HOME/behavior-2026}
OPENPI=${OPENPI:-$HOME/openpi}
DATA=${DATA:-$HOME/data/2026-challenge-demos}
CKPT=${CKPT:-$HOME/checkpoints/pi05_turning_on_radio/pi05_turn_on_the_radio}   # init + norm stats (radio release)
RUN=${RUN:-$HOME/pi05_runs}
MODE=${MODE:-expert}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH=$CUDA_HOME/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH
export FT_WORK=${FT_WORK:-$HOME/fasttrain_work} FT_CUDA_ARCH=89 PI05_ARCH="-gencode arch=compute_89,code=sm_89"
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}
PY="$OPENPI/.venv/bin/python"
NB=$HOME/pi05_native_build TB=$HOME/pi05_train_build

case "$STEP" in
check)
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
  df -h "$HOME" | tail -1
  echo "need: driver >= 570 (CUDA 12.8), compute_cap 8.9, ~60 GB free (checkpoint 17 GB, data 4.9 GB/task, states 2-7 GB each)"
  ;;
deps)
  sudo apt-get update
  sudo apt-get install -y build-essential git git-lfs rsync curl wget cmake pkg-config ffmpeg libssl-dev
  if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then  # toolkit only (the driver comes from linux_setup.sh driver)
    . /etc/os-release
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID/./}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/ck.deb
    sudo dpkg -i /tmp/ck.deb && sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8
  fi
  command -v cargo >/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  ;;
repos)
  [ -d "$REPO" ] || git clone https://github.com/juyoung020/behavior-2026.git "$REPO"   # private: gh auth login first
  if [ ! -d "$OPENPI" ]; then
    git clone -b behavior https://github.com/wensi-ai/openpi.git "$OPENPI"
    (cd "$OPENPI" && git am "$REPO/src/fasttrain/openpi-fast-data.patch")
  fi
  (cd "$OPENPI" && GIT_LFS_SKIP_SMUDGE=1 uv sync)
  ;;
fetch)
  mkdir -p "$(dirname "$DATA")" "$(dirname "$CKPT")"
  if [ -n "${SRC:-}" ]; then   # copy from the old PC (WSL paths), e.g. SRC=juyoung@192.168.0.10
    rsync -a --info=progress2 "$SRC:data/2026-challenge-demos/" "$DATA/"
    rsync -a --info=progress2 "$SRC:checkpoints/pi05_turning_on_radio/" "$(dirname "$CKPT")/"
  else
    # task 0 (turning_on_radio) of the demos: meta + chunk-000 of data and the three RGB videos (4.9 GB with depth)
    "$OPENPI/.venv/bin/huggingface-cli" download behavior-1k/2026-challenge-demos --repo-type dataset \
      --local-dir "$DATA" --include "meta/*" "data/chunk-000/*" "videos/*rgb*/chunk-000/*"
    echo "checkpoint: download the pi0.5 radio release (docs/베이스라인.md 2절, Google Drive) into $(dirname "$CKPT")"
  fi
  ;;
build)
  bash "$REPO/src/pi05_native/build_linux.sh" "$NB"
  PI05_NATIVE_BUILD=$NB bash "$REPO/src/pi05_train/build_linux.sh" "$TB"
  bash "$REPO/src/fasttrain/build.sh"
  PI05_NATIVE_BUILD=$NB bash "$REPO/src/pi05_train/build_linux.sh" "$TB"   # pi05_train links libftcore.so
  ls -la "$TB"/pi05_train "$NB"/pi05_server
  ;;
prepare)
  mkdir -p "$FT_WORK/assets/pi05_b1k" "$RUN"
  cp -r "$CKPT/assets/turning_on_radio" "$FT_WORK/assets/pi05_b1k/" 2>/dev/null || cp -r "$CKPT/assets/"* "$FT_WORK/assets/pi05_b1k/"
  "$PY" "$REPO/src/fasttrain/lut.py" build && "$PY" "$REPO/src/fasttrain/lut.py" check
  "$PY" -c "from fasttrain import fast, orig; print(fast.ensure_table(orig.train_config()))" | tee "$RUN/table_dir.txt"
  "$PY" "$REPO/src/pi05_native/tools/export_weights.py" --ckpt "$CKPT" --asset turning_on_radio --out "$RUN/init.pi05w"
  for m in expert lora; do
    "$PY" "$REPO/src/pi05_train/tools/make_state.py" --ckpt "$CKPT" --mode $m --out "$RUN/state_${m}_init.pi05d"
  done
  ;;
verify)
  "$TB/tgemm_test" | tail -1
  "$PY" "$REPO/tools/ft_verify.py" order && "$PY" "$REPO/tools/ft_verify.py" loader --batches 2
  ;;
train)
  STATE=${STATE:-$RUN/state_${MODE}_init.pi05d}
  mkdir -p "$RUN/$MODE"
  # 24 GB: optimizer state stays on the GPU (--offload 0); use 1 (EMA on host) or 2 if another process needs memory
  "$TB/pi05_train" --state "$STATE" --model "$RUN/init.pi05w" --table "$(tail -1 "$RUN/table_dir.txt")" \
    --lut "$FT_WORK/lut/lut_w720.bin" --steps "${STEPS:-30000}" --offload "${OFFLOAD:-0}" --save-every "${SAVE:-1000}" \
    --log-every 10 --out "$RUN/$MODE" 2>&1 | tee -a "$RUN/$MODE/train.log"
  ;;
export)
  : "${STATE:?STATE=<run>/state_stepN.pi05d}"
  OUT=${OUT:-$RUN/export_$(basename "$STATE" .pi05d)}
  "$PY" "$REPO/src/pi05_train/tools/state_to_orbax.py" --state "$STATE" --base "$CKPT" --out "$OUT" --ema --merge-lora
  cp -r "$CKPT/assets" "$OUT/"
  "$PY" "$REPO/src/pi05_native/tools/export_weights.py" --ckpt "$OUT" --asset turning_on_radio --out "$OUT.pi05w"
  echo "serve: $NB/pi05_server --weights $OUT.pi05w --port 8000"
  ;;
*)
  sed -n 2,15p "$0"
  ;;
esac
