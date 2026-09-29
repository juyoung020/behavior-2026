#!/usr/bin/env bash
# tracecmp 빌드(WSL, Rust 1.81+). 산출물은 저장소 밖 ~/cargo-target/tracecmp/release/tracecmp (tools/exp_run.ps1 compare 가 이 경로를 쓴다)
#   wsl -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/src/fasteval/tracecmp/build.sh
set -euo pipefail
cd "$(dirname "$0")"
export CARGO_TARGET_DIR=$HOME/cargo-target/tracecmp
cargo build --release
cargo test --release
bash verify_vs_python.sh
