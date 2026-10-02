#!/bin/bash
# Reference dumps (JAX). usage: run_dumps.sh <platform cpu|cuda> <tag> <samples> <layer-samples> [extra args]
# CPU runs are pinned to half the cores so a simulator running at the same time keeps its CPU.
set -e
P=$1; TAG=$2; S=$3; LS=$4; shift 4
OUT=/mnt/c/behavior-2026/data/pi05_native/ref
if [ "$P" = cpu ]; then PIN="taskset -c 0-15 nice -n 10"; else PIN=""; fi
JAX_PLATFORMS=$P $PIN bash /mnt/c/behavior-2026/src/vla/pi05_native/tools/wsl_py.sh \
  /mnt/c/behavior-2026/src/vla/pi05_native/tools/dump_reference.py --tag $TAG --out $OUT --samples $S --layer-samples "$LS" "$@" 2>&1 \
  | grep -v -i -E "warn|^\s*$|jax_plugins|cuda_plugin"
