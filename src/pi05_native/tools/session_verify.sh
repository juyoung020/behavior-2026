#!/bin/bash
# One GPU-lock session: JAX GPU reference dumps (+bench), CPU teacher-forced floor dumps, native verify.
# usage: session_verify.sh [steps...]  steps: gpu cpuplant verify gemm (default: all)
T=/mnt/c/behavior-2026/src/pi05_native/tools
B=~/pi05_native_build
R=/mnt/c/behavior-2026/data/pi05_native/ref
W=/mnt/c/behavior-2026/data/pi05_native/pi05_radio.pi05w
STEPS=${@:-gpu cpuplant verify gemm}
source /mnt/c/behavior-2026/src/engine/scripts/gpu_lock.sh
gpu_lock_acquire "pi05_native" "JAX GPU dumps + native layer check (+CPU JAX 19 GB RAM)" 30 "10GB"
trap gpu_lock_release EXIT
for s in $STEPS; do
  echo "=== $s $(date +%T)"
  case $s in
    gpu) bash $T/run_dumps.sh cuda gpu 0-15 0-3 --bench 20 ;;
    cpuplant) bash $T/run_dumps.sh cpu cpu_planted 0-3 0-3 --plant $R --plant-tag gpu ;;
    verify) $B/pi05_verify --weights $W --ref $R --tag gpu --floor cpu --floor-single cpu_planted --samples 0-3 --time ;;
    verify8) $B/pi05_verify --weights $W --ref $R --tag gpu --floor cpu --samples 0-7 --mode chain ;;
    gemm) bash $T/gemm_run.sh --bench ;;
  esac
done
echo "=== done $(date +%T)"
