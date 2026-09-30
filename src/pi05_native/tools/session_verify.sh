#!/bin/bash
# One GPU-lock session: JAX GPU reference dumps (+bench), CPU teacher-forced floor dumps, native verify.
# usage: session_verify.sh [steps...]  steps: gpu cpuplant verify gemm (default: all)
T=/mnt/c/behavior-2026/src/pi05_native/tools
B=~/pi05_native_build
R=/mnt/c/behavior-2026/data/pi05_native/ref
W=/mnt/c/behavior-2026/data/pi05_native/pi05_radio.pi05w
STEPS=${@:-gpu cpuplant verify gemm}
source /mnt/c/behavior-2026/src/engine/scripts/gpu_lock.sh
gpu_lock_acquire "pi05_native" "pi05 native: ${STEPS}" ${PI05_LOCK_MIN:-20} 8
trap "gpu_lock_release pi05_native" EXIT  # owner given: never removes another holder's lock
for s in $STEPS; do
  echo "=== $s $(date +%T)"
  case $s in
    gpu) bash $T/run_dumps.sh cuda gpu 0-15 0-3 --bench 20 ;;
    cpuplant) bash $T/run_dumps.sh cpu cpu_planted 0-3 0-3 --plant $R --plant-tag gpu ;;
    verify) $B/pi05_verify --weights $W --ref $R --tag gpu --floor cpu --floor-single cpu_planted --samples 0-3 --time ;;
    verify8) $B/pi05_verify --weights $W --ref $R --tag gpu --floor cpu --samples 0-7 --mode chain ;;
    gemm) bash $T/gemm_run.sh --bench ;;
    gemmtest) $B/gemm_test | tail -3 ;;
    pbdump) OPENPI_BEHAVIOR_DATA_ROOT=/mnt/c/behavior-2026/data JAX_PLATFORMS=cuda bash $T/wsl_py.sh $T/dump_reference_pb.py --tag gpu --samples 0-3 --layers 0-1 --bench 10 2>&1 | grep -v -i -E "warn|^s*$" ;;
    pbverify) $B/pi05_verify_pb --weights /mnt/c/behavior-2026/data/pi05_native/pb2025_ckpt2.pi05w --ref /mnt/c/behavior-2026/data/pi05_native/ref_pb --tag gpu --floor cpu --samples 0-3 --time ;;
    pbcpu) OPENPI_BEHAVIOR_DATA_ROOT=/mnt/c/behavior-2026/data JAX_PLATFORMS=cpu taskset -c 0-15 bash $T/wsl_py.sh $T/dump_reference_pb.py --tag cpu --samples 0-1 --layers 0-1 2>&1 | grep -v -i warn ;;
    pbcpu23) OPENPI_BEHAVIOR_DATA_ROOT=/mnt/c/behavior-2026/data JAX_PLATFORMS=cpu taskset -c 0-15 bash $T/wsl_py.sh $T/dump_reference_pb.py --tag cpu --samples 2-3 --layers 2-3 2>&1 | grep -v -i warn ;;
    pbv01) $B/pi05_verify_pb --weights /mnt/c/behavior-2026/data/pi05_native/pb2025_ckpt2.pi05w --ref /mnt/c/behavior-2026/data/pi05_native/ref_pb --tag gpu --floor cpu --samples 0-1 | grep -E "^(suf.s1[0-9]|suf.s[5-9].x|actions|inp|FAIL|PASS)" ;;
    pbv23) $B/pi05_verify_pb --weights /mnt/c/behavior-2026/data/pi05_native/pb2025_ckpt2.pi05w --ref /mnt/c/behavior-2026/data/pi05_native/ref_pb --tag gpu --samples 2-3 | grep -E "^(suf.s1[5-9]|actions|FAIL|PASS)" ;;
    server) $B/pi05_server --weights /mnt/c/behavior-2026/data/pi05_native/pi05_radio.pi05w --port 8765 > /tmp/pi05_server.log 2>&1 &
            SP=$!; for i in $(seq 60); do grep -q ready /tmp/pi05_server.log && break; sleep 1; done; cat /tmp/pi05_server.log
            bash $T/wsl_py.sh $T/server_client_test.py --port 8765 --steps 64 2>&1 | grep -v -i warn; kill $SP ;;
    batch) $B/pi05_batch_test --weights $W --ref $R --cap ${PI05_BATCH_CAP:-64} ;;
    pbbatch) $B/pi05_batch_test --weights /mnt/c/behavior-2026/data/pi05_native/pb2025_ckpt2.pi05w --ref /mnt/c/behavior-2026/data/pi05_native/ref_pb --cap ${PI05_BATCH_CAP:-64} ;;
    srvradio) bash $T/server_check.sh radio 2 60 0; bash $T/server_check.sh radio 1 40 8 ;;
    srvpb) bash $T/server_check.sh pb 2 60 0 ;;
    quick) $B/pi05_verify --weights $W --ref $R --tag gpu --floor cpu --floor-single cpu_planted --samples 0-3 --time | grep -E "^(sample|actions|jax|graph|eager|device|PASS|FAIL)|final max" ;;
    winradio) /mnt/c/behavior-2026/src/pi05_native/${WINB:-build_win_next}/pi05_verify.exe --weights C:/behavior-2026/data/pi05_native/pi05_radio.pi05w --ref C:/behavior-2026/data/pi05_native/ref --tag gpu --floor cpu --floor-single cpu_planted --samples 0-3 --time | grep -E "^(sample|actions|graph|eager|device|PASS|FAIL)|final max" ;;
    winpb) /mnt/c/behavior-2026/src/pi05_native/${WINB:-build_win_next}/pi05_verify_pb.exe --weights C:/behavior-2026/data/pi05_native/pb2025_ckpt2.pi05w --ref C:/behavior-2026/data/pi05_native/ref_pb --tag gpu --samples 0-3 --time | tail -14 ;;
  esac
done
echo "=== done $(date +%T)"
