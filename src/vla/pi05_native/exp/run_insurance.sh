#!/usr/bin/env bash
# Native pi0.5 experiments through the shared experiment runner (tools/exp_run.py: GPU lock + queue per run,
# black-frame detection, result table). The evaluator itself is unmodified; our policy goes in with --native-policy.
# (Linux version of the old Windows runner archive/src/vla/pi05_native/exp/run_insurance.ps1, same options.)
#   insurance (2025 1st place, 4 checkpoints, tasks 0-49 by 2025 score):
#     bash src/vla/pi05_native/exp/run_insurance.sh
#   one full radio episode with the openpi radio checkpoint:
#     bash src/vla/pi05_native/exp/run_insurance.sh --matrix src/vla/pi05_native/exp/radio_native_full.json --model radio
# options: --matrix <json> (default pb2025_insurance.json next to this script), --build <dir> (default build, see ../build_linux.sh),
#          --model pb2025|radio (exported as PI05_MODEL), --reuse, --dry-run
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
MATRIX=$HERE/pb2025_insurance.json; BUILD=build; MODEL=pb2025; EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in
    --matrix) MATRIX=$2; shift 2 ;;
    --build) BUILD=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --reuse) EXTRA+=(--reuse); shift ;;
    --dry-run) EXTRA+=(--dry-run); shift ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case $MODEL in pb2025|radio) ;; *) echo "--model must be pb2025|radio" >&2; exit 2 ;; esac
GLUE=$(realpath "$HERE/../glue")
BIN=$(realpath "$HERE/../$BUILD")
export PYTHONPATH="$GLUE:$BIN${PYTHONPATH:+:$PYTHONPATH}"
export PI05_MODEL=$MODEL
echo "PYTHONPATH=$PYTHONPATH  PI05_MODEL=$PI05_MODEL"
exec python3 "$REPO/tools/exp_run.py" run --matrix "$MATRIX" --backend original ${EXTRA[@]+"${EXTRA[@]}"}
