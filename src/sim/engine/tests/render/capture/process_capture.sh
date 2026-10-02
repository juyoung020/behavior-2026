#!/usr/bin/env bash
# 뜬 렌더 기준 자료를 렌더러 입력·비교용으로 바꾸고, 공식 자신의 잡음 폭을 잰다 (GPU 안 씀).
# (리눅스판; 옛 Windows 판 archive/src/sim/engine/tests/render/capture/process_capture.ps1 과 같은 선택지·순서·결과)
#   bash src/sim/engine/tests/render/capture/process_capture.sh [--tag radio_rgbd] [--tex-max 1024] [--ref224 <결과폴더,...>]
# 1) export_render_scene.py : .npy 묶음 + 공식 영상(off_/noise<r>_/ref224_<실행>_)  -> dumps/render_<tag>/export
# 2) convert_scene.py       : scene.rsc + frame_<k>.rfr (렌더러 입력)               -> dumps/render_<tag>/rsc
# 3) rsc_check              : 렌더러 불러오기로 읽어 개수·카메라·영상 목록 확인
# 4) render_compare         : off:noise0, noise0:noise1, off:ref224_bkref (공식끼리 = 잡음 폭)  -> export/noise_band.csv
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE=$(cd "$HERE/../../.." && pwd)
REPO=$(cd "$ENGINE/../../.." && pwd)
TAG=radio_rgbd; TEX_MAX=1024
REF224="$REPO/outputs/eval_turning_on_radio_20260929_202701_bk_ref,$REPO/outputs/eval_turning_on_radio_20260929_203038_bk_waitidle,$REPO/outputs/eval_turning_on_radio_20260929_204753_bk_resetuser"
while [ $# -gt 0 ]; do
  case $1 in
    --tag) TAG=$2; shift 2 ;;
    --tex-max) TEX_MAX=$2; shift 2 ;;
    --ref224) REF224=$2; shift 2 ;;
    -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "모르는 인자: $1" >&2; exit 2 ;;
  esac
done
DUMP=$ENGINE/dumps/render_$TAG
[ -d "$DUMP" ] || { echo "렌더 기준 자료가 없다: $DUMP (capture_with_lock.sh 로 먼저 뜬다)" >&2; exit 2; }
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
CONDA_BASE=${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u; conda activate behavior; set -u
python "$HERE/export_render_scene.py" "$DUMP" --tex-max 64 --ref224 "$REF224"
python "$HERE/convert_scene.py" "$DUMP" --tex-max "$TEX_MAX" --out "$DUMP/rsc"
bash "$ENGINE/tests/render/compare/build.sh" > /dev/null
T=~/engine-build/render-tools
mapfile -t FR < <(find "$DUMP/rsc" -maxdepth 1 -name 'frame_*.rfr' | sort | head -2)
"$T/rsc_check" "$DUMP/rsc/scene.rsc" "${FR[@]}"
"$T/render_compare" "$DUMP/export" --pairs off:noise0,noise0:noise1,off:ref224_bkref,ref224_bkref:ref224_bkwaitidle \
  --csv "$DUMP/export/noise_band.csv" | tail -20
