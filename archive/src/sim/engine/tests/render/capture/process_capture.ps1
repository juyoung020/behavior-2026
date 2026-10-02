# 뜬 렌더 기준 자료를 렌더러 입력·비교용으로 바꾸고, 공식 자신의 잡음 폭을 잰다 (GPU 안 씀).
#   powershell -ExecutionPolicy Bypass -File C:\behavior-2026\src\engine\tests\render\capture\process_capture.ps1 -Tag radio_rgbd
# 1) export_render_scene.py : .npy 묶음 + 공식 영상(off_/noise<r>_/ref224_<실행>_)  -> dumps\render_<Tag>\export
# 2) convert_scene.py       : scene.rsc + frame_<k>.rfr (렌더러 입력)               -> dumps\render_<Tag>\rsc
# 3) WSL rsc_check          : 렌더러 불러오기로 읽어 개수·카메라·영상 목록 확인
# 4) WSL render_compare     : off:noise0, noise0:noise1, off:ref224_bkref (공식끼리 = 잡음 폭)  -> export\noise_band.csv
param([string]$Tag = 'radio_rgbd', [int]$TexMax = 1024,
      [string]$Ref224 = 'C:\behavior-2026\outputs\eval_turning_on_radio_20260929_202701_bk_ref,C:\behavior-2026\outputs\eval_turning_on_radio_20260929_203038_bk_waitidle,C:\behavior-2026\outputs\eval_turning_on_radio_20260929_204753_bk_resetuser')
$ErrorActionPreference = 'Stop'
$dump = "C:\behavior-2026\src\engine\dumps\render_$Tag"
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'
. "$(conda info --base)\shell\condabin\conda-hook.ps1"
conda activate behavior
$cap = 'C:\behavior-2026\src\engine\tests\render\capture'
python "$cap\export_render_scene.py" $dump --tex-max 64 --ref224 $Ref224
python "$cap\convert_scene.py" $dump --tex-max $TexMax --out "$dump\rsc"
$w = '/mnt/c/behavior-2026/src/sim/engine/dumps/render_' + $Tag
$sh = @"
set -e
bash /mnt/c/behavior-2026/src/sim/engine/tests/render/compare/build.sh > /dev/null
T=~/engine-build/render-tools
`$T/rsc_check $w/rsc/scene.rsc `$(ls $w/rsc/frame_*.rfr | head -2)
`$T/render_compare $w/export --pairs off:noise0,noise0:noise1,off:ref224_bkref,ref224_bkref:ref224_bkwaitidle --csv $w/export/noise_band.csv | tail -20
"@
$tmp = "C:\behavior-2026\src\engine\dumps\render_$Tag\_process.sh"
[IO.File]::WriteAllText($tmp, ($sh -replace "`r`n", "`n"), [Text.UTF8Encoding]::new($false))
$env:MSYS_NO_PATHCONV = '1'
wsl -d Ubuntu-22.04 -u juyoung -- bash "$w/_process.sh"
