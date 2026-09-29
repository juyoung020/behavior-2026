# 공식 평가기를 기록한 행동열로 재생하면서 물리 층 기록(OVD·볼록 메시·곁기록·거르개 표)을 뜬다.
#   powershell -ExecutionPolicy Bypass -File C:\behavior-2026\src\engine\capture\run_capture.ps1 `
#       -Actions C:\behavior-2026\outputs\eval_turning_on_radio_20260929_195500_nf_a\actions.npz -Tag radio_nfa
# 결과: C:\behavior-2026\src\engine\dumps\<Tag>\  (에셋 파생물 -> git 제외)  + trace.npz 는 같은 폴더(평가기 --output-dir)
# 먼저 nvidia-smi 로 다른 시뮬레이터가 없는지 확인할 것 (GPU 한 장을 여러 작업이 씀).
param([Parameter(Mandatory = $true)][string]$Actions, [string]$Task = 'turning_on_radio', [string]$Instance = '0',
      [int]$MaxSteps = 500, [string]$Tag = 'capture')
$ErrorActionPreference = 'Stop'
$dump = "C:\behavior-2026\src\engine\dumps\$Tag"
New-Item -ItemType Directory -Force $dump | Out-Null
$actionsFull = (Resolve-Path $Actions).Path
function To-Wsl([string]$p) { '/mnt/' + $p.Substring(0, 1).ToLower() + ($p.Substring(2) -replace '\\', '/') }
$srvLog = "$dump\replay_server"
$cmd = "/home/juyoung/openpi/.venv/bin/python /mnt/c/behavior-2026/tools/replay_policy_server.py " +
       "--actions $(To-Wsl $actionsFull) --port 8010 --log $(To-Wsl $dump)/server_log.npz --once"
$srv = Start-Process wsl -ArgumentList "-d Ubuntu-22.04 -u juyoung -- $cmd" -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$srvLog.out.log" -RedirectStandardError "$srvLog.log"
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8010/healthz).StatusCode -eq 200) { $ok = $true; break } } catch {}
    Start-Sleep -Seconds 1
}
if (-not $ok) { "재생 서버가 안 떴다 ($srvLog.log)"; exit 1 }

$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:OMNI_KIT_ACCEPT_EULA = 'YES'; $env:KMP_DUPLICATE_LIB_OK = 'TRUE'
. "$(conda info --base)\shell\condabin\conda-hook.ps1"
conda activate behavior
Set-Location C:\behavior-2026\BEHAVIOR-1K\OmniGibson
$log = "$dump\eval.log"
python C:\behavior-2026\src\engine\capture\physx_capture.py --dump-dir $dump -- --trace -- `
    --task-name $Task --robot-config C:\behavior-2026\src\configs\r1pro_openpi.yaml `
    --env-wrapper omnigibson.eval.wrappers.DefaultWrapper --mode public_test `
    --host 127.0.0.1 --port 8010 --instance-indices $Instance --num-envs 1 --max-steps $MaxSteps `
    --output-dir $dump --write-video --headless 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
$srv.WaitForExit(30000) | Out-Null
python C:\behavior-2026\src\engine\capture\export_sidecar.py $dump
"=== 기록 끝: $dump ==="
