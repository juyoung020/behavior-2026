# 렌더 모듈 기준 자료: 기록해 둔 행동열을 재생하며 공식 평가기(RGB-D 720/480 래퍼)에서 장면 기하·카메라·공식 영상을 뜬다.
#   powershell -ExecutionPolicy Bypass -File C:\behavior-2026\src\engine\tests\render\capture\run_render_capture.ps1 `
#       -Actions C:\behavior-2026\outputs\eval_turning_on_radio_20260929_195500_nf_a\actions.npz -Tag radio_rgbd
# 결과: C:\behavior-2026\src\engine\dumps\render_<Tag>\  (에셋 파생물 -> git 제외) + trace.npz(평가기 --output-dir 도 같은 폴더)
# GPU 잠금(C:\behavior-2026\.gpu_lock)은 부르는 쪽이 잡는다. 다른 프로세스가 VRAM 5 GiB 넘게 잡으면 RTX 색이 빈다(plan.md 3절 (A)).
param([Parameter(Mandatory = $true)][string]$Actions, [string]$Task = 'turning_on_radio', [string]$Instance = '0',
      [int]$MaxSteps = 500, [string]$Tag = 'capture', [ValidateSet('Default', 'RGBD')][string]$Wrapper = 'RGBD',
      [string]$Steps = '', [int]$NoiseRenders = 2)
$ErrorActionPreference = 'Stop'
$dump = "C:\behavior-2026\src\engine\dumps\render_$Tag"
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

$wrapperTarget = @{ Default = 'omnigibson.eval.wrappers.DefaultWrapper'; RGBD = 'omnigibson.eval.wrappers.RGBDFullResWrapper' }[$Wrapper]
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:OMNI_KIT_ACCEPT_EULA = 'YES'; $env:KMP_DUPLICATE_LIB_OK = 'TRUE'
. "$(conda info --base)\shell\condabin\conda-hook.ps1"
conda activate behavior
Set-Location C:\behavior-2026\BEHAVIOR-1K\OmniGibson
$log = "$dump\eval.log"
$ours = @('--dump-dir', $dump, '--noise-renders', "$NoiseRenders")
if ($Steps) { $ours += @('--steps', $Steps) }
$ErrorActionPreference = 'Continue'
python C:\behavior-2026\src\engine\tests\render\capture\render_capture.py @ours -- --trace -- `
    --task-name $Task --robot-config C:\behavior-2026\src\configs\r1pro_openpi.yaml `
    --env-wrapper $wrapperTarget --mode public_test `
    --host 127.0.0.1 --port 8010 --instance-indices $Instance --num-envs 1 --max-steps $MaxSteps `
    --output-dir $dump --write-video --headless 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Encoding utf8
$srv.WaitForExit(30000) | Out-Null
# 평가기가 연결 전에 죽으면 --once 재생 서버가 8010 에서 계속 기다린다 -> 이 실행의 서버만 끈다(로그 경로로 가림)
wsl -d Ubuntu-22.04 -u juyoung -- pkill -f "replay_policy_server.py.*render_$Tag/server_log" 2>$null
"=== 렌더 기준 자료 끝: $dump ==="
