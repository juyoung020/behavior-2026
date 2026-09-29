# 기록해 둔 행동열(actions.npz)을 재생 서버로 먹이며 공식 평가기를 돌린다 -- 일치 검증·시뮬레이터 시간 측정용.
#   재생 서버: tools\replay_policy_server.py (WSL openpi venv, 포트 8010, 받은 관측을 <결과폴더>\server_log.npz 로)
#   평가기:   tools\run_eval_radio.ps1 -Port 8010 (공식 명령 그대로, -Timing/-Trace/-Deep 는 그대로 넘김)
# 예) powershell -ExecutionPolicy Bypass -File C:\behavior-2026\tools\run_replay_eval.ps1 `
#       -Actions C:\behavior-2026\outputs\<기준 실행>\actions.npz -Trace -Tag nf_a
#   -Perturb 'STEP:DIM:DELTA'  음성 대조용으로 한 스텝 행동을 조금 바꾼다
#   -WindowsServer             재생 서버를 WSL 대신 Windows(conda behavior)에서
param([Parameter(Mandatory = $true)][string]$Actions, [string]$Task = 'turning_on_radio', [string]$Instances = '0',
      [int]$MaxSteps = 500, [ValidateSet('Default', 'RGBD')][string]$Wrapper = 'Default', [string]$Tag = 'replay',
      [string]$Perturb = '', [switch]$Timing, [switch]$Trace, [switch]$Deep, [switch]$WindowsServer)
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$out = "C:\behavior-2026\outputs\eval_${Task}_${stamp}_$Tag"
New-Item -ItemType Directory -Force $out | Out-Null
$actionsFull = (Resolve-Path $Actions).Path
function To-Wsl([string]$p) { '/mnt/' + $p.Substring(0, 1).ToLower() + ($p.Substring(2) -replace '\\', '/') }
$srvLog = "C:\behavior-2026\logs\replay_server_${stamp}_$Tag"
if ($WindowsServer) {
    . "$(conda info --base)\shell\condabin\conda-hook.ps1"
    conda activate behavior
    $a = @('C:\behavior-2026\tools\replay_policy_server.py', '--actions', $actionsFull, '--port', '8010',
           '--log', "$out\server_log.npz", '--once')
    if ($Perturb) { $a += @('--perturb', $Perturb) }
    $srv = Start-Process python -ArgumentList $a -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$srvLog.out.log" -RedirectStandardError "$srvLog.log"
} else {
    $cmd = "/home/juyoung/openpi/.venv/bin/python /mnt/c/behavior-2026/tools/replay_policy_server.py " +
           "--actions $(To-Wsl $actionsFull) --port 8010 --log $(To-Wsl $out)/server_log.npz --once"
    if ($Perturb) { $cmd += " --perturb $Perturb" }
    $srv = Start-Process wsl -ArgumentList "-d Ubuntu-22.04 -u juyoung -- $cmd" `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput "$srvLog.out.log" -RedirectStandardError "$srvLog.log"
}
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8010/healthz).StatusCode -eq 200) { $ok = $true; break } } catch {}
    Start-Sleep -Seconds 1
}
if (-not $ok) { "재생 서버가 안 떴다 ($srvLog.log)"; exit 1 }
$ev = @{ Task = $Task; Instances = $Instances; MaxSteps = $MaxSteps; Wrapper = $Wrapper; Port = 8010; OutDir = $out }
if ($Timing) { $ev.Timing = $true }
if ($Trace) { $ev.Trace = $true }
if ($Deep) { $ev.Deep = $true }
& C:\behavior-2026\tools\run_eval_radio.ps1 @ev
$srv.WaitForExit(30000) | Out-Null
"=== 재생 평가 끝: $out ==="
