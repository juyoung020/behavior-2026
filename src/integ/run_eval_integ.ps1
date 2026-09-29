# 종단 한 판: 공식 평가기(Windows, 네이티브 π0.5 프로세스 안) + 계획기(simlink, WSL) + meridian(WSL, 실시간 그래프).
#   powershell -ExecutionPolicy Bypass -File C:\behavior-2026\src\integ\run_eval_integ.ps1 [-Task turning_on_radio] [-Instance 0]
#       [-MaxSteps 600] [-Llm kau|oracle|none] [-Meridian 1|0] [-Wrapper rgbd|default] [-SimlinkArgs '--stage vote'] [-Video] [-Tag x]
# 순서: GPU 잠금(선착순) → WSL 묶음(meridian launch → frontend 준비 → VRAM 기록 → simlink) → 평가기 → 정리 → 요약.
# 결과: C:\behavior-2026\outputs\integ\<이름>\ (평가 JSON·영상, native_steps.csv, glue_steps.csv, decisions.jsonl,
#       trace\trace.jsonl(계획기), meridian 기록, VRAM 기록, summary.json)
param([string]$Task = 'turning_on_radio', [int]$Instance = 0, [int]$MaxSteps = 600,
      [string]$Weights = 'C:\behavior-2026\data\pi05_native\pi05_radio.pi05w', [int]$Replan = 16,
      [string]$Llm = 'kau', [int]$Meridian = 1, [string]$Wrapper = 'rgbd', [string]$SimlinkArgs = '',
      [switch]$Video, [string]$Tag = '', [switch]$NoLock, [int]$VramWarnMiB = 3500, [int]$LockMinutes = 30, [int]$MaxWaitMin = 360,
      [string]$RobotConfig = 'C:\behavior-2026\src\configs\r1pro_openpi.yaml')
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'   # run_eval_radio.ps1 과 같은 우회(torch MKL + conda llvm-openmp)
. "$(conda info --base)\shell\condabin\conda-hook.ps1"
conda activate behavior
. C:\behavior-2026\tools\gpu_lock.ps1 -Lib
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$name = "integ_${Task}_$stamp" + $(if ($Tag) { "_$Tag" } else { '' })
$out = "C:\behavior-2026\outputs\integ\$name"
$wout = "/mnt/c/behavior-2026/outputs/integ/$name"
New-Item -ItemType Directory -Force $out | Out-Null
$log = "$out\eval.log"
function Gpu-Used { [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1) }

if (-not $NoLock) {
    while ($true) {
        if (-not (Enter-GpuLock 'integ' "통합 한 판: 평가기+네이티브 pi0.5+meridian ($Task, $MaxSteps 스텝)" $LockMinutes '14' $MaxWaitMin)) { throw 'no GPU lock' }
        $sims = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                  Where-Object { $_.CommandLine -match 'omnigibson|og_black_repro|isaac_black_repro' })
        $used = Gpu-Used
        if ($sims.Count -eq 0 -and $used -lt $VramWarnMiB) { "다른 프로세스 GPU 사용(시작 전): $used MiB"; break }
        [void](Exit-GpuLock 'integ')
        "기다림: 다른 시뮬레이터 $($sims.Count), GPU $used MiB"
        Start-Sleep -Seconds 60
    }
}
$sampler = $null; $wsl = $null
try {
    "gpu_used_mib_windows_before=$(Gpu-Used)" | Out-File -Encoding utf8 "$out\vram_windows.txt"
    # ---- WSL 묶음 ----
    $wargs = @('-d', 'Ubuntu-22.04', '-u', 'juyoung', '--', 'bash', '/mnt/c/behavior-2026/src/integ/wsl_stack.sh', $wout, "$Meridian", $Llm,
               '--task', $Task, '--max-steps', "$MaxSteps")
    if ($SimlinkArgs) { $wargs += ($SimlinkArgs -split ' ' | Where-Object { $_ }) }
    $env:MSYS_NO_PATHCONV = '1'
    $wsl = Start-Process wsl.exe -ArgumentList $wargs -PassThru -NoNewWindow -RedirectStandardOutput "$out\wsl_stdout.log" -RedirectStandardError "$out\wsl_stderr.log"
    $t0 = Get-Date
    while (-not (Test-Path "$out\wsl_ready")) {
        if ($wsl.HasExited) { throw "WSL 묶음이 먼저 끝남 — $out\wsl_stack.log" }
        if (((Get-Date) - $t0).TotalSeconds -gt 420) { throw 'WSL 묶음 준비 시간 초과(420 s)' }
        Start-Sleep -Seconds 2
    }
    $usedReady = Gpu-Used
    "gpu_used_mib_windows_meridian_ready=$usedReady" | Out-File -Append -Encoding utf8 "$out\vram_windows.txt"
    "WSL 준비 ($([int]((Get-Date) - $t0).TotalSeconds) s), 평가기 시작 전 GPU 사용 $usedReady MiB"
    if ($usedReady -ge $VramWarnMiB) { "경고: 평가기 밖 GPU 사용 $usedReady MiB >= $VramWarnMiB — 검은 화면 (A) 위험(plan 3절)" }
    $sampler = Start-Process powershell -ArgumentList @('-ExecutionPolicy', 'Bypass', '-File', 'C:\behavior-2026\tools\gpu_mem_sampler.ps1',
                                                          '-Out', "$out\gpu_mem.csv", '-Seconds', '1800', '-Every', '2') -PassThru -WindowStyle Hidden
    # ---- 평가기 ----
    $evalArgs = @('--task-name', $Task, '--mode', 'public_test', '--instance-indices', "$Instance", '--num-envs', '1',
                  '--output-dir', "$out\eval", '--robot-config', $RobotConfig, '--headless')
    if ($MaxSteps -gt 0) { $evalArgs += @('--max-steps', "$MaxSteps") }
    if ($Video) { $evalArgs += '--write-video' }
    $pyArgs = @('--weights', $Weights, '--replan', "$Replan", '--out', $out, '--wrapper', $Wrapper, '--robot-config', $RobotConfig)
    if ($Llm -eq 'none' -and $Meridian -eq 0) { $pyArgs += '--no-link' }
    Set-Location C:\behavior-2026\BEHAVIOR-1K\OmniGibson
    $te = Get-Date
    python C:\behavior-2026\src\integ\glue\run_eval_integ.py @pyArgs -- @evalArgs 2>&1 | Tee-Object -FilePath $log
    "평가기 끝: $([int]((Get-Date) - $te).TotalSeconds) s"
    $t1 = Get-Date
    while (-not (Test-Path "$out\wsl_done") -and -not $wsl.HasExited) {
        if (((Get-Date) - $t1).TotalSeconds -gt 180) { 'WSL 묶음 정리 시간 초과 — 계속'; break }
        Start-Sleep -Seconds 2
    }
} finally {
    if ($sampler -and -not $sampler.HasExited) { Stop-Process -Id $sampler.Id -Force -ErrorAction SilentlyContinue }
    if ($wsl -and -not $wsl.HasExited) {
        # 평가기가 비정상으로 끝났으면 WSL 쪽 정리(simlink 가 연결 끝을 못 봤을 수 있음)
        & wsl.exe -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/src/integ/wsl_cleanup.sh $wout | Out-Null
    }
    if (-not $NoLock) { [void](Exit-GpuLock 'integ') }
}
python C:\behavior-2026\src\integ\summarize_run.py $out 2>&1 | Tee-Object -FilePath "$out\summary.txt"
"결과: $out"
