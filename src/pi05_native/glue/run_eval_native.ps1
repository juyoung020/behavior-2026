# Official evaluator with the native pi0.5 engine inside the simulator process (no policy server, no WSL).
#   run_eval_native.ps1 [-Task turning_on_radio] [-Instance 0] [-MaxSteps 0] [-Weights ...] [-Replan 16] [-Video] [-Gui] [-Tag x]
# Same environment setup as tools\run_eval_radio.ps1 (robot config r1pro_openpi.yaml = robot name "robot", 224 RGB).
# Writes results to outputs\eval_native_<task>_<stamp>\ and the per-step engine log native_steps.csv there.
# For experiment matrices use tools\exp_run.ps1 with policy "native", module "pi05_policy:make_policy" and
# PYTHONPATH=<this folder>;<build_win> (see docs\π05_네이티브엔진.md).
param([string]$Task = 'turning_on_radio', [int]$Instance = 0, [int]$MaxSteps = 0,
      [string]$Weights = 'C:\behavior-2026\data\pi05_native\pi05_radio.pi05w', [int]$Replan = 16,
      [switch]$Video, [switch]$Gui, [string]$Tag = '', [switch]$NoLock,
      [string]$RobotConfig = 'C:\behavior-2026\src\configs\r1pro_openpi.yaml')
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'   # same workaround as run_eval_radio.ps1 (torch MKL + conda llvm-openmp)
. "$env:USERPROFILE\anaconda3\shell\condabin\conda-hook.ps1"  # user anaconda3 holds the behavior env (another conda can come first on PATH, so not conda info --base)
conda activate behavior
. C:\behavior-2026\tools\gpu_lock.ps1 -Lib   # shared lock (owner.txt: owner= purpose= start_epoch= end_epoch= vram_gb=)
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$name = "eval_native_${Task}_$stamp" + $(if ($Tag) { "_$Tag" } else { '' })
$out = "C:\behavior-2026\outputs\$name"
New-Item -ItemType Directory -Force $out | Out-Null
$log = "C:\behavior-2026\logs\$name.log"
$evalArgs = @('--task-name', $Task, '--mode', 'public_test', '--instance-indices', "$Instance", '--num-envs', '1',
              '--output-dir', $out, '--env-wrapper', 'omnigibson.eval.wrappers.DefaultWrapper', '--robot-config', $RobotConfig,
              $(if ($Gui) { '--no-headless' } else { '--headless' }))
if ($MaxSteps -gt 0) { $evalArgs += @('--max-steps', "$MaxSteps") }
if ($Video) { $evalArgs += '--write-video' }
if (-not $NoLock) {
    # lock, then require no other simulator and < 3.5 GB of other-process VRAM (black frames (A)); else release and retry
    while ($true) {
        if (-not (Enter-GpuLock 'pi05_native' "evaluator + native pi0.5 in-process ($Task)" 30 '12')) { throw 'no GPU lock' }
        $sims = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                  Where-Object { $_.CommandLine -match 'omnigibson|og_black_repro|isaac_black_repro' })
        $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
        if ($sims.Count -eq 0 -and $used -lt 3500) { "GPU used by other processes before start: $used MiB"; break }
        [void](Exit-GpuLock 'pi05_native')
        "waiting: other simulators $($sims.Count), GPU used $used MiB"
        Start-Sleep -Seconds 60
    }
}
try {
    Set-Location C:\behavior-2026\BEHAVIOR-1K\OmniGibson
    python "$PSScriptRoot\run_eval_native.py" --weights $Weights --replan $Replan --native-log "$out\native_steps.csv" -- @evalArgs 2>&1 |
        Tee-Object -FilePath $log
} finally {
    if (-not $NoLock) { [void](Exit-GpuLock 'pi05_native') }
}
"results: $out   log: $log"
