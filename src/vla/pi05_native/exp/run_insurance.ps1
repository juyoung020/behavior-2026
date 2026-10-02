# Native pi0.5 experiments through the shared experiment runner (tools\exp_run.ps1: GPU lock + queue per run,
# black-frame detection, result table). The evaluator itself is unmodified; our policy goes in with --native-policy.
#   insurance (2025 1st place, 4 checkpoints, tasks 0-49 by 2025 score):
#     powershell -File src\pi05_native\exp\run_insurance.ps1
#   one full radio episode with the openpi radio checkpoint:
#     powershell -File src\pi05_native\exp\run_insurance.ps1 -Matrix src\pi05_native\exp\radio_native_full.json -Model radio
param([string]$Matrix = "$PSScriptRoot\pb2025_insurance.json", [string]$Build = 'build_win',
      [ValidateSet('pb2025', 'radio')][string]$Model = 'pb2025', [switch]$Reuse, [switch]$DryRun)
$glue = Join-Path $PSScriptRoot '..\glue' | Resolve-Path
$bin = Join-Path $PSScriptRoot "..\$Build" | Resolve-Path
$env:PYTHONPATH = "$glue;$bin" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { '' })
$env:PI05_MODEL = $Model
"PYTHONPATH=$env:PYTHONPATH  PI05_MODEL=$env:PI05_MODEL"
$a = @('-ExecutionPolicy', 'Bypass', '-File', 'C:\behavior-2026\tools\exp_run.ps1', 'run', '-Matrix', $Matrix, '-Backend', 'original')
if ($Reuse) { $a += '-Reuse' }
if ($DryRun) { $a += '-DryRun' }
& powershell @a
