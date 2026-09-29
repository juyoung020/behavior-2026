# 공식 평가기(omnigibson.eval.eval)를 돌린다. 정책 서버(WSL 포트 8000, 또는 -Port 의 재생 서버)가 먼저 떠 있어야 한다.
#   -Task 과제 (기본 turning_on_radio)       -Instances '0,1' 처럼 여러 개 -> --num-envs 자동 (기본 -Instance 0 하나)
#   -MaxSteps 0 = 공식 기본 제한시간(사람 평균 x 1.5). 제출용 결과는 반드시 0(기본)으로.
#   -Wrapper Default(RGB 224, π0.5 기본) | RGBD(공식 RGB-D 720/480)
#   -Gui 시뮬레이터 창.  -Port 정책 서버 포트(재생 서버는 8010).  -Tag 결과 폴더 이름 뒤에 붙일 말
#   -Timing / -Trace : tools\eval_instrumented.py 로 감싸 구간별 시간(timing.json) / 스텝별 기록(trace.npz)을 결과 폴더에.
#                      평가기 코드는 안 바꾼다. (예전 -Profile(cProfile) 은 Kit 이 프로파일 훅을 가져가 쓸모가 없어 없앴다)
param([string]$Task = 'turning_on_radio', [int]$Instance = 0, [string]$Instances = '', [int]$MaxSteps = 0,
      [ValidateSet('Default', 'RGBD')][string]$Wrapper = 'Default', [int]$Port = 8000, [string]$Tag = '',
      [switch]$Gui, [switch]$Timing, [switch]$Trace, [switch]$Deep)
$headless = if ($Gui) { '--no-headless' } else { '--headless' }
$stepArgs = if ($MaxSteps -gt 0) { @('--max-steps', $MaxSteps) } else { @() }
$inst = if ($Instances) { @($Instances -split '[, ]+' | Where-Object { $_ -ne '' }) } else { @("$Instance") }
$wrapperTarget = @{ Default = 'omnigibson.eval.wrappers.DefaultWrapper'; RGBD = 'omnigibson.eval.wrappers.RGBDFullResWrapper' }[$Wrapper]
$Host.UI.RawUI.WindowTitle = "BEHAVIOR eval $Task"
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
# torch/MKL(libiomp5md) 와 conda llvm-openmp(libomp) 가 같이 올라와 장면 로딩 중 'OMP: Error #15' 로 죽는다
# -> 오류 메시지가 안내하는 우회책 (환경은 안 바꾸고 이 실행에만 적용)
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
. "$(conda info --base)\shell\condabin\conda-hook.ps1"
conda activate behavior
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$name = "eval_${Task}_$stamp" + $(if ($Tag) { "_$Tag" } else { '' })
$out = "C:\behavior-2026\outputs\$name"
$log = "C:\behavior-2026\logs\$name.log"
Set-Location C:\behavior-2026\BEHAVIOR-1K\OmniGibson
$runner = @('-m', 'omnigibson.eval.eval')
if ($Timing -or $Trace) {
    $flags = @()
    if ($Timing) { $flags += '--timing' }
    if ($Trace) { $flags += '--trace' }
    if ($Deep) { $flags += '--deep' }
    $runner = @('C:\behavior-2026\tools\eval_instrumented.py') + $flags + @('--')
}
New-Item -ItemType Directory -Force $out | Out-Null
# 로그는 UTF-8 로 (Tee-Object 는 PowerShell 5.1 에서 UTF-16 으로 써서 다른 도구로 읽기 어렵다)
$sw = [System.IO.StreamWriter]::new($log, $true, [Text.UTF8Encoding]::new($false))
$sw.AutoFlush = $true
python @runner `
    --task-name $Task `
    --robot-config C:\behavior-2026\src\configs\r1pro_openpi.yaml `
    --env-wrapper $wrapperTarget `
    --mode public_test `
    --host 127.0.0.1 --port $Port `
    --instance-indices @inst --num-envs $inst.Count @stepArgs `
    --output-dir $out --write-video $headless 2>&1 | ForEach-Object { $l = "$_"; $sw.WriteLine($l); $l }
$code = $LASTEXITCODE
$sw.WriteLine("=== eval exit $code, 결과: $out ===")
$sw.Close()
"=== eval exit $code, 결과: $out ==="
