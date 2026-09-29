# 공식 평가기(omnigibson.eval.eval)를 돌린다. 정책 서버(WSL 포트 8000, 또는 -Port 의 재생 서버)가 먼저 떠 있어야 한다.
#   -Task 과제 (기본 turning_on_radio)       -Instances '0,1' 처럼 여러 개 -> --num-envs 자동 (기본 -Instance 0 하나)
#   -MaxSteps 0 = 공식 기본 제한시간(사람 평균 x 1.5). 제출용 결과는 반드시 0(기본)으로.
#   -Wrapper Default(RGB 224, π0.5 기본) | RGBD(공식 RGB-D 720/480)
#   -Gui 시뮬레이터 창.  -Port 정책 서버 포트(재생 서버는 8010).  -Tag 결과 폴더 이름 뒤에 붙일 말, -OutDir 결과 폴더 직접 지정
#   -Timing / -Trace : tools\eval_instrumented.py 로 감싸 구간별 시간(timing.json) / 스텝별 기록(trace.npz)을 결과 폴더에.
#   -ChunkSize 16    : 공식 인자 --replay-action-chunk-size (서버가 action_chunk 를 줘야 함: src\fasteval\pi05_chunk_server.py)
#   -BlackGuard warn|abort : 정책에 들어갈 카메라 영상이 전부 0(검은 화면)이면 경고/중단.  -BlackDiag : 호스트/GPU 경로 비교(진단)
#   -KitSet '/키=값' : 진단 실험용 Kit 설정 바꾸기 (공식 결과에는 쓰지 않는다)
#   -RobotConfig 경로 : 로봇 설정 (기본 src\configs\r1pro_openpi.yaml = 로봇 이름 robot, openpi 베이스라인용).
#                      2위 Comet 서버는 robot_r1 키를 쓰므로 공식 BEHAVIOR-1K\OmniGibson\omnigibson\eval\r1pro.yaml 을 준다.
#   -Deep            : 컨트롤러 콜백·물체 상태 캐시 안쪽까지 잘게 (-Timing 과 같이)
#                      평가기 코드는 안 바꾼다. (예전 -Profile(cProfile) 은 Kit 이 프로파일 훅을 가져가 쓸모가 없어 없앴다)
param([string]$Task = 'turning_on_radio', [int]$Instance = 0, [string]$Instances = '', [int]$MaxSteps = 0,
      [ValidateSet('Default', 'RGBD')][string]$Wrapper = 'Default', [int]$Port = 8000, [string]$Tag = '', [string]$OutDir = '',
      [int]$ChunkSize = 0, [switch]$Gui, [switch]$Timing, [switch]$Trace, [switch]$Deep,
      [switch]$BlackDiag, [ValidateSet('', 'warn', 'abort')][string]$BlackGuard = '', [string[]]$KitSet = @(),
      [string]$RobotConfig = 'C:\behavior-2026\src\configs\r1pro_openpi.yaml',
      [ValidateSet('websocket', 'local')][string]$Policy = 'websocket', [int]$RenderIters = 0, [string[]]$KitArg = @(),
      [switch]$VkNvidiaOnly)
# -VkNvidiaOnly     : 진단용 -- 이 실행 프로세스에서만 Vulkan 이 NVIDIA 드라이버만 보게 한다(AMD 내장 GPU 숨김). 시스템 설정은 안 바꾼다
# -RobotConfig none : --robot-config 를 아예 안 넘김(평가기 기본 = 공식 eval\r1pro.yaml, 수정 0 재현용)
# -Policy local     : 공식 평가기의 0 행동 정책(서버 없이, 평가기 점검용 공식 옵션)
# -KitArg '--/app/vulkan=false' : 진단용 Kit 시작 인자
# -RenderIters N    : 진단용 -- 스텝마다 렌더를 N 번(공식은 1, eval/evaluator.py:383). 물리·판정은 안 바뀌어야 한다
$headless = if ($Gui) { '--no-headless' } else { '--headless' }
$stepArgs = if ($MaxSteps -gt 0) { @('--max-steps', $MaxSteps) } else { @() }
if ($ChunkSize -gt 1) { $stepArgs += @('--replay-action-chunk-size', $ChunkSize) }
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
if ($VkNvidiaOnly) {
    # 레지스트리 VulkanDriverName 에서 RTX 5070 Ti 의 ICD 를 찾아 이 프로세스 환경변수로만 지정 (Vulkan 로더 VK_DRIVER_FILES / 옛 이름 VK_ICD_FILENAMES)
    $icd = Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' -ErrorAction SilentlyContinue |
        ForEach-Object { Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue } |
        Where-Object { $_.DriverDesc -like '*RTX 5070 Ti*' } | Select-Object -First 1 -ExpandProperty VulkanDriverName
    $env:VK_DRIVER_FILES = "$icd"; $env:VK_ICD_FILENAMES = "$icd"
    "Vulkan ICD (이 실행만): $icd"
}
. "$(conda info --base)\shell\condabin\conda-hook.ps1"
conda activate behavior
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$name = "eval_${Task}_$stamp" + $(if ($Tag) { "_$Tag" } else { '' })
if ($OutDir) { $name = Split-Path $OutDir -Leaf }
$out = if ($OutDir) { $OutDir } else { "C:\behavior-2026\outputs\$name" }
$log = "C:\behavior-2026\logs\$name.log"
Set-Location C:\behavior-2026\BEHAVIOR-1K\OmniGibson
$runner = @('-m', 'omnigibson.eval.eval')
if ($Timing -or $Trace -or $BlackDiag -or $BlackGuard -or $KitSet.Count -or $RenderIters -or $KitArg.Count) {
    $flags = @()
    if ($Timing) { $flags += '--timing' }
    if ($Trace) { $flags += '--trace' }
    if ($Deep) { $flags += '--deep' }
    if ($BlackDiag) { $flags += '--black-diag' }
    if ($BlackGuard) { $flags += "--black-guard=$BlackGuard" }
    foreach ($kv in $KitSet) { $flags += "--set=$kv" }
    if ($RenderIters) { $flags += "--render-iters=$RenderIters" }
    foreach ($ka in $KitArg) { $flags += "--kit-arg=$ka" }
    $runner = @('C:\behavior-2026\tools\eval_instrumented.py') + $flags + @('--')
}
New-Item -ItemType Directory -Force $out | Out-Null
# 로그는 UTF-8 로 (Tee-Object 는 PowerShell 5.1 에서 UTF-16 으로 써서 다른 도구로 읽기 어렵다)
$sw = [System.IO.StreamWriter]::new($log, $true, [Text.UTF8Encoding]::new($false))
$sw.AutoFlush = $true
$robotArgs = if ($RobotConfig -eq 'none') { @() } else { @('--robot-config', $RobotConfig) }
python @runner `
    --task-name $Task @robotArgs --policy $Policy `
    --env-wrapper $wrapperTarget `
    --mode public_test `
    --host 127.0.0.1 --port $Port `
    --instance-indices @inst --num-envs $inst.Count @stepArgs `
    --output-dir $out --write-video $headless 2>&1 | ForEach-Object { $l = "$_"; $sw.WriteLine($l); $l }
$code = $LASTEXITCODE
$sw.WriteLine("=== eval exit $code, 결과: $out ===")
$sw.Close()
"=== eval exit $code, 결과: $out ==="
