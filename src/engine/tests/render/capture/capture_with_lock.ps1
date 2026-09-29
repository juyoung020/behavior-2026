# GPU 잠금을 잡고(기다림) 렌더 기준 자료를 뜬 뒤 바로 푼다. 인자는 run_render_capture.ps1 로 그대로 넘긴다.
#   powershell -ExecutionPolicy Bypass -File C:\behavior-2026\src\engine\tests\render\capture\capture_with_lock.ps1 `
#       -Actions ...\actions.npz -Tag radio_rgbd [-Wrapper RGBD|Default] [-MaxSteps 500]
# 검은 화면 (A)(다른 프로세스가 VRAM 약 5 GiB 넘게 잡으면 RTX 색이 빔)를 피하려고, 잠금을 잡은 뒤에도 GPU 전체 사용량이
# 3.5 GB 아래일 때만 뜬다(윈도 nvidia-smi 는 프로세스별 VRAM 을 N/A 로 보여서 전체 사용량으로 본다. 바탕화면 몫 약 1 GB 포함).
param([Parameter(Mandatory = $true)][string]$Actions, [string]$Tag = 'capture', [ValidateSet('Default', 'RGBD')][string]$Wrapper = 'RGBD',
      [int]$MaxSteps = 500, [string]$Steps = '', [int]$Minutes = 20, [int]$MaxOtherMiB = 3500)
. C:\behavior-2026\tools\gpu_lock.ps1 -Lib
if (-not (Enter-GpuLock 'render' "render 기준 자료 ($Tag, 공식 평가기 RTX $Wrapper)" $Minutes '9' 180)) { exit 2 }
try {
    $ok = $false
    for ($i = 0; $i -lt 20; $i++) {
        $used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1)
        "[capture_with_lock] GPU 사용량 $used MiB (기준 < $MaxOtherMiB)"
        if ($used -lt $MaxOtherMiB) { $ok = $true; break }
        Start-Sleep -Seconds 30
    }
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    if (-not $ok) { "[capture_with_lock] 다른 프로세스 VRAM 이 안 줄어 뜨지 않음"; exit 3 }
    $a = @{ Actions = $Actions; Tag = $Tag; Wrapper = $Wrapper; MaxSteps = $MaxSteps }
    if ($Steps) { $a.Steps = $Steps }
    & C:\behavior-2026\src\engine\tests\render\capture\run_render_capture.ps1 @a
} finally {
    Exit-GpuLock 'render' | Out-Null
    "[capture_with_lock] 잠금 풀림"
}
