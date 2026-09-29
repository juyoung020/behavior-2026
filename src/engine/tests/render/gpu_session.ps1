# render B GPU 시험 한 판: 공용 대기열(tools\gpu_lock.ps1)에 줄 서서 잡고 -> WSL 에서 시험 스크립트 -> 바로 풂.
#   powershell -ExecutionPolicy Bypass -File C:\behavior-2026\src\engine\tests\render\gpu_session.ps1 [-Script <WSL 경로>] [-Minutes 5]
# 기본 스크립트: /mnt/c/behavior-2026/src/engine/tests/render/gpu_tests.sh (합성 층1=층2 + 뜬 장면 있으면 장면 층1=층2·처리량)
param([string]$Script = '/mnt/c/behavior-2026/src/engine/tests/render/gpu_tests.sh', [int]$Minutes = 5,
      [string]$Log = 'C:\behavior-2026\logs\render_gpu_session.log')
. C:\behavior-2026\tools\gpu_lock.ps1 -Lib
if (-not (Enter-GpuLock 'render-B' 'render B: 층1=층2 비트 시험·처리량 (수 분)' $Minutes '3' 240)) { exit 2 }
try {
    & wsl -d Ubuntu-22.04 -u juyoung -- bash $Script *>&1 | ForEach-Object { "$_" } | Out-File -FilePath $Log -Encoding utf8
} finally {
    Exit-GpuLock 'render-B' | Out-Null
    "[gpu_session] 잠금 풀림" | Out-File -FilePath $Log -Append -Encoding utf8
}
