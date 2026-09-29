# 대회 기본 제공 체크포인트 (turning_on_radio) 를 Google Drive 에서 받는다.
$Host.UI.RawUI.WindowTitle = 'BEHAVIOR checkpoints'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$log = "C:\behavior-2026\logs\ckpt_download_$(Get-Date -Format yyyyMMdd_HHmm).log"
$dst = 'C:\behavior-2026\checkpoints'
New-Item -ItemType Directory -Force $dst | Out-Null
Set-Location $dst
$items = @(
    @('pi05_turning_on_radio', '1KojwNUz0HVwU3Ww2SVh3NKt-4asuI3y2'),
    @('groot_n17_turning_on_radio', '1OXNm3SPLvWOSJR1e8In6xHMHxOYDp789')
)
foreach ($it in $items) {
    $name = $it[0]; $id = $it[1]
    New-Item -ItemType Directory -Force (Join-Path $dst $name) | Out-Null
    Add-Content -Path $log -Value "=== $name ($id) ===" -Encoding UTF8
    & uvx gdown --continue -O "$dst\$name\" "https://drive.google.com/uc?id=$id" 2>&1 | ForEach-Object { $_; Add-Content -Path $log -Value "$_" -Encoding UTF8 }
    Add-Content -Path $log -Value "=== $name exit $LASTEXITCODE ===" -Encoding UTF8
}
Add-Content -Path $log -Value '=== ALL DONE ===' -Encoding UTF8
