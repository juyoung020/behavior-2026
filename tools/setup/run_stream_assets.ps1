# BEHAVIOR-1K 에셋(34.7 GB 풀린 크기)을 스트리밍으로 받는다. 끊기면 이 창을 다시 띄우면 이어받는다.
$Host.UI.RawUI.WindowTitle = 'BEHAVIOR assets stream'
$log = "C:\behavior-2026\logs\assets_stream_$(Get-Date -Format yyyyMMdd_HHmm).log"
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Set-Location C:\behavior-2026\tools\setup
& "C:\Users\user one\anaconda3\python.exe" -u stream_assets.py 2>&1 | ForEach-Object { $_; Add-Content -Path $log -Value "$_" -Encoding UTF8 }
Add-Content -Path $log -Value "=== stream exit $LASTEXITCODE ===" -Encoding UTF8
