# Isaac Sim 5.1 + 나머지 setup 단계 (install_isaacsim.py) 를 보이는 창에서 실행하고 로그를 남긴다.
$Host.UI.RawUI.WindowTitle = 'BEHAVIOR Isaac Sim install'
$log = "C:\behavior-2026\logs\isaac_install_$(Get-Date -Format yyyyMMdd_HHmm).log"
$py = "C:\Users\user one\anaconda3\envs\behavior\python.exe"
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [Text.Encoding]::UTF8   # 파이썬 UTF-8 출력을 창·로그에서 그대로 읽는다 (기본 CP949 면 한글이 깨진다)
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
Set-Location C:\behavior-2026\tools\setup

function Tee-Log { process { $_; Add-Content -Path $log -Value "$_" -Encoding UTF8 } }

& $py -u install_isaacsim.py 2>&1 | Tee-Log
$code = $LASTEXITCODE
Add-Content -Path $log -Value "=== install_isaacsim exit $code ===" -Encoding UTF8
if ($code -eq 0) {
    $hook = "$(conda info --base)\shell\condabin\conda-hook.ps1"
    . $hook
    conda install -n behavior av "numpy<2" -c conda-forge -y 2>&1 | Tee-Log
    Add-Content -Path $log -Value "=== conda av exit $LASTEXITCODE ===" -Encoding UTF8
}
Add-Content -Path $log -Value '=== ALL DONE ===' -Encoding UTF8
