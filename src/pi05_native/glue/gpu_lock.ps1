# Shared GPU lock from Windows (same rule/format as src\engine\scripts\gpu_lock.sh): creating the directory is the lock.
#   . C:\behavior-2026\src\pi05_native\glue\gpu_lock.ps1 ; Get-GpuLock "pi05_native" "purpose" 20 6 ; ... ; Remove-GpuLock
# owner.txt keys: owner= purpose= start_epoch= end_epoch= vram_gb=   stale only when now > end_epoch + 900
$script:GpuLockDir = 'C:\behavior-2026\.gpu_lock'
function Get-GpuLock([string]$Owner, [string]$Purpose, [int]$Minutes = 20, [double]$VramGb = 6) {
    if ($Minutes -gt 30) { $Minutes = 30 }
    while ($true) {
        try {
            New-Item -ItemType Directory -Path $script:GpuLockDir -ErrorAction Stop | Out-Null
            $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
            "owner=$Owner`npurpose=$Purpose`nstart_epoch=$now`nend_epoch=$($now + 60 * $Minutes)`nvram_gb=$VramGb`n" |
                Set-Content -Path "$script:GpuLockDir\owner.txt" -Encoding ascii -NoNewline
            Write-Host "[gpu_lock] acquired by $Owner ($Purpose)"
            return
        } catch {
            $txt = Get-Content "$script:GpuLockDir\owner.txt" -Raw -ErrorAction SilentlyContinue
            $end = if ($txt -match 'end_epoch=(\d+)') { [int64]$Matches[1] } else { $null }
            if ($end -and [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() -gt $end + 900) {
                Write-Host "[gpu_lock] removing stale lock: $txt"
                Remove-Item -Recurse -Force $script:GpuLockDir -ErrorAction SilentlyContinue
                continue
            }
            Write-Host "[gpu_lock] busy: $($txt -replace "`n", ' ') -- retry in 45 s"
            Start-Sleep -Seconds 45
        }
    }
}
function Remove-GpuLock { Remove-Item -Recurse -Force $script:GpuLockDir -ErrorAction SilentlyContinue; Write-Host "[gpu_lock] released" }
