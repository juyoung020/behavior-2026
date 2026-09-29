# 공용 GPU 잠금 (전원 공통 규칙, Windows 쪽): C:\behavior-2026\.gpu_lock 디렉터리를 만드는 것이 곧 잠금.
# owner.txt 형식(전원 공통): owner= purpose= start_epoch= end_epoch=(유닉스 초) vram_gb=
# 오래됨 판정은 now_epoch > end_epoch + 900 하나뿐(시간 문자열 계산 안 함). end_epoch 를 못 읽으면 지우지 않는다.
#   명령:  powershell -File C:\behavior-2026\tools\gpu_lock.ps1 acquire -Owner fasteval -Purpose '검은 화면 (B) 3 판' -Minutes 10 -VramGb 6
#          powershell -File C:\behavior-2026\tools\gpu_lock.ps1 release -Owner fasteval     (owner 가 같을 때만 푼다)
#          powershell -File C:\behavior-2026\tools\gpu_lock.ps1 status
#   함수:  . C:\behavior-2026\tools\gpu_lock.ps1 -Lib ; Enter-GpuLock <owner> <purpose> <minutes> <vram_gb> ; Exit-GpuLock <owner>
# - 이미 있으면 45 초마다 다시 시도. 한 번에 30 분 이내(넘게 주면 30 으로 자름).
# - 잡은 뒤에도 VRAM 에 민감한 일은 nvidia-smi 로 다른 프로세스가 없는지 따로 확인한다.
param([Parameter(Position = 0)][ValidateSet('acquire', 'release', 'status', '')][string]$Action = '',
      [string]$Owner = 'fasteval', [string]$Purpose = '', [int]$Minutes = 30, [string]$VramGb = '?', [int]$MaxWaitMin = 120, [switch]$Lib)
$script:GpuLockDir = 'C:\behavior-2026\.gpu_lock'

function Get-NowEpoch { [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() }

function Get-GpuLockOwner {
    $f = Join-Path $script:GpuLockDir 'owner.txt'
    if (-not (Test-Path $f)) { return $null }
    $h = @{}
    try { $lines = [IO.File]::ReadAllLines($f) } catch { return $null }
    foreach ($l in $lines) { if ($l -match '^\s*([A-Za-z_]+)\s*=\s*(.*)$') { $h[$Matches[1]] = $Matches[2].Trim() } }
    return $h
}

function Test-GpuLockStale {
    $o = Get-GpuLockOwner
    if ($null -eq $o -or -not $o.ContainsKey('end_epoch')) { return $false }
    $end = 0L
    if (-not [long]::TryParse($o['end_epoch'], [ref]$end)) { return $false }
    return ((Get-NowEpoch) -gt ($end + 900))
}

function Enter-GpuLock([string]$Owner, [string]$Purpose, [int]$Minutes = 30, [string]$VramGb = '?', [int]$MaxWaitMin = 120) {
    if ($Minutes -gt 30) { $Minutes = 30 }
    $t0 = Get-NowEpoch
    while ($true) {
        # cmd mkdir = CreateDirectoryW 한 번 -- 이미 있으면 실패(원자적)
        cmd /c mkdir "$script:GpuLockDir" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $s = Get-NowEpoch
            $txt = "owner=$Owner`npurpose=$Purpose`nstart_epoch=$s`nend_epoch=$($s + 60 * $Minutes)`nvram_gb=$VramGb`n"
            [IO.File]::WriteAllText((Join-Path $script:GpuLockDir 'owner.txt'), $txt, [Text.UTF8Encoding]::new($false))
            Write-Host "[gpu_lock] 잡음: $Owner / $Purpose ($Minutes 분)"
            return $true
        }
        if (Test-GpuLockStale) {
            $o = Get-GpuLockOwner
            Write-Host "[gpu_lock] 오래된 잠금 지움(end_epoch + 900 초 지남): $(($o.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' ')"
            Remove-Item $script:GpuLockDir -Recurse -Force -ErrorAction SilentlyContinue
            continue
        }
        if (((Get-NowEpoch) - $t0) -gt 60 * $MaxWaitMin) { Write-Host "[gpu_lock] $MaxWaitMin 분 기다려도 못 잡음"; return $false }
        $o = Get-GpuLockOwner
        $left = if ($o -and $o['end_epoch']) { "끝 예정까지 $([int](([long]$o['end_epoch'] - (Get-NowEpoch)) / 60)) 분" } else { '' }
        Write-Host "[gpu_lock] 사용 중: $(if ($o) { "$($o['owner']) / $($o['purpose']) / $left" } else { '(owner.txt 없음)' }) -- 45 초 뒤 다시"
        Start-Sleep -Seconds 45
    }
}

function Exit-GpuLock([string]$Owner) {
    # 내 잠금만 푼다(owner 가 같을 때). owner.txt 가 없으면(형식이 다른 남의 잠금일 수 있음) 건드리지 않는다.
    $o = Get-GpuLockOwner
    if ($null -eq $o -or $o['owner'] -ne $Owner) {
        if (Test-Path $script:GpuLockDir) { Write-Host "[gpu_lock] 내 잠금이 아니라 안 푼다: $(if ($o) { $o['owner'] } else { '?' })" }
        return $false
    }
    Remove-Item $script:GpuLockDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "[gpu_lock] 풂: $Owner"
    return $true
}

if (-not $Lib) {
    switch ($Action) {
        'acquire' { if (Enter-GpuLock $Owner $Purpose $Minutes $VramGb $MaxWaitMin) { exit 0 } else { exit 1 } }
        'release' { if (Exit-GpuLock $Owner) { exit 0 } else { exit 1 } }
        'status' {
            $o = Get-GpuLockOwner
            if ($o) { $o.GetEnumerator() | Sort-Object Key | ForEach-Object { "$($_.Key)=$($_.Value)" }; "now_epoch=$(Get-NowEpoch)" }
            elseif (Test-Path $script:GpuLockDir) { '잠김(owner.txt 없음 또는 형식 다름)' } else { 'free' }
        }
    }
}
