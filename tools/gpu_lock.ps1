# 공용 GPU 잠금 (전원 공통 규칙, Windows 쪽): C:\behavior-2026\.gpu_lock 디렉터리를 만드는 것이 곧 잠금. 선착순 대기열 C:\behavior-2026\.gpu_queue\.
# owner.txt 형식(전원 공통): owner= purpose= start_epoch= end_epoch=(유닉스 초) vram_gb=
# 오래됨 판정은 epoch 초 계산만: now_epoch > end_epoch + 900. end_epoch 를 못 읽으면 지우지 않는다.
#   명령:  powershell -File C:\behavior-2026\tools\gpu_lock.ps1 acquire -Owner fasteval -Purpose '검은 화면 (B) 3 판' -Minutes 10 -VramGb 6
#          powershell -File C:\behavior-2026\tools\gpu_lock.ps1 release -Owner fasteval     (owner 가 같을 때만 푼다)
#          powershell -File C:\behavior-2026\tools\gpu_lock.ps1 status                      (잠금 + 대기열)
#   함수:  . C:\behavior-2026\tools\gpu_lock.ps1 -Lib ; Enter-GpuLock <owner> <purpose> <minutes> <vram_gb> ; Exit-GpuLock <owner>
# 대기열(선착순)
# - 줄 서기: .gpu_queue\<start_epoch_ns>_<owner> 표 파일(내용 owner= purpose= host=windows pid= start_epoch= end_epoch= minutes= vram_gb=).
#   기다리는 동안 45 초마다 표의 end_epoch 를 now + 60 으로 새로 적는다(살아 있다는 표시).
# - 내 표가 이름 앞 숫자 순으로 가장 앞일 때만 잠금 디렉터리를 만든다. 잡으면 내 표를 지운다. 못 잡고 포기해도 내 표를 지운다.
# - 오래된 표: now_epoch > end_epoch + 900, 또는 host=windows 인데 그 pid 프로세스가 없음 -> 지운다. (host=wsl 표의 pid 는 WSL 쪽 도구가 본다)
# - 잠금 주인과 owner 가 같은 가장 앞 표는 "잡았음" 으로 보고 지운다(대기열을 모르는 도구로 잡은 경우).
# - 한 번에 30 분 이내(넘게 주면 30 으로 자름). 잡은 뒤에도 VRAM 에 민감한 일은 nvidia-smi 로 다른 프로세스가 없는지 따로 확인한다.
param([Parameter(Position = 0)][ValidateSet('acquire', 'release', 'status', '')][string]$Action = '',
      [string]$Owner = 'fasteval', [string]$Purpose = '', [int]$Minutes = 30, [string]$VramGb = '?', [int]$MaxWaitMin = 120, [switch]$Lib)
$script:GpuLockDir = 'C:\behavior-2026\.gpu_lock'
$script:GpuQueueDir = 'C:\behavior-2026\.gpu_queue'
$script:Utf8NoBom = [Text.UTF8Encoding]::new($false)

function Get-NowEpoch { [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() }
function Get-NowEpochNs { ([DateTime]::UtcNow.Ticks - 621355968000000000L) * 100L }

function Read-KeyFile([string]$f) {
    if (-not (Test-Path $f)) { return $null }
    $h = @{}
    try { $lines = [IO.File]::ReadAllLines($f) } catch { return $null }
    foreach ($l in $lines) { if ($l -match '^\s*([A-Za-z_]+)\s*=\s*(.*)$') { $h[$Matches[1]] = $Matches[2].Trim() } }
    return $h
}
function Get-GpuLockOwner { Read-KeyFile (Join-Path $script:GpuLockDir 'owner.txt') }

function Test-EpochStale($h) {
    if ($null -eq $h -or -not $h.ContainsKey('end_epoch')) { return $false }
    $end = 0L
    if (-not [long]::TryParse($h['end_epoch'], [ref]$end)) { return $false }
    return ((Get-NowEpoch) -gt ($end + 900))
}
function Test-GpuLockStale { Test-EpochStale (Get-GpuLockOwner) }

# ---- 대기열 ----
function Get-QueueTickets {
    if (-not (Test-Path $script:GpuQueueDir)) { return @() }
    $t = @(Get-ChildItem $script:GpuQueueDir -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^(\d+)_' } |
           ForEach-Object { [pscustomobject]@{ Path = $_.FullName; Name = $_.Name; Ns = [decimal]($_.Name -replace '^(\d+)_.*$', '$1') } })
    return @($t | Sort-Object Ns, Name)
}
function Write-Ticket([string]$path, $h) {
    $txt = ($h.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join "`n"
    [IO.File]::WriteAllText($path, "$txt`n", $script:Utf8NoBom)
}
function Remove-StaleTickets([string]$mine) {
    # 대기열을 모르는 도구(mkdir 만 되풀이)로 잠금을 잡은 주인의 표: 잠금 주인과 owner 가 같은 가장 앞 표 하나는 "잡았음" 으로 보고 지운다
    $lo = Get-GpuLockOwner
    $holder = if ($lo) { $lo['owner'] } else { $null }
    foreach ($t in (Get-QueueTickets)) {
        if ($t.Path -eq $mine) { continue }
        $h = Read-KeyFile $t.Path
        $dead = $false
        if ($holder -and $h -and $h['owner'] -eq $holder) { $dead = $true; $holder = $null }
        elseif (Test-EpochStale $h) { $dead = $true }
        elseif ($h -and $h['host'] -eq 'windows' -and $h['pid']) {
            $p = 0
            if ([int]::TryParse($h['pid'], [ref]$p) -and -not (Get-Process -Id $p -ErrorAction SilentlyContinue)) { $dead = $true }
        }
        if ($dead) { Write-Host "[gpu_lock] 오래된 표 지움: $($t.Name)"; Remove-Item $t.Path -Force -ErrorAction SilentlyContinue }
    }
}

function Enter-GpuLock([string]$Owner, [string]$Purpose, [int]$Minutes = 30, [string]$VramGb = '?', [int]$MaxWaitMin = 120) {
    if ($Minutes -gt 30) { $Minutes = 30 }
    $t0 = Get-NowEpoch
    New-Item -ItemType Directory -Force $script:GpuQueueDir | Out-Null
    $safe = ($Owner -replace '[^A-Za-z0-9._-]', '-')
    $mine = Join-Path $script:GpuQueueDir ("{0:D20}_{1}" -f [long](Get-NowEpochNs), $safe)
    $tk = [ordered]@{ owner = $Owner; purpose = $Purpose; host = 'windows'; pid = $PID; start_epoch = $t0; end_epoch = ($t0 + 60); minutes = $Minutes; vram_gb = $VramGb }
    Write-Ticket $mine $tk
    try {
        while ($true) {
            $tk.end_epoch = (Get-NowEpoch) + 60
            Write-Ticket $mine $tk
            Remove-StaleTickets $mine
            $q = @(Get-QueueTickets)
            $pos = [array]::IndexOf([string[]]@($q | ForEach-Object { $_.Path }), $mine)
            if ($pos -eq 0) {
                # cmd mkdir = CreateDirectoryW 한 번 -- 이미 있으면 실패(원자적)
                cmd /c mkdir "$script:GpuLockDir" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    $s = Get-NowEpoch
                    $txt = "owner=$Owner`npurpose=$Purpose`nstart_epoch=$s`nend_epoch=$($s + 60 * $Minutes)`nvram_gb=$VramGb`n"
                    [IO.File]::WriteAllText((Join-Path $script:GpuLockDir 'owner.txt'), $txt, $script:Utf8NoBom)
                    Write-Host "[gpu_lock] 잡음: $Owner / $Purpose ($Minutes 분)"
                    return $true
                }
                if (Test-GpuLockStale) {
                    $o = Get-GpuLockOwner
                    Write-Host "[gpu_lock] 오래된 잠금 지움(end_epoch + 900 초 지남): $(($o.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' ')"
                    Remove-Item $script:GpuLockDir -Recurse -Force -ErrorAction SilentlyContinue
                    continue
                }
            }
            if (((Get-NowEpoch) - $t0) -gt 60 * $MaxWaitMin) { Write-Host "[gpu_lock] $MaxWaitMin 분 기다려도 못 잡음"; return $false }
            $o = Get-GpuLockOwner
            $left = if ($o -and $o['end_epoch']) { "끝 예정까지 $([int](([long]$o['end_epoch'] - (Get-NowEpoch)) / 60)) 분" } else { '' }
            $ahead = @($q | Select-Object -First ([Math]::Max($pos, 0)) | ForEach-Object { $_.Name -replace '^\d+_', '' }) -join ', '
            $held = if ($o) { "$($o['owner']) / $($o['purpose']) / $left" } elseif (Test-Path $script:GpuLockDir) { '(owner.txt 없음)' } else { '없음' }
            Write-Host "[gpu_lock] 대기열 $($pos + 1)/$($q.Count) (앞: $ahead) · 잠금: $held -- 45 초 뒤 다시"
            Start-Sleep -Seconds 45
        }
    } finally {
        Remove-Item $mine -Force -ErrorAction SilentlyContinue  # 잡았든 포기했든 내 표는 지운다
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
            if ($o) { '[잠금]'; $o.GetEnumerator() | Sort-Object Key | ForEach-Object { "  $($_.Key)=$($_.Value)" } }
            elseif (Test-Path $script:GpuLockDir) { '[잠금] 잠김(owner.txt 없음 또는 형식 다름)' } else { '[잠금] free' }
            "[대기열] now_epoch=$(Get-NowEpoch)"
            foreach ($t in (Get-QueueTickets)) { $h = Read-KeyFile $t.Path; "  $($t.Name)  $(if ($h) { "$($h['purpose']) / end_epoch=$($h['end_epoch']) / host=$($h['host']) pid=$($h['pid'])" })" }
        }
    }
}
