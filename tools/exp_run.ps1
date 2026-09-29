# 한 명령 폐루프 실험 실행기 -- 과제 x 인스턴스 x 설정을 평가기로 돌리고, 검은 프레임이 나온 판은 그 자리에서 끊어 "무효"로 적고,
# 결과 JSON 을 모아 과제별·설정별 표를 만든다. 설계: docs\평가기_가속설계.md 13절.
#
#   run     : powershell -ExecutionPolicy Bypass -File C:\behavior-2026\tools\exp_run.ps1 run -Matrix C:\behavior-2026\tools\exp\smoke_radio.json
#             [-Backend original|ported] [-Reuse] [-DryRun]
#   table   : ... exp_run.ps1 table -Dir <실험 폴더>
#   compare : ... exp_run.ps1 compare -A <실험 폴더 또는 판 폴더> -B <...>   (같은 설정\과제\인덱스끼리 짝지어 물리·판정·JSON 비트 비교)
#
# -Backend original : 공식 v3.9.3 평가기 그대로(BEHAVIOR-1K 무수정) + 계측·검은 프레임 검출만(tools\eval_instrumented.py). 제출 수치는 이것으로만.
# -Backend ported   : 포팅 평가기(src\engine + 네이티브 π0.5 + 가속 부품) -- 아직 없음. 진입점이 생기면 환경변수 BEHAVIOR_PORTED_EVAL 에
#                     파이썬 모듈 경로를 넣는다. 약속: 공식 eval.py 와 같은 인자, 같은 결과 형식(json\*.json, trace.npz).
# -Reuse            : 한 프로세스에서 인스턴스를 차례로(장면 로딩 한 번). 공식 evaluator.run() 을 인스턴스마다 다시 부른다.
#                     새 프로세스 결과와 비트 동일 확인 전까지는 개발용. 검은 프레임으로 끊기면 남은 인스턴스는 새 프로세스로 이어 간다.
# -DryRun           : 명령만 찍고 안 돈다.
#
# 행렬 JSON (예: tools\exp\smoke_radio.json)
#   name, mode(public_test), tasks[], instances[] (과제별로 instances_by_task.<과제>[] 로 바꿀 수 있음), max_steps(0 = 공식 1.5x),
#   black_guard(abort|warn|off, 기본 abort), trace(기본 true), write_video(기본 false), gpu_busy_mib(기본 3500),
#   settings[]: name, policy(local|replay|websocket|native), robot_config(없으면 공식 기본), wrapper(Default|RGBD|전체 경로), max_steps,
#               chunk(--replay-action-chunk-size), port, extra_eval_args[],
#               replay: actions(행동열 npz), quickack(기본 true), server(wsl|windows, 기본 wsl)
#               websocket: server.start(명령, {port}·{task} 치환), server.kind(wsl|windows), server.ready_s(기본 600)
#               native: module("패키지.모듈:함수", 평가기 프로세스 안 정책)
param(
    [Parameter(Position = 0, Mandatory = $true)][ValidateSet('run', 'table', 'compare')][string]$Cmd,
    [string]$Matrix = '', [ValidateSet('original', 'ported')][string]$Backend = 'original',
    [switch]$Reuse, [switch]$DryRun, [string]$Dir = '', [string]$A = '', [string]$B = '', [string]$Out = ''
)
$ErrorActionPreference = 'Continue'
$Root = 'C:\behavior-2026'
$OgDir = "$Root\BEHAVIOR-1K\OmniGibson"
$WslPy = '/home/juyoung/openpi/.venv/bin/python'
$Wrappers = @{ Default = 'omnigibson.eval.wrappers.DefaultWrapper'; RGBD = 'omnigibson.eval.wrappers.RGBDFullResWrapper' }
$Utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.Encoding]::UTF8

function To-Wsl([string]$p) { '/mnt/' + $p.Substring(0, 1).ToLower() + ($p.Substring(2) -replace '\\', '/') }
function P($o, [string]$name, $default) {
    if ($null -ne $o -and ($o.PSObject.Properties.Name -contains $name) -and $null -ne $o.$name) { return $o.$name }
    return $default
}
function Say([string]$msg) { $l = "[exp $(Get-Date -Format HH:mm:ss)] $msg"; Write-Host $l; if ($script:RunLog) { [IO.File]::AppendAllText($script:RunLog, "$l`n", $Utf8) } }
function Add-Status($rec) { [IO.File]::AppendAllText($script:StatusPath, (($rec | ConvertTo-Json -Compress -Depth 5) + "`n"), $Utf8) }

function Use-Conda {
    if ($script:Py) { return }
    . "$(conda info --base)\shell\condabin\conda-hook.ps1"
    conda activate behavior
    $script:Py = (Get-Command python).Source
    $env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:OMNI_KIT_ACCEPT_EULA = 'YES'
    # torch/MKL(libiomp5md) 와 conda llvm-openmp 가 같이 올라와 'OMP: Error #15' 로 죽는 것의 공식 안내 우회책 -- 이 실행 프로세스에만
    $env:KMP_DUPLICATE_LIB_OK = 'TRUE'
}

# ---- GPU: 공용 잠금(tools\gpu_lock.ps1)을 잡은 뒤, 다른 시뮬레이터가 없고 다른 프로세스 VRAM 이 문턱 아래인지 확인 (검은 화면 (A), 5.2.1) ----
# 조건이 안 맞으면 잠금을 풀고 60 초 뒤 다시(잠금을 쥔 채 기다리지 않는다). 원본 평가기 판(프로세스)마다 잡고 푼다.
. "$Root\tools\gpu_lock.ps1" -Lib
function Enter-EvalGpu([int]$busyMib, [string]$purpose, [int]$minutes, [int]$maxWaitMin = 120) {
    $t0 = Get-Date
    while ($true) {
        if (-not (Enter-GpuLock 'exp_run' $purpose $minutes '11' $maxWaitMin)) { return -1 }
        $sims = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'omnigibson|og_black_repro|isaac_black_repro' })
        $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
        if ($sims.Count -eq 0 -and $used -lt $busyMib) { return $used }
        [void](Exit-GpuLock 'exp_run')
        if (((Get-Date) - $t0).TotalMinutes -gt $maxWaitMin) { return -$used }
        Say "GPU 기다림(잠금 풂): 다른 시뮬레이터 $($sims.Count) 개, 사용 중 $used MiB (문턱 $busyMib)"
        Start-Sleep -Seconds 60
    }
}

# ---- 정책 서버 ----
function Start-PolicyServer($s, [string]$task, [string]$outDir, [string]$tag) {
    $pol = P $s 'policy' 'local'
    $port = [int](P $s 'port' $(if ($pol -eq 'replay') { 8010 } else { 8000 }))
    $logBase = "$script:Exp\logs\server_$tag"
    if ($pol -eq 'replay') {
        $act = (Resolve-Path (P $s 'actions' '')).Path
        $qa = [bool](P $s 'quickack' $true)
        if ((P $s 'server' 'wsl') -eq 'windows') {
            $a = @("$Root\tools\replay_policy_server.py", '--actions', $act, '--port', "$port", '--log', "$outDir\server_log.npz", '--once')
            if ($qa) { $a += '--quickack' }
            $p = Start-Process $script:Py -ArgumentList $a -WindowStyle Hidden -PassThru -RedirectStandardOutput "$logBase.out.log" -RedirectStandardError "$logBase.log"
        } else {
            $cmd = "$WslPy /mnt/c/behavior-2026/tools/replay_policy_server.py --actions $(To-Wsl $act) --port $port --log $(To-Wsl $outDir)/server_log.npz --once"
            if ($qa) { $cmd += ' --quickack' }
            $p = Start-Process wsl -ArgumentList "-d Ubuntu-22.04 -u juyoung -- $cmd" -WindowStyle Hidden -PassThru -RedirectStandardOutput "$logBase.out.log" -RedirectStandardError "$logBase.log"
        }
        $ready = 60
    } elseif ($pol -eq 'websocket') {
        $srv = P $s 'server' $null
        if ($null -eq $srv) { return @{ Port = $port; Proc = $null } }  # 이미 떠 있는 서버를 쓴다
        $cmd = ((P $srv 'start' '') -replace '\{port\}', "$port") -replace '\{task\}', $task
        if ((P $srv 'kind' 'wsl') -eq 'wsl') {
            $p = Start-Process wsl -ArgumentList "-d Ubuntu-22.04 -u juyoung -- $cmd" -WindowStyle Hidden -PassThru -RedirectStandardOutput "$logBase.out.log" -RedirectStandardError "$logBase.log"
        } else {
            $p = Start-Process powershell -ArgumentList "-NoProfile -Command $cmd" -WindowStyle Hidden -PassThru -RedirectStandardOutput "$logBase.out.log" -RedirectStandardError "$logBase.log"
        }
        $ready = [int](P $srv 'ready_s' 600)
    } else {
        return @{ Port = $port; Proc = $null }
    }
    for ($i = 0; $i -lt $ready; $i++) {
        try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$port/healthz").StatusCode -eq 200) { return @{ Port = $port; Proc = $p } } } catch {}
        if ($p.HasExited) { break }
        Start-Sleep -Seconds 1
    }
    Say "정책 서버가 안 떴다: $logBase.log"
    return @{ Port = $port; Proc = $p; Failed = $true }
}
function Stop-PolicyServer($srv, $s) {
    if ($null -eq $srv -or $null -eq $srv.Proc) { return }
    if (-not $srv.Proc.WaitForExit(30000)) { try { $srv.Proc.Kill() } catch {} }
    if ((P $s 'policy' '') -eq 'replay' -and (P $s 'server' 'wsl') -eq 'wsl') {
        wsl -d Ubuntu-22.04 -u juyoung -- pkill -f "replay_policy_server.py --actions .* --port $($srv.Port) " 2>$null | Out-Null
    }
}

# ---- 평가기 한 프로세스 ----
function Invoke-Eval($m, $s, [string]$task, [int[]]$idx, [string]$outDir, [string]$tag) {
    $pol = P $s 'policy' 'local'
    $maxSteps = [int](P $s 'max_steps' (P $m 'max_steps' 0))
    $wrap = P $s 'wrapper' 'Default'
    if ($Wrappers.ContainsKey($wrap)) { $wrap = $Wrappers[$wrap] }
    $guard = P $m 'black_guard' 'abort'
    $ours = @()
    if ($guard -ne 'off') { $ours += "--black-guard=$guard" }
    if ([bool](P $m 'trace' $true)) { $ours += '--trace' }
    if ($Reuse) { $ours += "--instances-seq=$($idx -join ',')" }
    $evalPol = if ($pol -eq 'local' -or $pol -eq 'native') { 'local' } else { 'websocket' }
    if ($pol -eq 'native') { $ours += "--native-policy=$(P $s 'module' '')" }
    $srv = $null
    if ($pol -eq 'replay' -or $pol -eq 'websocket') {
        if ($DryRun) { $srv = @{ Port = [int](P $s 'port' 8010) } } else { $srv = Start-PolicyServer $s $task $outDir $tag }
        if ($srv.Failed) { Stop-PolicyServer $srv $s; return @{ Code = -2; Log = ''; Wall = 0 } }
    }
    $port = if ($srv) { $srv.Port } else { 8000 }
    $ea = @('--task-name', $task, '--mode', (P $m 'mode' 'public_test'), '--policy', $evalPol, '--host', '127.0.0.1', '--port', "$port",
            '--env-wrapper', $wrap, '--instance-indices') + ($idx | ForEach-Object { "$_" }) + @('--num-envs', '1', '--output-dir', $outDir, '--headless')
    $rc = P $s 'robot_config' 'none'
    if ($rc -ne 'none') { $ea += @('--robot-config', $rc) }
    if ($maxSteps -gt 0) { $ea += @('--max-steps', "$maxSteps") }
    $chunk = [int](P $s 'chunk' 0)
    if ($chunk -gt 1) { $ea += @('--replay-action-chunk-size', "$chunk") }
    if ([bool](P $m 'write_video' $false)) { $ea += '--write-video' }
    $ea += @(P $s 'extra_eval_args' @())
    if ($Backend -eq 'ported') {
        $entry = $env:BEHAVIOR_PORTED_EVAL
        $argv = @('-m', $entry) + $ea
    } else {
        $argv = @("$Root\tools\eval_instrumented.py") + $ours + @('--') + $ea
    }
    $log = "$script:Exp\logs\eval_$tag.log"
    Say "평가기: python $($argv -join ' ')"
    if ($DryRun) { return @{ Code = 0; Log = $log; Wall = 0; Dry = $true } }
    New-Item -ItemType Directory -Force $outDir | Out-Null
    $t0 = Get-Date
    $p = Start-Process $script:Py -ArgumentList ($argv | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -WorkingDirectory $OgDir `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $null = $p.Handle  # PowerShell 5.1: 핸들을 먼저 잡아 두지 않으면 끝난 뒤 ExitCode 가 비어 있다
    $tmo = [int](P $m 'timeout_min' 0)
    if ($tmo -gt 0) { if (-not $p.WaitForExit($tmo * 60000)) { try { $p.Kill() } catch {}; Say "시간 초과 $tmo 분: $tag" } } else { $p.WaitForExit() }
    $wall = ((Get-Date) - $t0).TotalSeconds
    Stop-PolicyServer $srv $s
    return @{ Code = $p.ExitCode; Log = $log; Wall = $wall }
}

# 로그를 읽어 인스턴스마다 무슨 일이 있었는지: 검은 프레임(스텝·카메라), 재사용 모드의 인스턴스 시작/끝 시각
function Read-EvalLog([string]$log) {
    $r = @{ Black = @{}; Start = @{}; End = @{}; Cur = $null; Tail = '' }
    $lines = @()
    foreach ($f in @($log, "$log.err")) { if (Test-Path $f) { $lines += [IO.File]::ReadAllLines($f, $Utf8) } }
    foreach ($l in $lines) {
        if ($l -match '\[instances-seq\] 시작 인덱스 (\d+) .* t=([\d.]+)') { $r.Cur = [int]$Matches[1]; $r.Start[$r.Cur] = [double]$Matches[2] }
        elseif ($l -match '\[instances-seq\] 끝 인덱스 (\d+) .* t=([\d.]+)') { $r.End[[int]$Matches[1]] = [double]$Matches[2] }
        elseif ($l -match '\[black-guard\] 스텝 (\d+) 카메라 (\S+):') {
            $k = if ($null -ne $r.Cur) { $r.Cur } else { -1 }
            if (-not $r.Black.ContainsKey($k)) { $r.Black[$k] = @{ Step = [int]$Matches[1]; Cam = $Matches[2] } }
        }
    }
    $r.Tail = (($lines | Where-Object { $_ -match 'Error|Traceback|exception' } | Select-Object -Last 3) -join ' | ')
    return $r
}

function Invoke-RunMatrix {
    Use-Conda
    try { $m = [IO.File]::ReadAllText((Resolve-Path $Matrix).Path, $Utf8) | ConvertFrom-Json -ErrorAction Stop }
    catch { Write-Host "행렬 JSON 을 못 읽었다($Matrix): $($_.Exception.Message) -- 경로는 / 로 쓰거나 \\ 로 적는다"; exit 2 }
    foreach ($s in $m.settings) {
        foreach ($k in @('actions', 'robot_config')) {
            $v = P $s $k ''
            if ($v -and $v -ne 'none' -and -not (Test-Path $v)) { Write-Host "설정 $($s.name) 의 $k 파일이 없다: $v"; exit 2 }
        }
    }
    $name = P $m 'name' 'exp'
    $script:Exp = "$Root\outputs\exp_${name}_$(Get-Date -Format yyyyMMdd_HHmmss)"
    New-Item -ItemType Directory -Force "$script:Exp\logs" | Out-Null
    Copy-Item $Matrix "$script:Exp\matrix.json"
    $script:RunLog = "$script:Exp\run.log"
    $script:StatusPath = "$script:Exp\status.jsonl"
    if ($Backend -eq 'ported' -and -not $env:BEHAVIOR_PORTED_EVAL) {
        Say "ported 평가기가 아직 없다 (src\engine 이 준비되면 BEHAVIOR_PORTED_EVAL 에 진입 모듈을 넣는다). 멈춤."
        return
    }
    $busy = [int](P $m 'gpu_busy_mib' 3500)
    Say "실험 $name -> $script:Exp (backend $Backend, reuse $([bool]$Reuse))"
    foreach ($s in $m.settings) {
        foreach ($task in $m.tasks) {
            $byTask = P $m 'instances_by_task' $null
            $idxAll = @(P $byTask $task (P $m 'instances' @(0)))
            $base = "$script:Exp\$Backend\$($s.name)\$task"
            $todo = [Collections.ArrayList]::new(); $idxAll | ForEach-Object { [void]$todo.Add([int]$_) }
            $reuseN = [int](P $m 'reuse_batch' 5)  # 재사용 때 한 프로세스(= 한 잠금)에 넣는 인스턴스 수 -- 잠금 30 분 이내
            $lockMin = [int](P $m 'lock_minutes' 30)
            while ($todo.Count -gt 0) {
                $batch = if ($Reuse) { @($todo.ToArray() | Select-Object -First $reuseN) } else { @($todo[0]) }
                $outDir = if ($Reuse) { $base } else { "$base\i$($batch[0])" }
                $tag = "$($s.name)_${task}_i$($batch -join '-')_$(Get-Date -Format HHmmss)"
                $gpu = if ($DryRun) { 0 } else { Enter-EvalGpu $busy "exp_run $name $($s.name) $task i$($batch -join ',')" $lockMin }
                if ($gpu -lt 0) {
                    foreach ($ix in $batch) { Add-Status @{ backend = $Backend; setting = $s.name; task = $task; index = $ix; status = 'skipped_gpu_busy'; note = "GPU $(-$gpu) MiB 또는 잠금 못 잡음" } }
                    break
                }
                try { $res = Invoke-Eval $m $s $task $batch $outDir $tag } finally { if (-not $DryRun) { [void](Exit-GpuLock 'exp_run') } }
                if ($res.Dry) { foreach ($ix in $batch) { Add-Status @{ backend = $Backend; setting = $s.name; task = $task; index = $ix; status = 'dry' } }; break }
                $info = Read-EvalLog $res.Log
                $progress = $false
                foreach ($ix in $batch) {
                    $d = if ($Reuse) { "$base\i$ix" } else { $outDir }
                    $js = @(Get-ChildItem "$d\json\*.json" -ErrorAction SilentlyContinue)
                    $bk = if ($info.Black.ContainsKey($ix)) { $info.Black[$ix] } elseif (-not $Reuse -and $info.Black.ContainsKey(-1)) { $info.Black[-1] } else { $null }
                    $wall = if ($Reuse -and $info.Start.ContainsKey($ix) -and $info.End.ContainsKey($ix)) { $info.End[$ix] - $info.Start[$ix] } elseif (-not $Reuse) { $res.Wall } else { $null }
                    $rec = @{ backend = $Backend; setting = $s.name; task = $task; index = $ix; dir = $d; log = $res.Log; exit = $res.Code; wall_s = $wall; gpu_mib_before = $gpu }
                    if ($bk) {
                        $rec.status = 'invalid_black'; $rec.black_step = $bk.Step; $rec.black_cam = $bk.Cam
                    } elseif ($js.Count -gt 0) {
                        $rec.status = 'valid'
                    } elseif ($Reuse -and -not $info.Start.ContainsKey($ix)) {
                        continue  # 이 프로세스에서 아직 시작 못 함 -> 다음 프로세스로
                    } else {
                        $rec.status = 'error'; $rec.note = $info.Tail
                    }
                    Add-Status $rec
                    [void]$todo.Remove($ix); $progress = $true
                    Say "$($s.name) / $task / 인덱스 $ix -> $($rec.status)$(if ($bk) { " (스텝 $($bk.Step) $($bk.Cam))" })"
                }
                if (-not $progress) { foreach ($ix in @($todo.ToArray())) { Add-Status @{ backend = $Backend; setting = $s.name; task = $task; index = $ix; status = 'error'; note = "진행 없음: $($info.Tail)" } }; break }
            }
        }
    }
    Invoke-Table $script:Exp
}

# ---- 표 ----
function Invoke-Table([string]$exp) {
    $st = "$exp\status.jsonl"
    if (-not (Test-Path $st)) { Write-Host "status.jsonl 없음: $exp"; return }
    $rows = @()
    foreach ($l in [IO.File]::ReadAllLines($st, $Utf8)) {
        if (-not $l.Trim()) { continue }
        $r = $l | ConvertFrom-Json
        $q = $null; $succ = $null; $steps = $null; $simt = $null; $iid = $null
        $d = P $r 'dir' ''
        if ($d -and (Test-Path "$d\json")) {
            $j = Get-ChildItem "$d\json\*.json" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($j) {
                $o = [IO.File]::ReadAllText($j.FullName, $Utf8) | ConvertFrom-Json
                $q = $o.q_score.final; $succ = $o.success; $steps = $o.steps; $simt = $o.time.simulator_time; $iid = $o.instance_id
            }
        }
        $rows += [pscustomobject]@{ backend = $r.backend; setting = $r.setting; task = $r.task; index = $r.index; instance_id = $iid; status = $r.status
            q_score = $q; success = $succ; steps = $steps; sim_time_s = $simt; wall_s = $(if ($null -ne (P $r 'wall_s' $null)) { [math]::Round([double]$r.wall_s, 1) } else { $null })
            black = $(if ($r.status -eq 'invalid_black') { "스텝 $($r.black_step) $($r.black_cam)" } else { '' }); note = (P $r 'note' '') }
    }
    $rows | Export-Csv "$exp\results.csv" -NoTypeInformation -Encoding UTF8
    $md = @("# 실험 결과 $(Split-Path $exp -Leaf)", '', '## 설정 x 과제', '',
            '| backend | 설정 | 과제 | 판 | 유효 | 무효(검은 프레임) | 오류 | 평균 q_score(유효) | 성공(유효) | 평균 스텝 | 판당 벽시계 s |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    foreach ($g in ($rows | Group-Object backend, setting, task)) {
        $v = @($g.Group | Where-Object { $_.status -eq 'valid' })
        $nb = @($g.Group | Where-Object { $_.status -eq 'invalid_black' }).Count
        $ne = @($g.Group | Where-Object { $_.status -notin @('valid', 'invalid_black', 'dry') }).Count
        $mq = if ($v.Count) { '{0:N3}' -f (($v | Measure-Object q_score -Average).Average) } else { '-' }
        $sc = if ($v.Count) { "$(@($v | Where-Object { $_.success -eq $true }).Count)/$($v.Count)" } else { '-' }
        $ms = if ($v.Count) { '{0:N0}' -f (($v | Measure-Object steps -Average).Average) } else { '-' }
        $w = @($g.Group | Where-Object { $null -ne $_.wall_s })
        $mw = if ($w.Count) { '{0:N0}' -f (($w | Measure-Object wall_s -Average).Average) } else { '-' }
        $f = $g.Group[0]
        $md += "| $($f.backend) | $($f.setting) | $($f.task) | $($g.Count) | $($v.Count) | $nb | $ne | $mq | $sc | $ms | $mw |"
    }
    $md += @('', '## 판마다', '', '| backend | 설정 | 과제 | 인덱스 | 인스턴스 | 상태 | q_score | 성공 | 스텝 | 시뮬 시간 s | 벽시계 s | 검은 프레임 | 메모 |', '|---|---|---|---:|---:|---|---:|---|---:|---:|---:|---|---|')
    foreach ($r in $rows) { $md += "| $($r.backend) | $($r.setting) | $($r.task) | $($r.index) | $($r.instance_id) | $($r.status) | $($r.q_score) | $($r.success) | $($r.steps) | $($r.sim_time_s) | $($r.wall_s) | $($r.black) | $($r.note -replace '\|', '/') |" }
    [IO.File]::WriteAllLines("$exp\summary.md", $md, $Utf8)
    Write-Host ($md -join "`n")
    Write-Host "`n표: $exp\summary.md, $exp\results.csv"
}

# ---- 비교 ----
function Get-RunDirs([string]$root) {
    if (Test-Path "$root\trace.npz") { return @{ '.' = $root } }
    $h = @{}
    Get-ChildItem $root -Recurse -Filter trace.npz -ErrorAction SilentlyContinue | ForEach-Object {
        $rel = $_.DirectoryName.Substring($root.TrimEnd('\').Length).TrimStart('\')
        $parts = $rel -split '\\'
        # <backend>\<설정>\<과제>\i<n> -> 설정\과제\i<n> 로 짝짓는다(백엔드끼리 비교가 되게)
        $key = if ($parts.Count -ge 4) { ($parts[-3..-1] -join '\') } else { $rel }
        $h[$key] = $_.DirectoryName
    }
    return $h
}
function Invoke-Compare {
    Use-Conda
    $da = Get-RunDirs (Resolve-Path $A).Path
    $db = Get-RunDirs (Resolve-Path $B).Path
    $outMd = if ($Out) { $Out } else { Join-Path (Resolve-Path $A).Path 'compare.md' }
    $md = @("# 비교: A = $A", "#       B = $B", '', '물리·판정·지표·결과 JSON 은 비트 동일만 통과, 영상은 RTX 잡음이라 표에만(trace_compare --pixels-report-only).', '',
            '| 짝 | 판정 | 요약 |', '|---|---|---|')
    $keys = if ($da.ContainsKey('.') -and $db.ContainsKey('.')) { @('.') } else { @($da.Keys | Where-Object { $db.ContainsKey($_) } | Sort-Object) }
    if (-not $keys.Count) { Write-Host '짝이 되는 판이 없다(trace.npz 가 있는 같은 설정\과제\인덱스)'; return }
    $fail = 0
    foreach ($k in $keys) {
        $txt = & $script:Py "$Root\tools\trace_compare.py" $da[$k] $db[$k] --check --strict --pixels-report-only 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
        if ($code -ne 0) { $fail++ }
        $sum = (($txt | Where-Object { $_ -match '^요약:|^픽셀' }) -join ' / ')
        $md += "| $k | $(if ($code -eq 0) { '통과' } else { '**실패**' }) | $sum |"
        [IO.File]::WriteAllLines("$(Split-Path $outMd)\compare_$($k -replace '[\\/.]', '_').txt", $txt, $Utf8)
    }
    $md += @('', "짝 $($keys.Count) 개 중 실패 $fail 개. 항목별 표: compare_*.txt")
    [IO.File]::WriteAllLines($outMd, $md, $Utf8)
    Write-Host ($md -join "`n")
    if ($fail) { exit 1 }
}

switch ($Cmd) {
    'run' { if (-not $Matrix) { throw '-Matrix 가 필요하다' }; Invoke-RunMatrix }
    'table' { Invoke-Table ((Resolve-Path $Dir).Path) }
    'compare' { if (-not ($A -and $B)) { throw '-A 와 -B 가 필요하다' }; Invoke-Compare }
}
