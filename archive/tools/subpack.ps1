# 제출 패키지 도구 -- 평가 결과 폴더들에서 판별 JSON·영상·래퍼·로봇 설정을 모으고, 검사하고, README(영문)·자체 점수·zip 을 만든다.
# 포털 제출은 하지 않는다(폴더와 zip 까지). 규칙: docs\제출지침.md (4절 패키지, 7절 체크리스트).
#
#   powershell -ExecutionPolicy Bypass -File C:\behavior-2026\tools\subpack.ps1 -Sources <결과폴더 또는 실험폴더>... -Out <패키지 폴더>
#       [-Meta tools\exp\submission_meta.json] [-ScanVideos] [-NoZip] [-AllowIssues]
#
# -Sources  : 평가기 결과 폴더(json\, videos\ 가 있는 곳) 또는 그 위 폴더(exp_run 실험 폴더 등). 아래를 다 뒤져 json\*.json 을 찾는다.
# -Meta     : README 에 넣을 정보(팀, 방법, 래퍼·로봇 설정 파일, 평가기 실행 방식·Kit 인자·이유·비트 동일 증거, 정책 서버·가중치). 예: tools\exp\submission_meta.example.json
# -ScanVideos : 영상에서 검은 칸도 센다(tools\black_frame_check.py, 느림)
# -AllowIssues: 검사에 오류가 있어도 패키지를 만든다(기본은 오류면 zip 을 안 만든다)
#
# 검사: 과제·인스턴스 누락·중복(내용이 다르면 오류 -- 골라 내지 않는다, 규칙 "No cherry-picking"), 공개 인스턴스 0~9(id 301~310) 한 번씩·rollout 0,
#       기본 제한시간(실패 판 steps = int(사람 평균 x 1.5) 또는 +1, 성공 판 steps <= 그 값 + 1), 검은 프레임 무효 판(black_frames.json·실험 status.jsonl),
#       JSON-영상 짝, 복사본이 원본과 바이트 같음(sha256), 제출지침 7절 체크리스트.
# 자체 점수: 과제마다 q_score 합 / 10 (없는 인스턴스 0) -> 100 과제 평균. 리더보드 표시식(낸 판들의 평균)도 같이.
param(
    [Parameter(Mandatory = $true)][string[]]$Sources,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Meta = '',
    [switch]$ScanVideos, [switch]$NoZip, [switch]$AllowIssues
)
$ErrorActionPreference = 'Stop'
# powershell -File 로 부르면 배열 인자가 한 문자열로 온다 -> ; 또는 , 로 나눈다
$Sources = @($Sources | ForEach-Object { $_ -split '[;,]' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Root = 'C:\behavior-2026'
$B1K = "$Root\BEHAVIOR-1K"
$Utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function P($o, [string]$name, $default) {
    if ($null -ne $o -and ($o.PSObject.Properties.Name -contains $name) -and $null -ne $o.$name) { return $o.$name }
    return $default
}
function Sha([string]$f) { (Get-FileHash -Algorithm SHA256 -LiteralPath $f).Hash.ToLower() }
function Exists([string]$p) { try { return [bool]($p -and (Test-Path -LiteralPath $p)) } catch { return $false } }  # 자리표시(<...>) 같은 잘못된 경로도 없음으로
function New-Zip([string]$dir, [string]$zipPath, $level, [string[]]$only = @()) {
    # 항목 이름은 / 로(PowerShell 5.1 의 CreateFromDirectory 는 \ 를 넣어 리눅스 unzip·파이썬 zipfile 에서 경로가 깨진다)
    # $only: 넣을 맨 위 폴더·파일 이름(비면 전부)
    $fs = [IO.File]::Open($zipPath, [IO.FileMode]::CreateNew)
    $za = [IO.Compression.ZipArchive]::new($fs, [IO.Compression.ZipArchiveMode]::Create)
    try {
        $base = (Resolve-Path $dir).Path.TrimEnd('\') + '\'
        foreach ($f in (Get-ChildItem $dir -Recurse -File | Sort-Object FullName)) {
            $name = $f.FullName.Substring($base.Length).Replace('\', '/')
            if ($only.Count -and ($name.Split('/')[0] -notin $only)) { continue }
            [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($za, $f.FullName, $name, $level)
        }
    } catch {
        $za.Dispose(); $fs.Dispose(); Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue; throw
    }
    $za.Dispose(); $fs.Dispose()
}

$errors = [Collections.Generic.List[string]]::new()
$warns = [Collections.Generic.List[string]]::new()
function Err([string]$m) { $errors.Add($m) }
function Warn([string]$m) { $warns.Add($m) }

# ---- 공식 과제 목록·사람 평균 길이 (기본 제한시간 = int(length * 1.5), eval/utils/eval_utils.py EVAL_TIMEOUT_MULTIPLIER) ----
$tasks = [ordered]@{}
foreach ($l in [IO.File]::ReadAllLines("$B1K\datasets\2026-challenge-task-instances\metadata\task.jsonl", $Utf8)) {
    if (-not $l.Trim()) { continue }
    $t = $l | ConvertFrom-Json
    $tasks[$t.task_name] = @{ index = [int]$t.task_index; length = [double]$t.length; timeout = [int][math]::Floor([double]$t.length * 1.5) }
}
$ExpectedIds = 301..310   # public_test 인덱스 0~9 (eval/utils/eval_utils.py TEST_INSTANCE_IDS[:20] 의 앞 10 개)

# ---- 메타 ----
$m = $null
if ($Meta) { $m = [IO.File]::ReadAllText((Resolve-Path $Meta).Path, $Utf8) | ConvertFrom-Json }

# ---- 판 찾기 ----
$eps = @{}   # key "task|iid|rid" -> list of @{...}
$srcStatus = @{}  # 실험 폴더 status.jsonl 의 판 폴더 -> 상태
foreach ($src in $Sources) {
    $sp = (Resolve-Path $src).Path
    Get-ChildItem $sp -Recurse -Filter status.jsonl -ErrorAction SilentlyContinue | ForEach-Object {
        foreach ($l in [IO.File]::ReadAllLines($_.FullName, $Utf8)) {
            if (-not $l.Trim()) { continue }
            $r = $l | ConvertFrom-Json
            $d = P $r 'dir' ''
            if ($d) { $srcStatus[$d.TrimEnd('\').ToLower()] = $r }
        }
    }
    $jsons = @(Get-ChildItem $sp -Recurse -Filter *.json -ErrorAction SilentlyContinue | Where-Object { $_.Directory.Name -eq 'json' })
    foreach ($f in $jsons) {
        try { $o = [IO.File]::ReadAllText($f.FullName, $Utf8) -replace '\bNaN\b', 'null' -replace '-?\bInfinity\b', 'null' | ConvertFrom-Json }
        catch { Err "JSON 을 못 읽음: $($f.FullName)"; continue }
        $task = P $o 'task' ''; $iid = [int](P $o 'instance_id' -1); $rid = [int](P $o 'rollout_id' -1)
        $run = $f.Directory.Parent.FullName
        $key = "$task|$iid|$rid"
        $e = @{ task = $task; iid = $iid; rid = $rid; json = $f.FullName; run = $run; base = $f.BaseName
                steps = [int](P $o 'steps' -1); success = [bool](P $o 'success' $false); q = [double](P (P $o 'q_score' $null) 'final' 0.0)
                sha = (Sha $f.FullName) }
        $vid = Join-Path $run "videos\$($f.BaseName).mp4"
        $e.video = if (Test-Path -LiteralPath $vid) { $vid } else { $null }
        # 검은 프레임: eval_instrumented 의 black_frames.json(판 폴더) 또는 실험 status.jsonl 의 invalid_black
        $bf = Join-Path $run 'black_frames.json'
        $e.black = 0
        if (Test-Path -LiteralPath $bf) {
            try { $bj = [IO.File]::ReadAllText($bf, $Utf8) | ConvertFrom-Json; $e.black = [int](@($bj.black.PSObject.Properties | ForEach-Object { [int]$_.Value }) | Measure-Object -Sum).Sum } catch {}
        }
        $e.black_known = (Test-Path -LiteralPath $bf)
        $st = $srcStatus[$run.TrimEnd('\').ToLower()]
        if ($st -and $st.status -eq 'invalid_black') { $e.black = [math]::Max($e.black, 1) }
        # 실행기 판이면: 제출 수치는 공식 명령 그대로가 원칙 -> 장면 재사용(-Reuse)·포팅 평가기 판은 넣지 않는다
        if ($st -and [bool](P $st 'reuse' $false)) { Err "장면 재사용(-Reuse, 개발용) 판: $key -- 제출에는 판마다 새 프로세스(공식 명령)로 다시 돌린 판만" }
        if ($st -and (P $st 'backend' 'original') -ne 'original') { Err "포팅 평가기 판: $key -- 제출은 원본 평가기 판만" }
        if (Test-Path -LiteralPath (Join-Path $run 'native_steps.csv')) { $e.native = $true }
        if (-not $eps.ContainsKey($key)) { $eps[$key] = [Collections.Generic.List[object]]::new() }
        $eps[$key].Add($e)
    }
}
if ($eps.Count -eq 0) { Write-Host "판별 JSON 을 하나도 못 찾았다: $($Sources -join ', ')"; exit 2 }

# ---- 판마다 하나로 ----
# 검은 프레임 무효 판은 먼저 뺀다(환경 결함이라 다시 돌린 것 -- 골라 내기가 아님). 남은 유효 판이 여럿이고 내용이 다르면 오류(고르지 않는다, "No cherry-picking").
$chosen = @{}
foreach ($k in $eps.Keys) {
    $l = @($eps[$k])
    $valid = @($l | Where-Object { $_.black -eq 0 })
    $invalid = @($l | Where-Object { $_.black -gt 0 })
    $pool = @(if ($valid.Count) { $valid } else { $l })  # @() 로 감싼다: 하나면 hashtable 한 개가 되어 .Count 가 키 수가 된다
    if ($valid.Count -and $invalid.Count) { Warn "검은 프레임 무효 판 $($invalid.Count) 개를 빼고 다시 돌린 유효 판을 씀: $k" }
    $shas = @($pool | ForEach-Object { $_.sha } | Sort-Object -Unique)
    if ($shas.Count -gt 1) {
        Err "중복(유효 판끼리 내용 다름) $k : $(@($pool | ForEach-Object { $_.json }) -join ' ; ') -- 규칙상 골라 낼 수 없다. 한 판만 남기고 다시 돌려라."
    } elseif ($pool.Count -gt 1) {
        Warn "중복(내용 같음, 하나만 씀) $k : $($pool.Count) 곳"
    }
    # 내용이 같은 중복이면 영상 있는 쪽
    $chosen[$k] = @($pool | Sort-Object @{ e = { [int](-not $_.video) } })[0]
}

# ---- 검사 ----
foreach ($e in $chosen.Values) {
    $k = "$($e.task)|$($e.iid)|$($e.rid)"
    if (-not $tasks.Contains($e.task)) { Err "모르는 과제 $($e.task) ($($e.json))"; continue }
    if ($e.iid -notin $ExpectedIds) { Err "공개 인스턴스 0~9(id 301~310) 밖의 판: $k" }
    if ($e.rid -ne 0) { Err "rollout_id 가 0 이 아님(인스턴스당 1 번이어야 함): $k" }
    $T = $tasks[$e.task].timeout
    if ($e.success) {
        if ($e.steps -gt $T + 1) { Err "제한시간보다 긴 성공 판(steps $($e.steps) > $($T + 1)): $k" }
    } elseif ($e.steps -ne $T -and $e.steps -ne $T + 1) {
        Err "기본 제한시간이 아님(실패 판 steps $($e.steps), 기본 $T): $k -- --max-steps 를 준 것 같다"
    }
    if (-not $e.black_known -and -not $ScanVideos) { Warn "검은 프레임 기록(black_frames.json) 없는 판: $k -- -ScanVideos 로 영상을 검사하라" }
    if ($e.black -gt 0) { Err "검은 프레임 무효 판(검은 카메라-프레임 $($e.black)): $k -- 다시 돌려야 한다" }
    if (-not $e.video) { Err "영상 없음: $k ($($e.run)\videos\$($e.base).mp4)" }
}
# 영상만 있고 JSON 이 없는 것
foreach ($src in $Sources) {
    Get-ChildItem (Resolve-Path $src).Path -Recurse -Filter *.mp4 -ErrorAction SilentlyContinue | Where-Object { $_.Directory.Name -eq 'videos' } | ForEach-Object {
        $j = Join-Path $_.Directory.Parent.FullName "json\$($_.BaseName).json"
        if (-not (Test-Path -LiteralPath $j)) { Warn "JSON 없는 영상(패키지에 안 넣음): $($_.FullName)" }
    }
}
# 누락
$missing = [Collections.Generic.List[string]]::new()
$perTask = [ordered]@{}
foreach ($t in $tasks.Keys) {
    $got = @($chosen.Values | Where-Object { $_.task -eq $t -and $_.rid -eq 0 -and $_.iid -in $ExpectedIds })
    $perTask[$t] = $got
    foreach ($id in $ExpectedIds) { if (-not ($got | Where-Object { $_.iid -eq $id })) { $missing.Add("$t|$id") } }
}
$nTasksFull = @($perTask.Keys | Where-Object { $perTask[$_].Count -eq 10 }).Count
$nTasksAny = @($perTask.Keys | Where-Object { $perTask[$_].Count -gt 0 }).Count
if ($missing.Count) { Warn "누락 $($missing.Count) 판(0 점으로 셈, 규칙 'Missing rollout instances count as zero'): 과제 $nTasksAny 개에 결과 있음, 10 판 다 있는 과제 $nTasksFull 개" }

# ---- 영상 검은 칸 (선택) ----
if ($ScanVideos) {
    . "$env:USERPROFILE\anaconda3\shell\condabin\conda-hook.ps1"; conda activate behavior  # behavior 환경이 있는 사용자 anaconda3 (PATH 에 다른 conda 가 먼저 잡히는 일이 있어 conda info --base 를 안 씀)
    $runs = @($chosen.Values | ForEach-Object { $_.run } | Sort-Object -Unique)
    $env:PYTHONIOENCODING = 'utf-8'
    $scanTxt = & python "$Root\tools\black_frame_check.py" @runs --max-ratio 0 2>&1 | ForEach-Object { "$_" }  # ($Out 과 이름이 겹치지 않게 -- PowerShell 변수는 대소문자 구분 없음)
    if ($LASTEXITCODE -ne 0) { Err "영상 검사에서 검은 칸이 나옴(black_frame_check.py): $(($scanTxt | Select-Object -Last 4) -join ' / ')" }
}

# ---- 점수 ----
$taskScores = [ordered]@{}
foreach ($t in $tasks.Keys) {
    $got = $perTask[$t]
    $sum = 0.0
    foreach ($g in $got) { $sum += [double]$g.q }
    $taskScores[$t] = @{ n = $got.Count; official = $sum / 10.0; submitted_mean = $(if ($got.Count) { $sum / $got.Count } else { 0.0 }); success = @($got | Where-Object { $_.success }).Count }
}
$official = ([double](($taskScores.Values | ForEach-Object { $_.official } | Measure-Object -Sum).Sum)) / $tasks.Count
$lbStyle = ([double](($taskScores.Values | ForEach-Object { $_.submitted_mean } | Measure-Object -Sum).Sum)) / $tasks.Count

# ---- 패키지 폴더 ----
$Out = [IO.Path]::GetFullPath($Out)
if (Test-Path $Out) { Write-Host "이미 있는 폴더라 멈춘다(덮어쓰지 않음): $Out"; exit 2 }
New-Item -ItemType Directory -Force "$Out\metrics", "$Out\videos", "$Out\wrapper", "$Out\robot_config" | Out-Null
$copyBad = 0
foreach ($e in ($chosen.Values | Sort-Object task, iid)) {
    $dj = "$Out\metrics\$($e.base).json"
    Copy-Item -LiteralPath $e.json $dj
    if ((Sha $dj) -ne $e.sha) { $copyBad++; Err "복사본이 원본과 다름: $dj" }
    if ($e.video) { Copy-Item -LiteralPath $e.video "$Out\videos\$($e.base).mp4" }
}
$wrapperFile = P $m 'wrapper_file' "$B1K\OmniGibson\omnigibson\eval\wrappers\default_wrapper.py"
$wrapperTarget = P $m 'wrapper_target' 'omnigibson.eval.wrappers.DefaultWrapper'
$robotCfg = P $m 'robot_config' "$Root\src\configs\r1pro_openpi.yaml"
foreach ($f in @($wrapperFile, $robotCfg)) { if (-not (Exists $f)) { Err "파일 없음: $f" } }
if (Exists $wrapperFile) { Copy-Item -LiteralPath $wrapperFile "$Out\wrapper\" }
if (Exists $robotCfg) { Copy-Item -LiteralPath $robotCfg "$Out\robot_config\" }
$launch = P $m 'launcher' $null
$kitArgs = @(P $launch 'kit_args' @())
$guardUsed = [bool](P $launch 'black_guard' $false)   # 결과를 겉싸개(eval_instrumented --black-guard=abort)로 뽑았나
$useLauncher = ($kitArgs.Count -gt 0) -or $guardUsed
if ($useLauncher) {
    New-Item -ItemType Directory -Force "$Out\evaluator_launcher" | Out-Null
    Copy-Item "$Root\tools\eval_instrumented.py" "$Out\evaluator_launcher\"
}
if ($kitArgs.Count) {
    $ev = P $launch 'evidence' ''
    if (Exists $ev) { Copy-Item -LiteralPath $ev "$Out\evaluator_launcher\" } elseif ($ev) { Err "비트 동일 증거 파일이 없다: $ev" } else { Err "Kit 인자를 썼는데 비트 동일 증거 파일(launcher.evidence)이 없다" }
}
$b1kCommit = (git -C $B1K rev-parse HEAD 2>$null)
$b1kTag = (git -C $B1K describe --tags 2>$null)

# ---- README (영문, 주최 측이 읽는다) ----
$wrapperName = Split-Path $wrapperFile -Leaf
$robotName = Split-Path $robotCfg -Leaf
$pol = P $m 'policy' $null
$evalArgs = "--task-name <TASK> --mode public_test --robot-config robot_config/$robotName --env-wrapper $wrapperTarget --host <HOST> --port <PORT> --instance-indices <i> --num-envs 1 --num-rollouts 1 --output-dir <OUT> --write-video"
$readme = @(
    "# BEHAVIOR Challenge 2026 - $(P $m 'team' '<team>') - evaluation README", '',
    "Method: $(P $m 'method' '<method>')", '',
    '## Results in this package', '',
    "- metrics/: $($chosen.Count) rollout JSON files exactly as written by the evaluator (unmodified; SHA-256 of every copy checked against the original).",
    "- Tasks with results: $nTasksAny / $($tasks.Count) (all 10 public instances: $nTasksFull). Missing rollouts count as zero.",
    "- Videos: $(@($chosen.Values | Where-Object { $_.video }).Count) MP4 files (one per rollout, same base name as the JSON), provided as a separate link.",
    ('- Self-evaluation score (mean over 100 tasks of sum(q_score)/10, missing = 0): {0:N4}' -f $official), '',
    '## Evaluator', '',
    "- BEHAVIOR-1K $b1kTag (commit $b1kCommit), evaluator ``OmniGibson/omnigibson/eval/eval.py``, unmodified.",
    "- Wrapper: ``$wrapperTarget`` (file: wrapper/$wrapperName). Robot config: robot_config/$robotName.",
    '- Every task, public instances 0-9, one rollout each, default time limit (1.5x mean human length, no ``--max-steps``), videos on.', '',
    'Full evaluator command (one process per instance index i = 0..9, per task):', '', '```',
    "python -m omnigibson.eval.eval $evalArgs", '```', ''
)
if ($useLauncher) {
    $readme += @(
        '### Local launcher used for these results (no effect on physics or scoring)', '',
        'These results were produced on Windows 11 (Isaac Sim 5.1). On this machine the RTX renderer intermittently returns an empty (all-zero RGBA)',
        'camera buffer in a fixed 3-step pattern. The evaluator was therefore run through a thin launcher that calls the unmodified ``omnigibson.eval.eval``',
        'main (evaluator_launcher/eval_instrumented.py) and only adds a read-only check that stops a rollout as soon as an empty camera frame would be given to the policy:', '', '```',
        "python evaluator_launcher/eval_instrumented.py --black-guard=abort $(@($kitArgs | ForEach-Object { "--kit-arg=$_" }) -join ' ') -- $evalArgs", '```', '',
        '- Rollouts stopped by this check were discarded as invalid (environment defect, not a policy outcome) and re-run from scratch; every rollout in metrics/ ran to completion without an empty frame.',
        '- On Linux (the organizers'' evaluation machines) this is not needed; the plain command above is the reference.'
    )
    if ($kitArgs.Count) {
        $readme += @(
            "- Kit start-up argument(s): $($kitArgs -join ' '). Reason: $(P $launch 'reason' '<reason>')",
            "- Evidence that physics, BDDL goal evaluation, metrics and result JSON are bit-identical to the default setting: evaluator_launcher/$(if (P $launch 'evidence' '') { Split-Path (P $launch 'evidence' '') -Leaf } else { '<missing>' })"
        )
    }
    $readme += ''
}
$readme += @(
    '## Policy serving', '',
    "- Submission method: $(P $pol 'method' '<Docker image | Policy server URL>')",
    "- Docker image: $(P $pol 'docker_uri' '<n/a>') $(if (P $pol 'docker_digest' '') { '(digest ' + (P $pol 'docker_digest' '') + ')' })",
    "- Policy server URL / ports: $(P $pol 'server_url' '<n/a>') $(P $pol 'ports' '')",
    "- Model / weights: $(P $pol 'weights' '<describe>')",
    "- Authentication: $(P $pol 'auth' 'none')",
    "- Capacity: $(P $pol 'capacity' 'single 24 GB GPU')", '',
    '## Notes', '', "$(P $m 'notes' '')"
)
[IO.File]::WriteAllLines("$Out\README.md", $readme, $Utf8)

# ---- 자체 점수 파일 ----
$score = [ordered]@{ official_self_score = $official; leaderboard_style_score = $lbStyle; rollouts = $chosen.Count; tasks_with_results = $nTasksAny; tasks_complete = $nTasksFull
                     per_task = $taskScores; missing = @($missing) }
[IO.File]::WriteAllText("$Out\self_score.json", ($score | ConvertTo-Json -Depth 5), $Utf8)

# ---- 검사 보고(한국어, 우리용 -- zip 에는 안 넣음) + 제출지침 7절 체크리스트 ----
$ck = @(
    @('평가기 태그·커밋을 README 에 적음', [bool]([bool]$b1kCommit)),
    @('로봇 설정 파일 포함', [bool]((Test-Path "$Out\robot_config\$robotName"))),
    @('래퍼 .py 포함', [bool]((Test-Path "$Out\wrapper\$wrapperName"))),
    @('과제마다 인스턴스 0~9, rollout 1 번(범위 밖·rollout≠0 판 없음)', [bool](-not ($errors | Where-Object { $_ -match '공개 인스턴스|rollout_id' }))),
    @('기본 제한시간(--max-steps 없음)', [bool](-not ($errors | Where-Object { $_ -match '제한시간' }))),
    @('결과 JSON 은 평가기가 쓴 그대로(sha256 같음)', [bool]($copyBad -eq 0)),
    @('JSON-영상 짝', [bool](-not ($errors | Where-Object { $_ -match '영상 없음' }))),
    @('검은 프레임 무효 판 없음', [bool](-not ($errors | Where-Object { $_ -match '검은' }))),
    @('중복(내용 다름) 없음 -- 골라 내기 없음', [bool](-not ($errors | Where-Object { $_ -match '중복' }))),
    @('Kit 인자를 썼다면 README 에 인자·이유·비트 동일 증거', [bool](($kitArgs.Count -eq 0) -or (Exists (P $launch 'evidence' '')))),
    @('정책 서버: Docker URI 또는 서버 URL(포트 50 개 이상)', [bool]([bool]((P $pol 'docker_uri' '') -or (P $pol 'server_url' '')))),
    @('Method description 25 자 이하', [bool](((P $m 'method' '').Length -le 25) -and [bool](P $m 'method' '')))
)
$md = @("# 제출 패키지 검사 $(Split-Path $Out -Leaf)", '', "만든 시각 $(Get-Date -Format 'yyyy-MM-dd HH:mm'), 출처: $($Sources -join ', ')", '',
        ('자체 점수(공식식: 과제마다 q 합/10, 100 과제 평균) **{0:N4}**, 리더보드 표시식(낸 판 평균) {1:N4}, 판 {2}, 결과 있는 과제 {3}, 10 판 다 있는 과제 {4}' -f $official, $lbStyle, $chosen.Count, $nTasksAny, $nTasksFull), '',
        '## 체크리스트 (docs\제출지침.md 7절)', '')
foreach ($c in $ck) { $md += "- [$(if ($c[1]) { 'x' } else { ' ' })] $($c[0])" }
$md += @('- [ ] (사람) 포털 필수 칸·규칙 확인 체크, 제출 ID 기록 -- 이 도구는 포털에 안 낸다', '', "## 오류 $($errors.Count)", '')
$md += @($errors | ForEach-Object { "- $_" })
$md += @('', "## 경고 $($warns.Count)", '')
$md += @($warns | ForEach-Object { "- $_" })
$md += @('', '## 과제별', '', '| 과제 | 판 | 성공 | q 합/10 | 낸 판 평균 |', '|---|---:|---:|---:|---:|')
foreach ($t in $taskScores.Keys) { $s = $taskScores[$t]; if ($s.n) { $md += ('| {0} | {1} | {2} | {3:N3} | {4:N3} |' -f $t, $s.n, $s.success, $s.official, $s.submitted_mean) } }
[IO.File]::WriteAllLines("$Out.checks.md", $md, $Utf8)

# ---- zip ----
if (-not $NoZip) {
    if ($errors.Count -and -not $AllowIssues) {
        Write-Host "오류 $($errors.Count) 개 -> zip 은 안 만든다(-AllowIssues 로 강제). 보고: $Out.checks.md"
    } else {
        # 제출 zip(주최 측 집계 스크립트가 zip 안 json 을 찾는다): metrics + wrapper + robot_config + README (+ evaluator_launcher)
        # (self_score.json 은 넣지 않는다 -- 집계 스크립트가 zip 안 json 을 판별 결과로 읽을 수 있어서)
        New-Zip $Out "$Out.final.zip" ([IO.Compression.CompressionLevel]::Optimal) @('metrics', 'wrapper', 'robot_config', 'evaluator_launcher', 'README.md')
        # 영상 zip(이미 압축된 mp4 라 저장만)
        New-Zip "$Out\videos" "$Out.videos.zip" ([IO.Compression.CompressionLevel]::NoCompression)
        Write-Host "zip: $Out.final.zip , $Out.videos.zip"
    }
}
Write-Host ($md[0..6] -join "`n")
Write-Host "오류 $($errors.Count), 경고 $($warns.Count). 보고: $Out.checks.md"
if ($errors.Count) { exit 1 } else { exit 0 }
