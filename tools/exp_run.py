#!/usr/bin/env python3
"""한 명령 폐루프 실험 실행기 (리눅스판; 옛 Windows 판 archive/tools/exp_run.ps1 과 같은 선택지·행렬 형식·결과 형식).

과제 x 인스턴스 x 설정을 평가기로 돌리고, 검은 프레임이 나온 판은 그 자리에서 끊어 "무효"로 적고,
결과 JSON 을 모아 과제별·설정별 표를 만든다. 설계: docs/평가기_가속설계.md 13절.

  run     : python3 tools/exp_run.py run --matrix tools/exp/smoke_radio.json [--backend original|ported] [--reuse] [--dry-run]
  table   : python3 tools/exp_run.py table --dir <실험 폴더>
  compare : python3 tools/exp_run.py compare -A <실험 폴더 또는 판 폴더> -B <...> [--out compare.md]
            (같은 설정/과제/인덱스끼리 짝지어 물리·판정·JSON 비트 비교)

--backend original : 공식 v3.9.3 평가기 그대로(BEHAVIOR-1K 무수정) + 계측·검은 프레임 검출만(tools/eval_instrumented.py). 제출 수치는 이것으로만.
--backend ported   : 포팅 평가기(src/sim/engine/eval/ported_eval.py --backend engine|dummy [--scene-root D] [--instrument=...] -- <공식 인자>).
                     행렬의 ported 블록: backend(engine|dummy, 기본 engine), python(기본 conda behavior 파이썬), scene_root.
                     (옛 host 키는 무시 — 이 PC 하나에서 돈다.) dummy 는 GPU 잠금 없이 돈다.
--reuse            : 한 프로세스에서 인스턴스를 차례로(장면 로딩 한 번). 공식 evaluator.run() 을 인스턴스마다 다시 부른다.
                     새 프로세스 결과와 비트 동일 확인 전까지는 개발용. 검은 프레임으로 끊기면 남은 인스턴스는 새 프로세스로 이어 간다.
--dry-run          : 명령만 찍고 안 돈다.

행렬 JSON (예: tools/exp/smoke_radio.json) — 옛 Windows 판과 같은 키. 경로는 저장소 기준 상대 경로 또는 절대 경로
(옛 C:/behavior-2026/... 는 이 저장소로 바꿔 읽는다).
  name, mode(public_test), tasks[], instances[] (과제별로 instances_by_task.<과제>[] 로 바꿀 수 있음), max_steps(0 = 공식 1.5x),
  black_guard(abort|warn|off, 기본 abort), trace(기본 true), write_video(기본 false), gpu_busy_mib(기본 3500),
  repeats, lock_scope(run | repeat), lock_minutes(최대 30), timeout_min, reuse_batch, kit_args[], ported{...},
  stop_on_black, instrument_args[],
  settings[]: name, policy(local|replay|websocket), robot_config(없으면 공식 기본), wrapper(Default|RGBD|전체 경로), max_steps,
              chunk(--replay-action-chunk-size), port, extra_eval_args[], kit_args[],
              replay: actions(행동열 npz), quickack(기본 true), server(rust = Rust replaysrv(기본) | python(옛 wsl) = REPLAY_PY 파이썬
                      | conda(옛 windows) = conda behavior 파이썬), port(기본 8110, 쓰이면 다음 빈 포트)
              websocket: server.start(명령, {port}·{task} 치환, bash 로 돈다), server.ready_s(기본 600)

환경: CONDA_BASE(기본 conda info --base 또는 ~/miniconda3), REPLAY_PY(기본 ~/miniconda3/envs/behavior/bin/python),
      REPLAYSRV_BIN(기본 ~/cargo-target/replaysrv/release/replaysrv), TRACECMP_BIN(기본 ~/cargo-target/tracecmp/release/tracecmp)
GPU 잠금: tools/gpu_lock.sh (owner exp_run).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OG_DIR = REPO / "BEHAVIOR-1K" / "OmniGibson"
GPU_LOCK = REPO / "tools" / "gpu_lock.sh"
HOME = Path.home()
WRAPPERS = {"Default": "omnigibson.eval.wrappers.DefaultWrapper", "RGBD": "omnigibson.eval.wrappers.RGBDFullResWrapper"}
_WIN_ROOT = re.compile(r"^(?:[A-Za-z]:[\\/]|/mnt/c/)behavior-2026(?:[\\/]|$)", re.I)


class S:  # 실행 상태 (옛 $script: 변수)
    args = None
    env = None
    py = None
    exp: Path | None = None
    run_log: Path | None = None
    status_path: Path | None = None
    stop_all = False


def P(o, name, default):
    if isinstance(o, dict) and o.get(name) is not None:
        return o[name]
    return default


def fix_path(p) -> str:
    """옛 Windows 경로(C:/behavior-2026/..., C:\\behavior-2026\\..., /mnt/c/behavior-2026/...)는 이 저장소로, 상대 경로는 저장소 기준으로."""
    if not p:
        return p
    s = str(p)
    m = _WIN_ROOT.match(s)
    if m:
        return str(REPO / s[m.end():].replace("\\", "/"))
    if not os.path.isabs(s) and not Path(s).exists() and (REPO / s).exists():
        return str(REPO / s)
    return s


def now_hms() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def say(msg: str):
    line = f"[exp {now_hms()}] {msg}"
    print(line, flush=True)
    if S.run_log:
        with open(S.run_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def add_status(rec: dict):
    with open(S.status_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


def use_conda():
    if S.env is not None:
        return
    base = os.environ.get("CONDA_BASE")
    if not base:
        try:
            base = subprocess.run(["conda", "info", "--base"], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            base = str(HOME / "miniconda3")
    out = subprocess.run(["bash", "-c", f'source "{base}/etc/profile.d/conda.sh" && conda activate behavior && env -0'],
                         capture_output=True, check=True).stdout
    env = dict(kv.split("=", 1) for kv in out.decode("utf-8", "replace").split("\0") if "=" in kv)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OMNI_KIT_ACCEPT_EULA="YES",
               KMP_DUPLICATE_LIB_OK="TRUE")  # torch MKL + conda llvm-openmp 'OMP: Error #15' 공식 우회책 — 이 실행 프로세스에만
    S.env = env
    S.py = shutil.which("python", path=env.get("PATH")) or "python"


# ---- GPU: 공용 잠금(tools/gpu_lock.sh)을 잡은 뒤, 다른 시뮬레이터가 없고 다른 프로세스 VRAM 이 문턱 아래인지 확인 ----
# 조건이 안 맞으면 잠금을 풀고 60 초 뒤 다시(잠금을 쥔 채 기다리지 않는다). 원본 평가기 판(프로세스)마다 잡고 푼다.
def lock_enter(purpose, minutes, max_wait_min) -> bool:
    return subprocess.run(["bash", str(GPU_LOCK), "acquire", "--owner", "exp_run", "--purpose", purpose, "--minutes", str(minutes),
                           "--vram-gb", "11", "--max-wait-min", str(max_wait_min)]).returncode == 0


def lock_exit():
    subprocess.run(["bash", str(GPU_LOCK), "release", "--owner", "exp_run"], stdout=subprocess.DEVNULL)


def other_sims() -> list[int]:
    pat = re.compile(r"omnigibson|og_black_repro|isaac_black_repro")
    me = os.getpid()
    found = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "python" in cmd and pat.search(cmd):
            found.append(int(d.name))
    return found


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.split()[0])
    except Exception:
        return 0


def enter_eval_gpu(busy_mib, purpose, minutes, max_wait_min=120) -> int:
    t0 = time.time()
    while True:
        if not lock_enter(purpose, minutes, max_wait_min):
            return -1
        sims = other_sims()
        used = gpu_used_mib()
        if not sims and used < busy_mib:
            return used
        lock_exit()
        if (time.time() - t0) / 60 > max_wait_min:
            return -used
        say(f"GPU 기다림(잠금 풂): 다른 시뮬레이터 {len(sims)} 개, 사용 중 {used} MiB (문턱 {busy_mib})")
        time.sleep(60)


# ---- 정책 서버 ----
def port_busy(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def healthz(port: int) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_policy_server(s, task, out_dir: Path, tag):
    pol = P(s, "policy", "local")
    # 재생 서버 기본 포트는 8110(다른 에이전트가 쓰는 8010 과 안 겹치게). 이미 누가 듣고 있으면 다음 빈 포트로(남의 서버에 붙지 않게)
    port = int(P(s, "port", 8110 if pol == "replay" else 8000))
    log_base = S.exp / "logs" / f"server_{tag}"
    if pol == "replay":
        for _ in range(20):
            if not port_busy(port):
                break
            say(f"포트 {port} 는 이미 누가 쓴다 -> {port + 1}")
            port += 1
        act = fix_path(P(s, "actions", ""))
        kind = P(s, "server", "rust")  # rust(기본, 파이썬판과 비트 동일, 09-30) | python(옛 wsl) | conda(옛 windows)
        if kind == "rust":
            argv = [os.environ.get("REPLAYSRV_BIN", str(HOME / "cargo-target/replaysrv/release/replaysrv"))]
        elif kind in ("conda", "windows"):
            argv = [S.py, str(REPO / "tools/replay_policy_server.py")]
        else:  # python / wsl
            argv = [os.environ.get("REPLAY_PY", str(HOME / "miniconda3/envs/behavior/bin/python")), str(REPO / "tools/replay_policy_server.py")]
        argv += ["--actions", act, "--port", str(port), "--log", str(out_dir / "server_log.npz"), "--once"]
        if bool(P(s, "quickack", True)):
            argv.append("--quickack")
        out_dir.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen(argv, stdout=open(f"{log_base}.out.log", "wb"), stderr=open(f"{log_base}.log", "wb"),
                             env=S.env, start_new_session=True)
        ready = 60
    elif pol == "websocket":
        srv = P(s, "server", None)
        if srv is None:
            return {"port": port, "proc": None}  # 이미 떠 있는 서버를 쓴다
        cmd = P(srv, "start", "").replace("{port}", str(port)).replace("{task}", task)
        p = subprocess.Popen(["bash", "-lc", cmd], stdout=open(f"{log_base}.out.log", "wb"), stderr=open(f"{log_base}.log", "wb"),
                             start_new_session=True)
        ready = int(P(srv, "ready_s", 600))
    else:
        return {"port": port, "proc": None}
    for _ in range(ready):
        if healthz(port):
            return {"port": port, "proc": p}
        if p.poll() is not None:
            break
        time.sleep(1)
    say(f"정책 서버가 안 떴다: {log_base}.log")
    return {"port": port, "proc": p, "failed": True}


def kill_group(p, sig=signal.SIGTERM):
    try:
        os.killpg(p.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def stop_policy_server(srv):
    if not srv or srv.get("proc") is None:
        return
    p = srv["proc"]
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        kill_group(p)  # 이 실행이 띄운 서버(자기 프로세스 그룹)만 끈다
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            kill_group(p, signal.SIGKILL)


# ---- 평가기 한 프로세스 ----
def invoke_eval(m, s, task, idx, out_dir: Path, tag):
    a = S.args
    pol = P(s, "policy", "local")
    max_steps = int(P(s, "max_steps", P(m, "max_steps", 0)))
    wrap = P(s, "wrapper", "Default")
    wrap = WRAPPERS.get(wrap, wrap)
    guard = P(m, "black_guard", "abort")
    ours = []
    if guard != "off":
        ours.append(f"--black-guard={guard}")
    if bool(P(m, "trace", True)):
        ours.append("--trace")
    ours += [str(x) for x in P(m, "instrument_args", []) if x]  # 예: --dump-settings=<경로> (진단)
    if a.reuse:
        ours.append("--instances-seq=" + ",".join(str(i) for i in idx))
    # Kit 시작 인자(진단·환경 대응용). 공식 파일은 안 바꾸고 이 프로세스의 Kit 시작에만 덧붙는다(eval_instrumented --kit-arg)
    for ka in list(P(m, "kit_args", [])) + list(P(s, "kit_args", [])):
        if ka:
            ours.append(f"--kit-arg={ka}")
    eval_pol = "local" if pol == "local" else "websocket"
    env = dict(S.env or os.environ)
    srv = None
    if pol in ("replay", "websocket"):
        srv = {"port": int(P(s, "port", 8010))} if a.dry_run else start_policy_server(s, task, out_dir, tag)
        if srv.get("failed"):
            stop_policy_server(srv)
            return {"code": -2, "log": "", "wall": 0}
    port = srv["port"] if srv else 8000
    ea = ["--task-name", task, "--mode", P(m, "mode", "public_test"), "--policy", eval_pol, "--host", "127.0.0.1", "--port", str(port),
          "--env-wrapper", wrap, "--instance-indices", *[str(i) for i in idx], "--num-envs", "1", "--output-dir", str(out_dir), "--headless"]
    rc = P(s, "robot_config", "none")
    if rc != "none":
        ea += ["--robot-config", fix_path(rc)]
    if max_steps > 0:
        ea += ["--max-steps", str(max_steps)]
    chunk = int(P(s, "chunk", 0))
    if chunk > 1:
        ea += ["--replay-action-chunk-size", str(chunk)]
    if bool(P(m, "write_video", False)):
        ea.append("--write-video")
    ea += [str(x) for x in P(s, "extra_eval_args", [])]
    log = S.exp / "logs" / f"eval_{tag}.log"
    pt = P(m, "ported", None)
    py = S.py
    if a.backend == "ported":
        pa = ["--backend", P(pt, "backend", "engine")]
        sr = P(pt, "scene_root", "")
        if sr:
            pa += ["--scene-root", fix_path(sr)]
        pa += [f"--instrument={o}" for o in ours]
        py = P(pt, "python", None) or S.py
        argv = [str(REPO / "src/sim/engine/eval/ported_eval.py"), *pa, "--", *ea]
    else:
        argv = [str(REPO / "tools/eval_instrumented.py"), *ours, "--", *ea]
    say(f"평가기({a.backend}): python {' '.join(argv)}")
    if a.dry_run:
        return {"code": 0, "log": str(log), "wall": 0, "dry": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = subprocess.Popen([py, *argv], cwd=OG_DIR, env=env, stdout=open(log, "wb"), stderr=open(f"{log}.err", "wb"),
                         start_new_session=True)
    tmo = int(P(m, "timeout_min", 0))
    try:
        p.wait(tmo * 60 if tmo > 0 else None)
    except subprocess.TimeoutExpired:
        kill_group(p, signal.SIGKILL)
        p.wait()
        say(f"시간 초과 {tmo} 분: {tag}")
    wall = time.time() - t0
    stop_policy_server(srv)
    return {"code": p.returncode, "log": str(log), "wall": wall}


# 로그를 읽어 인스턴스마다 무슨 일이 있었는지: 검은 프레임(스텝·카메라), 재사용 모드의 인스턴스 시작/끝 시각
def read_eval_log(log):
    r = {"black": {}, "start": {}, "end": {}, "cur": None, "tail": ""}
    lines = []
    for f in (log, f"{log}.err"):
        if log and os.path.exists(f):
            lines += Path(f).read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        if mm := re.search(r"\[instances-seq\] 시작 인덱스 (\d+) .* t=([\d.]+)", line):
            r["cur"] = int(mm[1])
            r["start"][r["cur"]] = float(mm[2])
        elif mm := re.search(r"\[instances-seq\] 끝 인덱스 (\d+) .* t=([\d.]+)", line):
            r["end"][int(mm[1])] = float(mm[2])
        elif mm := re.search(r"\[black-guard\] 스텝 (\d+) 카메라 (\S+):", line):
            k = r["cur"] if r["cur"] is not None else -1
            r["black"].setdefault(k, {"step": int(mm[1]), "cam": mm[2]})
    r["tail"] = " | ".join([l for l in lines if re.search(r"Error|Traceback|exception", l, re.I)][-3:])
    return r


def run_matrix():
    a = S.args
    use_conda()
    try:
        m = json.loads(Path(a.matrix).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"행렬 JSON 을 못 읽었다({a.matrix}): {e}")
        sys.exit(2)
    for s in m["settings"]:
        for k in ("actions", "robot_config"):
            v = P(s, k, "")
            if v and v != "none" and not os.path.exists(fix_path(v)):
                print(f"설정 {s['name']} 의 {k} 파일이 없다: {v}")
                sys.exit(2)
    name = P(m, "name", "exp")
    S.exp = REPO / "outputs" / f"exp_{name}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    (S.exp / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copy(a.matrix, S.exp / "matrix.json")
    S.run_log = S.exp / "run.log"
    S.status_path = S.exp / "status.jsonl"
    if a.backend == "ported" and not (REPO / "src/sim/engine/eval/ported_eval.py").exists():
        say("포팅 평가기 진입점(src/sim/engine/eval/ported_eval.py)이 없다. 멈춤.")
        return
    busy = int(P(m, "gpu_busy_mib", 3500))
    # 포팅 평가기 dummy 백엔드는 GPU 를 안 쓴다 -> 잠금 없이
    need_gpu = not (a.backend == "ported" and P(P(m, "ported", None), "backend", "engine") == "dummy")
    R = int(P(m, "repeats", 1))  # 같은 판을 R 번. 반복이 바깥 고리라 설정끼리 번갈아 돈다
    say(f"실험 {name} -> {S.exp} (backend {a.backend}, reuse {a.reuse}, 반복 {R})")
    lock_scope = P(m, "lock_scope", "run")  # run(판마다 잡고 풂) | repeat(반복 한 바퀴 = 설정 전부를 한 잠금으로)
    stop_on_black = bool(P(m, "stop_on_black", False))
    S.stop_all = False
    for rep in range(1, R + 1):
        if S.stop_all:
            say("검은 프레임이 나와 남은 반복을 멈춤(stop_on_black)")
            break
        rep_locked = False
        if lock_scope == "repeat" and not a.dry_run and need_gpu:
            g0 = enter_eval_gpu(busy, f"exp_run {name} 반복 {rep}/{R} ({len(m['settings'])} 설정)", int(P(m, "lock_minutes", 30)))
            if g0 < 0:
                say(f"반복 {rep} 잠금 못 잡음 -> 멈춤")
                break
            rep_locked = True
        try:
            for s in m["settings"]:
                for task in m["tasks"]:
                    by_task = P(m, "instances_by_task", None)
                    idx_all = list(P(by_task, task, P(m, "instances", [0])))
                    base = S.exp / a.backend / s["name"] / task
                    if R > 1:
                        base = base / f"r{rep}"
                    todo = [int(i) for i in idx_all]
                    reuse_n = int(P(m, "reuse_batch", 5))  # 재사용 때 한 프로세스(= 한 잠금)에 넣는 인스턴스 수
                    lock_min = int(P(m, "lock_minutes", 30))
                    while todo:
                        batch = todo[:reuse_n] if a.reuse else [todo[0]]
                        out_dir = base if a.reuse else base / f"i{batch[0]}"
                        tag = f"{s['name']}_{task}_i{'-'.join(map(str, batch))}{f'_r{rep}' if R > 1 else ''}_{dt.datetime.now():%H%M%S}"
                        if a.dry_run or not need_gpu or rep_locked:
                            gpu = 0
                        else:
                            gpu = enter_eval_gpu(busy, f"exp_run {name} {s['name']} {task} i{','.join(map(str, batch))}", lock_min)
                        if gpu < 0:
                            for ix in batch:
                                add_status({"backend": a.backend, "setting": s["name"], "task": task, "index": ix,
                                            "status": "skipped_gpu_busy", "note": f"GPU {-gpu} MiB 또는 잠금 못 잡음"})
                            break
                        try:
                            res = invoke_eval(m, s, task, batch, out_dir, tag)
                        finally:
                            if not a.dry_run and need_gpu and not rep_locked:
                                lock_exit()
                        if res.get("dry"):
                            for ix in batch:
                                add_status({"backend": a.backend, "setting": s["name"], "task": task, "index": ix, "status": "dry"})
                            break
                        info = read_eval_log(res["log"])
                        progress = False
                        for ix in batch:
                            d = base / f"i{ix}" if a.reuse else out_dir
                            js = list((d / "json").glob("*.json")) if (d / "json").is_dir() else []
                            if ix in info["black"]:
                                bk = info["black"][ix]
                            elif not a.reuse and -1 in info["black"]:
                                bk = info["black"][-1]
                            else:
                                bk = None
                            if a.reuse and ix in info["start"] and ix in info["end"]:
                                wall = info["end"][ix] - info["start"][ix]
                            elif not a.reuse:
                                wall = res["wall"]
                            else:
                                wall = None
                            rec = {"backend": a.backend, "setting": s["name"], "task": task, "index": ix, "rep": rep, "dir": str(d),
                                   "log": res["log"], "exit": res["code"], "wall_s": wall, "gpu_mib_before": gpu, "reuse": bool(a.reuse),
                                   "kit_args": list(P(m, "kit_args", [])) + list(P(s, "kit_args", [])),
                                   "max_steps": int(P(s, "max_steps", P(m, "max_steps", 0)))}
                            if bk:
                                rec.update(status="invalid_black", black_step=bk["step"], black_cam=bk["cam"])
                            elif js:
                                rec["status"] = "valid"
                            elif a.reuse and ix not in info["start"]:
                                continue  # 이 프로세스에서 아직 시작 못 함 -> 다음 프로세스로
                            else:
                                rec.update(status="error", note=info["tail"])
                            add_status(rec)
                            if stop_on_black and rec["status"] == "invalid_black":
                                S.stop_all = True
                            todo.remove(ix)
                            progress = True
                            say(f"{s['name']} / {task} / 인덱스 {ix} -> {rec['status']}" + (f" (스텝 {bk['step']} {bk['cam']})" if bk else ""))
                        if not progress:
                            for ix in list(todo):
                                add_status({"backend": a.backend, "setting": s["name"], "task": task, "index": ix, "rep": rep,
                                            "status": "error", "note": f"진행 없음: {info['tail']}"})
                            break
        finally:
            if rep_locked:
                lock_exit()
    invoke_table(S.exp)


# ---- 표 ----
def _n(x, d):  # PowerShell '{0:N<d>}' (천 단위 쉼표)
    return f"{x:,.{d}f}"


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def invoke_table(exp: Path):
    exp = Path(exp)
    st = exp / "status.jsonl"
    if not st.exists():
        print(f"status.jsonl 없음: {exp}")
        return
    rows = []
    for line in st.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        q = succ = steps = simt = iid = None
        d = fix_path(P(r, "dir", ""))
        if d and Path(d, "json").is_dir():
            j = sorted(Path(d, "json").glob("*.json"))
            if j:
                o = json.loads(j[0].read_text(encoding="utf-8"))
                q = (o.get("q_score") or {}).get("final")
                succ, steps, iid = o.get("success"), o.get("steps"), o.get("instance_id")
                simt = (o.get("time") or {}).get("simulator_time")
        # 검은 프레임 수(카메라별, eval_instrumented 의 black_frames.json)
        bc = ""
        bf = Path(d, "black_frames.json") if d else None
        if bf and bf.exists():
            try:
                bj = json.loads(bf.read_text(encoding="utf-8"))
                bc = ", ".join(f"{k.split(':')[1].replace('_link', '')} {v}/{bj['total'][k]}" for k, v in (bj.get("black") or {}).items())
            except Exception:
                pass
        wall = P(r, "wall_s", None)
        rows.append({"backend": r.get("backend"), "setting": r.get("setting"), "task": r.get("task"), "index": r.get("index"),
                     "rep": P(r, "rep", 1), "instance_id": iid, "status": r.get("status"), "q_score": q, "success": succ, "steps": steps,
                     "sim_time_s": simt, "wall_s": round(float(wall), 1) if wall is not None else None,
                     "black": f"스텝 {r.get('black_step')} {r.get('black_cam')}" if r.get("status") == "invalid_black" else "",
                     "black_counts": bc, "note": P(r, "note", "")})
    cols = ["backend", "setting", "task", "index", "rep", "instance_id", "status", "q_score", "success", "steps", "sim_time_s", "wall_s",
            "black", "black_counts", "note"]
    with open(exp / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(cols)
        for r in rows:
            w.writerow([_cell(r[c]) for c in cols])
    md = [f"# 실험 결과 {exp.name}", "", "## 설정 x 과제", "",
          "| backend | 설정 | 과제 | 판 | 유효 | 무효(검은 프레임) | 오류 | 평균 q_score(유효) | 성공(유효) | 평균 스텝 | 판당 벽시계 s |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["backend"], r["setting"], r["task"]), []).append(r)
    for (b, s, t), g in groups.items():
        v = [r for r in g if r["status"] == "valid"]
        nb = sum(r["status"] == "invalid_black" for r in g)
        ne = sum(r["status"] not in ("valid", "invalid_black", "dry") for r in g)
        qs = [float(r["q_score"]) for r in v if r["q_score"] is not None]
        mq = _n(sum(qs) / len(qs), 3) if v and qs else "-"
        sc = f"{sum(r['success'] is True for r in v)}/{len(v)}" if v else "-"
        ss = [float(r["steps"]) for r in v if r["steps"] is not None]
        ms = _n(sum(ss) / len(ss), 0) if v and ss else "-"
        w = [r["wall_s"] for r in g if r["wall_s"] is not None]
        mw = _n(sum(w) / len(w), 0) if w else "-"
        md.append(f"| {b} | {s} | {t} | {len(g)} | {len(v)} | {nb} | {ne} | {mq} | {sc} | {ms} | {mw} |")
    md += ["", "## 판마다", "",
           "| backend | 설정 | 과제 | 인덱스 | 반복 | 인스턴스 | 상태 | q_score | 성공 | 스텝 | 시뮬 시간 s | 벽시계 s | 첫 검은 프레임 | 검은 프레임 수 | 메모 |",
           "|---|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|---|---|---|"]
    for r in rows:
        md.append("| " + " | ".join(_cell(r[c]) for c in cols[:-1]) + f" | {_cell(r['note']).replace('|', '/')} |")
    (exp / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n표: {exp / 'summary.md'}, {exp / 'results.csv'}")


# ---- 비교 ----
def get_run_dirs(root: Path) -> dict:
    if (root / "trace.npz").exists():
        return {".": root}
    h = {}
    is_exp = (root / "status.jsonl").exists()
    for f in sorted(root.rglob("trace.npz")):
        parts = f.parent.relative_to(root).parts
        # 실험 폴더면 <backend>/ 를 뗀 나머지(설정/과제[/r<k>]/i<n>)로 짝짓는다(백엔드끼리 비교가 되게)
        key = "/".join(parts[1:]) if is_exp and len(parts) >= 2 else "/".join(parts)
        h[key] = f.parent
    return h


def invoke_compare():
    a = S.args
    A, B = Path(a.A).resolve(), Path(a.B).resolve()
    da, db = get_run_dirs(A), get_run_dirs(B)
    out_md = Path(a.out) if a.out else A / "compare.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md = [f"# 비교: A = {a.A}", f"#       B = {a.B}", "",
          "물리·판정·지표·결과 JSON 은 비트 동일만 통과, 영상은 RTX 잡음이라 표에만(trace_compare --pixels-report-only).", "",
          "| 짝 | 판정 | 요약 |", "|---|---|---|"]
    if len(da) == 1 and len(db) == 1:
        # 판 하나끼리(예: 원본 판 폴더 vs 포팅 실험 폴더의 판 하나)는 이름이 달라도 짝짓는다
        ka, kb = next(iter(da)), next(iter(db))
        da, db = {f"{ka} ~ {kb}": da[ka]}, {f"{ka} ~ {kb}": db[kb]}
    keys = sorted(k for k in da if k in db)
    if not keys:
        print("짝이 되는 판이 없다(trace.npz 가 있는 같은 설정/과제/인덱스)")
        return
    fail = 0
    # 비교기: Rust 판(src/sim/fasteval/tracecmp — 파이썬판과 출력·종료 코드가 글자까지 같음을 verify_vs_python.sh 로 확인)이 있으면 그것, 없으면 파이썬판
    rs = os.environ.get("TRACECMP_BIN", str(HOME / "cargo-target/tracecmp/release/tracecmp"))
    use_rs = os.access(rs, os.X_OK)
    if not use_rs:
        use_conda()
    md[3] += " 비교기: Rust tracecmp." if use_rs else " 비교기: 파이썬 trace_compare.py."
    for k in keys:
        cmd = [rs] if use_rs else [S.py, str(REPO / "tools/trace_compare.py")]
        cmd += [str(da[k]), str(db[k]), "--check", "--strict", "--pixels-report-only"]
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=None if use_rs else S.env)
        txt = (p.stdout + p.stderr).splitlines()
        if p.returncode != 0:
            fail += 1
        summ = " / ".join(l for l in txt if re.match(r"^요약:|^픽셀", l))
        md.append(f"| {k} | {'통과' if p.returncode == 0 else '**실패**'} | {summ} |")
        safe = re.sub(r"[\\/.~ ]", "_", k)
        (out_md.parent / f"compare_{safe}.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    md += ["", f"짝 {len(keys)} 개 중 실패 {fail} 개. 항목별 표: compare_*.txt"]
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if fail:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "table", "compare"])
    ap.add_argument("--matrix", default="")
    ap.add_argument("--backend", choices=["original", "ported"], default="original")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dir", default="")
    ap.add_argument("-A", default="")
    ap.add_argument("-B", default="")
    ap.add_argument("--out", default="")
    S.args = a = ap.parse_args()
    if a.cmd == "run":
        if not a.matrix:
            ap.error("--matrix 가 필요하다")
        a.matrix = fix_path(a.matrix)
        run_matrix()
    elif a.cmd == "table":
        if not a.dir:
            ap.error("--dir 가 필요하다")
        invoke_table(Path(a.dir).resolve())
    else:
        if not (a.A and a.B):
            ap.error("-A 와 -B 가 필요하다")
        invoke_compare()


if __name__ == "__main__":
    main()
