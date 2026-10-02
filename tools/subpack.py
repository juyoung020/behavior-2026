#!/usr/bin/env python3
"""제출 패키지 도구 (리눅스판; 옛 Windows 판 archive/tools/subpack.ps1 과 같은 선택지·검사·결과물).

평가 결과 폴더들에서 판별 JSON·영상·래퍼·로봇 설정을 모으고, 검사하고, README(영문)·자체 점수·zip 을 만든다.
포털 제출은 하지 않는다(폴더와 zip 까지). 규칙: docs/제출지침.md (4절 패키지, 7절 체크리스트).

  python3 tools/subpack.py --sources <결과폴더 또는 실험폴더>... --out <패키지 폴더>
      [--meta tools/exp/submission_meta.json] [--scan-videos] [--no-zip] [--allow-issues]

--sources     : 평가기 결과 폴더(json/, videos/ 가 있는 곳) 또는 그 위 폴더(exp_run 실험 폴더 등). 아래를 다 뒤져 json/*.json 을 찾는다.
                여러 개는 공백으로 나열하거나 ; , 로 이어 붙인다.
--meta        : README 에 넣을 정보(팀, 방법, 래퍼·로봇 설정 파일, 평가기 실행 방식·Kit 인자·이유·비트 동일 증거, 정책 서버·가중치).
                예: tools/exp/submission_meta.example.json. 경로는 저장소 기준 상대 경로도 된다(옛 C:/behavior-2026/... 도 읽음).
                launcher.platform(기본 "Ubuntu 22.04, Isaac Sim 5.1"), launcher.black_reason(검은 프레임 검사를 쓴 이유, 영문) 를 더 받는다.
--scan-videos : 영상에서 검은 칸도 센다(tools/black_frame_check.py, conda behavior 환경, 느림)
--allow-issues: 검사에 오류가 있어도 패키지를 만든다(기본은 오류면 zip 을 안 만든다)

검사: 과제·인스턴스 누락·중복(내용이 다르면 오류 -- 골라 내지 않는다, 규칙 "No cherry-picking"), 공개 인스턴스 0~9(id 301~310) 한 번씩·rollout 0,
      기본 제한시간(실패 판 steps = int(사람 평균 x 1.5) 또는 +1, 성공 판 steps <= 그 값 + 1), 검은 프레임 무효 판(black_frames.json·실험 status.jsonl),
      JSON-영상 짝, 복사본이 원본과 바이트 같음(sha256), 제출지침 7절 체크리스트.
자체 점수: 과제마다 q_score 합 / 10 (없는 인스턴스 0) -> 100 과제 평균. 리더보드 표시식(낸 판들의 평균)도 같이.
종료 코드: 오류 없으면 0, 오류 있으면 1, 입력 문제 2.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
B1K = REPO / "BEHAVIOR-1K"
EXPECTED_IDS = list(range(301, 311))  # public_test 인덱스 0~9 (eval/utils/eval_utils.py TEST_INSTANCE_IDS[:20] 의 앞 10 개)
_WIN_ROOT = re.compile(r"^(?:[A-Za-z]:[\\/]|/mnt/c/)behavior-2026(?:[\\/]|$)", re.I)


def fix_path(p) -> str:
    if not p:
        return p
    s = str(p)
    m = _WIN_ROOT.match(s)
    if m:
        return str(REPO / s[m.end():].replace("\\", "/"))
    if not os.path.isabs(s) and not Path(s).exists() and (REPO / s).exists():
        return str(REPO / s)
    return s


def P(o, name, default):
    if isinstance(o, dict) and o.get(name) is not None:
        return o[name]
    return default


def sha(f) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def exists(p) -> bool:
    try:
        return bool(p) and Path(p).exists()
    except (OSError, ValueError):
        return False


def norm_dir(d: str) -> str:
    return os.path.normpath(fix_path(d)).rstrip("/").lower()


def new_zip(src_dir: Path, zip_path: Path, compress: bool, only=()):
    """항목 이름은 / 로, 이름순. only: 넣을 맨 위 폴더·파일 이름(비면 전부)."""
    if zip_path.exists():
        raise FileExistsError(zip_path)
    try:
        with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED,
                             compresslevel=9 if compress else None) as z:
            for f in sorted(p for p in src_dir.rglob("*") if p.is_file()):
                name = f.relative_to(src_dir).as_posix()
                if only and name.split("/")[0] not in only:
                    continue
                z.write(f, name)
    except Exception:
        zip_path.unlink(missing_ok=True)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--scan-videos", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--allow-issues", action="store_true")
    a = ap.parse_args()
    sources = [x.strip() for s in a.sources for x in re.split(r"[;,]", s) if x.strip()]
    errors: list[str] = []
    warns: list[str] = []
    err, warn = errors.append, warns.append

    # ---- 공식 과제 목록·사람 평균 길이 (기본 제한시간 = int(length * 1.5), eval/utils/eval_utils.py EVAL_TIMEOUT_MULTIPLIER) ----
    tasks: dict = {}
    for line in (B1K / "datasets/2026-challenge-task-instances/metadata/task.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        tasks[t["task_name"]] = {"index": int(t["task_index"]), "length": float(t["length"]), "timeout": int(math.floor(float(t["length"]) * 1.5))}

    m = json.loads(Path(fix_path(a.meta)).read_text(encoding="utf-8")) if a.meta else None

    # ---- 판 찾기 ----
    eps: dict = {}  # "task|iid|rid" -> [ep]
    src_status: dict = {}  # 실험 폴더 status.jsonl 의 판 폴더 -> 상태
    for src in sources:
        sp = Path(src).resolve()
        if not sp.exists():
            print(f"출처가 없다: {src}")
            sys.exit(2)
        for stf in sp.rglob("status.jsonl"):
            for line in stf.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                d = P(r, "dir", "")
                if d:
                    src_status[norm_dir(d)] = r
        for f in sorted(sp.rglob("*.json")):
            if f.parent.name != "json":
                continue
            try:
                txt = re.sub(r"-?\bInfinity\b", "null", re.sub(r"\bNaN\b", "null", f.read_text(encoding="utf-8")))
                o = json.loads(txt)
            except Exception:
                err(f"JSON 을 못 읽음: {f}")
                continue
            task = P(o, "task", "")
            iid = int(P(o, "instance_id", -1))
            rid = int(P(o, "rollout_id", -1))
            run = f.parent.parent
            key = f"{task}|{iid}|{rid}"
            e = {"task": task, "iid": iid, "rid": rid, "json": f, "run": run, "base": f.stem, "steps": int(P(o, "steps", -1)),
                 "success": bool(P(o, "success", False)), "q": float(P(P(o, "q_score", None), "final", 0.0)), "sha": sha(f)}
            vid = run / "videos" / f"{f.stem}.mp4"
            e["video"] = vid if vid.exists() else None
            # 검은 프레임: eval_instrumented 의 black_frames.json(판 폴더) 또는 실험 status.jsonl 의 invalid_black
            bf = run / "black_frames.json"
            e["black"] = 0
            if bf.exists():
                try:
                    bj = json.loads(bf.read_text(encoding="utf-8"))
                    e["black"] = sum(int(v) for v in (bj.get("black") or {}).values())
                except Exception:
                    pass
            e["black_known"] = bf.exists()
            st = src_status.get(norm_dir(str(run)))
            if st and st.get("status") == "invalid_black":
                e["black"] = max(e["black"], 1)
            # 실행기 판이면: 제출 수치는 공식 명령 그대로가 원칙 -> 장면 재사용(--reuse)·포팅 평가기 판은 넣지 않는다
            if st and bool(P(st, "reuse", False)):
                err(f"장면 재사용(--reuse, 개발용) 판: {key} -- 제출에는 판마다 새 프로세스(공식 명령)로 다시 돌린 판만")
            if st and P(st, "backend", "original") != "original":
                err(f"포팅 평가기 판: {key} -- 제출은 원본 평가기 판만")
            if (run / "native_steps.csv").exists():
                e["native"] = True
            eps.setdefault(key, []).append(e)
    if not eps:
        print(f"판별 JSON 을 하나도 못 찾았다: {', '.join(sources)}")
        sys.exit(2)

    # ---- 판마다 하나로 ----
    # 검은 프레임 무효 판은 먼저 뺀다(환경 결함이라 다시 돌린 것 -- 골라 내기가 아님). 남은 유효 판이 여럿이고 내용이 다르면 오류.
    chosen: dict = {}
    for k, lst in eps.items():
        valid = [e for e in lst if e["black"] == 0]
        invalid = [e for e in lst if e["black"] > 0]
        pool = valid or lst
        if valid and invalid:
            warn(f"검은 프레임 무효 판 {len(invalid)} 개를 빼고 다시 돌린 유효 판을 씀: {k}")
        shas = sorted({e["sha"] for e in pool})
        if len(shas) > 1:
            err(f"중복(유효 판끼리 내용 다름) {k} : {' ; '.join(str(e['json']) for e in pool)} -- 규칙상 골라 낼 수 없다. 한 판만 남기고 다시 돌려라.")
        elif len(pool) > 1:
            warn(f"중복(내용 같음, 하나만 씀) {k} : {len(pool)} 곳")
        chosen[k] = sorted(pool, key=lambda e: int(not e["video"]))[0]  # 내용이 같은 중복이면 영상 있는 쪽

    # ---- 검사 ----
    for e in chosen.values():
        k = f"{e['task']}|{e['iid']}|{e['rid']}"
        if e["task"] not in tasks:
            err(f"모르는 과제 {e['task']} ({e['json']})")
            continue
        if e["iid"] not in EXPECTED_IDS:
            err(f"공개 인스턴스 0~9(id 301~310) 밖의 판: {k}")
        if e["rid"] != 0:
            err(f"rollout_id 가 0 이 아님(인스턴스당 1 번이어야 함): {k}")
        T = tasks[e["task"]]["timeout"]
        if e["success"]:
            if e["steps"] > T + 1:
                err(f"제한시간보다 긴 성공 판(steps {e['steps']} > {T + 1}): {k}")
        elif e["steps"] not in (T, T + 1):
            err(f"기본 제한시간이 아님(실패 판 steps {e['steps']}, 기본 {T}): {k} -- --max-steps 를 준 것 같다")
        if not e["black_known"] and not a.scan_videos:
            warn(f"검은 프레임 기록(black_frames.json) 없는 판: {k} -- --scan-videos 로 영상을 검사하라")
        if e["black"] > 0:
            err(f"검은 프레임 무효 판(검은 카메라-프레임 {e['black']}): {k} -- 다시 돌려야 한다")
        if not e["video"]:
            err(f"영상 없음: {k} ({e['run']}/videos/{e['base']}.mp4)")
    # 영상만 있고 JSON 이 없는 것
    for src in sources:
        for v in Path(src).resolve().rglob("*.mp4"):
            if v.parent.name == "videos" and not (v.parent.parent / "json" / f"{v.stem}.json").exists():
                warn(f"JSON 없는 영상(패키지에 안 넣음): {v}")
    # 누락
    missing: list[str] = []
    per_task: dict = {}
    for t in tasks:
        got = [e for e in chosen.values() if e["task"] == t and e["rid"] == 0 and e["iid"] in EXPECTED_IDS]
        per_task[t] = got
        for i in EXPECTED_IDS:
            if not any(g["iid"] == i for g in got):
                missing.append(f"{t}|{i}")
    n_full = sum(len(v) == 10 for v in per_task.values())
    n_any = sum(len(v) > 0 for v in per_task.values())
    if missing:
        warn(f"누락 {len(missing)} 판(0 점으로 셈, 규칙 'Missing rollout instances count as zero'): 과제 {n_any} 개에 결과 있음, 10 판 다 있는 과제 {n_full} 개")

    # ---- 영상 검은 칸 (선택) ----
    if a.scan_videos:
        runs = sorted({str(e["run"]) for e in chosen.values()})
        base = os.environ.get("CONDA_BASE") or str(Path.home() / "miniconda3")
        cmd = f'source "{base}/etc/profile.d/conda.sh" && conda activate behavior && exec python "$@"'
        p = subprocess.run(["bash", "-c", cmd, "bash", str(REPO / "tools/black_frame_check.py"), *runs, "--max-ratio", "0"],
                           capture_output=True, text=True, errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if p.returncode != 0:
            tail = (p.stdout + p.stderr).splitlines()[-4:]
            err(f"영상 검사에서 검은 칸이 나옴(black_frame_check.py): {' / '.join(tail)}")

    # ---- 점수 ----
    task_scores: dict = {}
    for t in tasks:
        got = per_task[t]
        s = sum(float(g["q"]) for g in got)
        task_scores[t] = {"n": len(got), "official": s / 10.0, "submitted_mean": s / len(got) if got else 0.0,
                          "success": sum(bool(g["success"]) for g in got)}
    official = sum(v["official"] for v in task_scores.values()) / len(tasks)
    lb_style = sum(v["submitted_mean"] for v in task_scores.values()) / len(tasks)

    # ---- 패키지 폴더 ----
    out = Path(a.out).resolve()
    if out.exists():
        print(f"이미 있는 폴더라 멈춘다(덮어쓰지 않음): {out}")
        sys.exit(2)
    for sub in ("metrics", "videos", "wrapper", "robot_config"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    copy_bad = 0
    for e in sorted(chosen.values(), key=lambda e: (e["task"], e["iid"])):
        dj = out / "metrics" / f"{e['base']}.json"
        shutil.copy2(e["json"], dj)
        if sha(dj) != e["sha"]:
            copy_bad += 1
            err(f"복사본이 원본과 다름: {dj}")
        if e["video"]:
            shutil.copy2(e["video"], out / "videos" / f"{e['base']}.mp4")
    wrapper_file = fix_path(P(m, "wrapper_file", str(B1K / "OmniGibson/omnigibson/eval/wrappers/default_wrapper.py")))
    wrapper_target = P(m, "wrapper_target", "omnigibson.eval.wrappers.DefaultWrapper")
    robot_cfg = fix_path(P(m, "robot_config", str(REPO / "src/sim/configs/r1pro_openpi.yaml")))
    for f in (wrapper_file, robot_cfg):
        if not exists(f):
            err(f"파일 없음: {f}")
    if exists(wrapper_file):
        shutil.copy2(wrapper_file, out / "wrapper")
    if exists(robot_cfg):
        shutil.copy2(robot_cfg, out / "robot_config")
    launch = P(m, "launcher", None)
    kit_args = list(P(launch, "kit_args", []))
    guard_used = bool(P(launch, "black_guard", False))  # 결과를 겉싸개(eval_instrumented --black-guard=abort)로 뽑았나
    use_launcher = bool(kit_args) or guard_used
    if use_launcher:
        (out / "evaluator_launcher").mkdir(exist_ok=True)
        shutil.copy2(REPO / "tools/eval_instrumented.py", out / "evaluator_launcher")
    evidence = fix_path(P(launch, "evidence", ""))
    if kit_args:
        if exists(evidence):
            shutil.copy2(evidence, out / "evaluator_launcher")
        elif evidence:
            err(f"비트 동일 증거 파일이 없다: {evidence}")
        else:
            err("Kit 인자를 썼는데 비트 동일 증거 파일(launcher.evidence)이 없다")

    def git(*args):
        p = subprocess.run(["git", "-C", str(B1K), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else ""

    b1k_commit, b1k_tag = git("rev-parse", "HEAD"), git("describe", "--tags")

    # ---- README (영문, 주최 측이 읽는다) ----
    wrapper_name, robot_name = Path(wrapper_file).name, Path(robot_cfg).name
    pol = P(m, "policy", None)
    eval_args = (f"--task-name <TASK> --mode public_test --robot-config robot_config/{robot_name} --env-wrapper {wrapper_target} "
                 "--host <HOST> --port <PORT> --instance-indices <i> --num-envs 1 --num-rollouts 1 --output-dir <OUT> --write-video")
    readme = [f"# BEHAVIOR Challenge 2026 - {P(m, 'team', '<team>')} - evaluation README", "",
              f"Method: {P(m, 'method', '<method>')}", "",
              "## Results in this package", "",
              f"- metrics/: {len(chosen)} rollout JSON files exactly as written by the evaluator (unmodified; SHA-256 of every copy checked against the original).",
              f"- Tasks with results: {n_any} / {len(tasks)} (all 10 public instances: {n_full}). Missing rollouts count as zero.",
              f"- Videos: {sum(1 for e in chosen.values() if e['video'])} MP4 files (one per rollout, same base name as the JSON), provided as a separate link.",
              f"- Self-evaluation score (mean over 100 tasks of sum(q_score)/10, missing = 0): {official:,.4f}", "",
              "## Evaluator", "",
              f"- BEHAVIOR-1K {b1k_tag} (commit {b1k_commit}), evaluator `OmniGibson/omnigibson/eval/eval.py`, unmodified.",
              f"- Wrapper: `{wrapper_target}` (file: wrapper/{wrapper_name}). Robot config: robot_config/{robot_name}.",
              "- Every task, public instances 0-9, one rollout each, default time limit (1.5x mean human length, no `--max-steps`), videos on.", "",
              "Full evaluator command (one process per instance index i = 0..9, per task):", "", "```",
              f"python -m omnigibson.eval.eval {eval_args}", "```", ""]
    if use_launcher:
        kit = " ".join(f"--kit-arg={k}" for k in kit_args)
        readme += ["### Local launcher used for these results (no effect on physics or scoring)", "",
                   f"These results were produced on {P(launch, 'platform', 'Ubuntu 22.04, Isaac Sim 5.1')}. "
                   + P(launch, "black_reason", "As a safeguard against empty (all-zero RGBA) camera buffers from the RTX renderer,"),
                   "the evaluator was run through a thin launcher that calls the unmodified `omnigibson.eval.eval`",
                   "main (evaluator_launcher/eval_instrumented.py) and only adds a read-only check that stops a rollout as soon as an empty camera frame would be given to the policy:",
                   "", "```", f"python evaluator_launcher/eval_instrumented.py --black-guard=abort {kit} -- {eval_args}".replace("  ", " "), "```", "",
                   "- Rollouts stopped by this check were discarded as invalid (environment defect, not a policy outcome) and re-run from scratch; every rollout in metrics/ ran to completion without an empty frame.",
                   "- The plain command above is the reference; the launcher does not change it."]
        if kit_args:
            readme += [f"- Kit start-up argument(s): {' '.join(kit_args)}. Reason: {P(launch, 'reason', '<reason>')}",
                       "- Evidence that physics, BDDL goal evaluation, metrics and result JSON are bit-identical to the default setting: "
                       f"evaluator_launcher/{Path(evidence).name if evidence else '<missing>'}"]
        readme.append("")
    dd = P(pol, "docker_digest", "")
    readme += ["## Policy serving", "",
               f"- Submission method: {P(pol, 'method', '<Docker image | Policy server URL>')}",
               f"- Docker image: {P(pol, 'docker_uri', '<n/a>')} {'(digest ' + dd + ')' if dd else ''}",
               f"- Policy server URL / ports: {P(pol, 'server_url', '<n/a>')} {P(pol, 'ports', '')}",
               f"- Model / weights: {P(pol, 'weights', '<describe>')}",
               f"- Authentication: {P(pol, 'auth', 'none')}",
               f"- Capacity: {P(pol, 'capacity', 'single 24 GB GPU')}", "",
               "## Notes", "", f"{P(m, 'notes', '')}"]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    # ---- 자체 점수 파일 ----
    score = {"official_self_score": official, "leaderboard_style_score": lb_style, "rollouts": len(chosen), "tasks_with_results": n_any,
             "tasks_complete": n_full, "per_task": task_scores, "missing": missing}
    (out / "self_score.json").write_text(json.dumps(score, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 검사 보고(한국어, 우리용 -- zip 에는 안 넣음) + 제출지침 7절 체크리스트 ----
    has = lambda pat: any(re.search(pat, x) for x in errors)  # noqa: E731
    method = P(m, "method", "")
    ck = [("평가기 태그·커밋을 README 에 적음", bool(b1k_commit)),
          ("로봇 설정 파일 포함", (out / "robot_config" / robot_name).exists()),
          ("래퍼 .py 포함", (out / "wrapper" / wrapper_name).exists()),
          ("과제마다 인스턴스 0~9, rollout 1 번(범위 밖·rollout≠0 판 없음)", not has("공개 인스턴스|rollout_id")),
          ("기본 제한시간(--max-steps 없음)", not has("제한시간")),
          ("결과 JSON 은 평가기가 쓴 그대로(sha256 같음)", copy_bad == 0),
          ("JSON-영상 짝", not has("영상 없음")),
          ("검은 프레임 무효 판 없음", not has("검은")),
          ("중복(내용 다름) 없음 -- 골라 내기 없음", not has("중복")),
          ("Kit 인자를 썼다면 README 에 인자·이유·비트 동일 증거", (not kit_args) or exists(evidence)),
          ("정책 서버: Docker URI 또는 서버 URL(포트 50 개 이상)", bool(P(pol, "docker_uri", "") or P(pol, "server_url", ""))),
          ("Method description 25 자 이하", bool(method) and len(method) <= 25)]
    md = [f"# 제출 패키지 검사 {out.name}", "", f"만든 시각 {dt.datetime.now():%Y-%m-%d %H:%M}, 출처: {', '.join(sources)}", "",
          f"자체 점수(공식식: 과제마다 q 합/10, 100 과제 평균) **{official:,.4f}**, 리더보드 표시식(낸 판 평균) {lb_style:,.4f}, "
          f"판 {len(chosen)}, 결과 있는 과제 {n_any}, 10 판 다 있는 과제 {n_full}", "",
          "## 체크리스트 (docs/제출지침.md 7절)", ""]
    md += [f"- [{'x' if ok else ' '}] {name}" for name, ok in ck]
    md += ["- [ ] (사람) 포털 필수 칸·규칙 확인 체크, 제출 ID 기록 -- 이 도구는 포털에 안 낸다", "", f"## 오류 {len(errors)}", ""]
    md += [f"- {x}" for x in errors]
    md += ["", f"## 경고 {len(warns)}", ""]
    md += [f"- {x}" for x in warns]
    md += ["", "## 과제별", "", "| 과제 | 판 | 성공 | q 합/10 | 낸 판 평균 |", "|---|---:|---:|---:|---:|"]
    for t, s in task_scores.items():
        if s["n"]:
            md.append(f"| {t} | {s['n']} | {s['success']} | {s['official']:,.3f} | {s['submitted_mean']:,.3f} |")
    Path(f"{out}.checks.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- zip ----
    if not a.no_zip:
        if errors and not a.allow_issues:
            print(f"오류 {len(errors)} 개 -> zip 은 안 만든다(--allow-issues 로 강제). 보고: {out}.checks.md")
        else:
            # 제출 zip(주최 측 집계 스크립트가 zip 안 json 을 찾는다): metrics + wrapper + robot_config + README (+ evaluator_launcher)
            # (self_score.json 은 넣지 않는다 -- 집계 스크립트가 zip 안 json 을 판별 결과로 읽을 수 있어서)
            new_zip(out, Path(f"{out}.final.zip"), True, ("metrics", "wrapper", "robot_config", "evaluator_launcher", "README.md"))
            new_zip(out / "videos", Path(f"{out}.videos.zip"), False)  # 이미 압축된 mp4 라 저장만
            print(f"zip: {out}.final.zip , {out}.videos.zip")
    print("\n".join(md[:7]))
    print(f"오류 {len(errors)}, 경고 {len(warns)}. 보고: {out}.checks.md")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
