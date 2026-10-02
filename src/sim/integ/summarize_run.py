"""통합 한 판 결과 요약(일회성 도구): run_eval_integ.sh 의 출력 폴더 하나 → summary.json + 화면 요약.

    python src/sim/integ/summarize_run.py outputs/integ/<이름>

모으는 것: 평가 결과(q_score·스텝), 계획기 결정(decisions.jsonl · trace), 그래프 노드 수(경계마다 · scene_server 시간선),
지연(관측 → 그래프 반영, 경계 → 결정, 접착부 비용), VRAM(시작 전·WSL 준비 뒤·판 중 최대), 검은 프레임(π0.5 입력 기준).
"""
import csv
import glob
import json
import pathlib
import statistics as st
import sys


def pct(v, unit=1.0):
    v = sorted(x * unit for x in v)
    if not v:
        return None
    q = lambda p: v[min(len(v) - 1, int(round((len(v) - 1) * p)))]
    return {"n": len(v), "p50": round(q(0.5), 2), "p90": round(q(0.9), 2), "max": round(v[-1], 2)}


def read_kv(p):
    d = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip()
    return d


def main():
    out = pathlib.Path(sys.argv[1])
    S = {"run": out.name}
    # 평가 결과
    res = sorted(glob.glob(str(out / "eval" / "**" / "*.json"), recursive=True))
    S["eval"] = [json.loads(pathlib.Path(r).read_text(encoding="utf-8")) for r in res]
    S["eval"] = [{"instance": e.get("instance_id"), "steps": e.get("steps"), "success": e.get("success"),
                  "q_score": e.get("q_score", {}).get("final"), "base_m": round(e.get("agent_distance", {}).get("base", 0), 2)}
                 for e in S["eval"]]
    # π0.5 스텝 기록(검은 프레임·스텝 시간)
    ns = out / "native_steps.csv"
    if ns.exists():
        rows = list(csv.DictReader(open(ns, encoding="utf-8")))
        sm = [float(r["step_ms"]) for r in rows if r.get("step_ms") not in (None, "", "nan")]
        blk = {c: sum(int(r[f"black_{c}"]) for r in rows) for c in ("head", "left", "right")}
        blk_steps = [int(r["step"]) for r in rows if any(int(r[f"black_{c}"]) for c in ("head", "left", "right"))]
        S["policy"] = {"steps": len(rows), "step_ms": pct(sm), "black_frames": blk, "black_steps_first": blk_steps[:10],
                       "infer_gpu_ms": pct([float(r["gpu_ms"]) for r in rows if r["new_chunk"] == "1"]),
                       "stages": sorted({r.get("stage") for r in rows if r.get("stage") not in (None, "", "-1")})}
    # 접착부 비용
    gs = out / "glue_steps.csv"
    if gs.exists():
        rows = list(csv.DictReader(open(gs, encoding="utf-8")))
        plain = [r for r in rows if r["hold"] == "0" and r["frames"] == "0"]
        fr = [r for r in rows if r["hold"] == "0" and r["frames"] != "0"]
        hold = [r for r in rows if r["hold"] == "1"]
        S["glue_us"] = {"plain_drain": pct([float(r["drain_us"]) for r in plain]), "plain_send": pct([float(r["send_us"]) for r in plain]),
                        "frame_send": pct([float(r["send_us"]) for r in fr]), "hold_total_ms": pct([float(r["send_us"]) / 1e3 for r in hold]),
                        "frame_steps": len(fr), "holds": len(hold), "mb_sent": round(sum(int(r["bytes"]) for r in rows) / 1e6, 1)}
    # 결정
    dp = out / "decisions.jsonl"
    if dp.exists():
        ds = [json.loads(l) for l in open(dp, encoding="utf-8") if l.strip()]
        S["decisions"] = [{k: d.get(k) for k in ("step", "kind", "trigger", "text", "stage", "stage_est", "applied", "decide_ms", "settle_ms", "source")}
                          for d in ds if d.get("kind") != "stage"]
        S["stage_updates"] = [(d["step"], d.get("stage")) for d in ds if d.get("kind") == "stage"]
    # 계획기 기록
    tr = out / "trace" / "trace.jsonl"
    if tr.exists():
        ev = [json.loads(l) for l in open(tr, encoding="utf-8") if l.strip()]
        b = [e for e in ev if e.get("type") == "boundary"]
        S["graph_nodes_at_boundaries"] = [(e["event"]["step"], len(e.get("nodes") or []),
                                            sorted({n.get("label") for n in (e.get("nodes") or []) if not n.get("structural")})[:12]) for e in b]
        llm = [e for e in ev if e.get("type") == "llm"]
        S["llm"] = {"calls": len(llm), "latency_ms": pct([e.get("result", {}).get("latency_ms", 0) for e in llm if isinstance(e.get("result"), dict)])}
        end = [e for e in ev if e.get("type") == "link_end"]
        if end:
            S["link_end"] = {k: end[-1].get(k) for k in ("link", "sink")}
        errs = [e for e in ev if e.get("type") in ("graph_error", "llm_error")]
        S["errors"] = [(e.get("type"), str(e.get("error"))[:120]) for e in errs[:8]]
    # VRAM
    S["vram"] = {**read_kv(out / "vram.txt"), **read_kv(out / "vram_windows.txt"), **read_kv(out / "vram_host.txt")}
    vt = out / "vram_timeline.csv"
    if vt.exists():
        vals = [int(l.split(",")[1]) for l in vt.read_text().splitlines() if "," in l and l.split(",")[1].strip().isdigit()]
        if vals:
            S["vram"]["gpu_used_mib_run_max"] = max(vals)
            S["vram"]["gpu_used_mib_run_p50"] = int(st.median(vals))
    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")
    # 화면
    print(f"== {out.name}")
    for e in S.get("eval", []):
        print(f"평가: 인스턴스 {e['instance']} 스텝 {e['steps']} 성공 {e['success']} q {e['q_score']} 이동 {e['base_m']} m")
    if "policy" in S:
        p = S["policy"]
        print(f"π0.5: 스텝 {p['steps']} 스텝시간 {p['step_ms']} 추론 {p['infer_gpu_ms']} 검은 프레임 {p['black_frames']} 단계 {p['stages']}")
    if "glue_us" in S:
        print(f"접착부: {S['glue_us']}")
    for d in S.get("decisions", []):
        print(f"결정 @{d['step']}: {d['kind']} ({d['trigger']}) {d['text']!r} stage {d['stage']}/est {d['stage_est']} "
              f"{d['decide_ms']} ms settle {d['settle_ms']} applied {d['applied']}")
    if S.get("stage_updates"):
        print(f"단계 갱신: {S['stage_updates']}")
    for s_, n, labels in S.get("graph_nodes_at_boundaries", []):
        print(f"경계 @{s_}: 그래프 노드 {n} {labels}")
    if "llm" in S:
        print(f"LLM: {S['llm']}")
    if "link_end" in S:
        le = S["link_end"]
        print(f"link: {json.dumps(le.get('link'), ensure_ascii=False)}")
        sk = dict(le.get("sink") or {})
        tl = sk.pop("nodes_timeline", None)
        tl = tl or sk.pop("objects_timeline", None)
        print(f"scenemap: {json.dumps(sk, ensure_ascii=False)}")
        if tl:
            print(f"물체 수 시간선(s, 물체): {tl[:40]}")
    print(f"VRAM: {S['vram']}")
    for e in S.get("errors", []):
        print("오류:", e)


if __name__ == "__main__":
    main()
