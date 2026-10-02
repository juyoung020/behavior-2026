"""두 평가 실행의 스텝별 기록을 비교해 항목마다 '비트 동일 / 허용오차 안 / 다름' 으로 판정한다.

    python C:\\behavior-2026\\tools\\trace_compare.py <기준 결과폴더> <비교 결과폴더> [--env-a 0 --env-b 0] [--check] [--strict]
    python C:\\behavior-2026\\tools\\trace_compare.py <A> <B> --negative      # 비교 도구 자체 음성 대조

읽는 것 (결과폴더 안)
    trace.npz        tools\\eval_instrumented.py --trace 가 쓴 평가기 쪽 기록 (행동·관측·로봇·과제 물체·목표 조건·지표)
    server_log.npz   tools\\replay_policy_server.py 가 쓴 서버 쪽 기록 (정책이 받은 관측의 해시·값) -- 평가기에 아무것도 안 붙인 실행도 비교 가능
    json\\*.json      공식 결과 JSON (q_score·success·steps·이동 거리)

판정 (JSBSim 포팅 때 fdm_verify --check 와 같은 틀: 항목별 최대 오차 vs 임계, --check 면 넘으면 exit 1)
    비트 동일   모든 스텝에서 값(또는 해시)이 완전히 같다
    허용오차 안 최대 |차이| <= 임계 (임계는 TOL, 공식 평가기 자기 자신과의 비교(노이즈 바닥)를 보고 정한다)
    다름       임계를 넘는 스텝이 있다 -> 처음 넘은 스텝을 적는다
--strict 면 '비트 동일' 만 통과. 결과 JSON 은 항상 정확히 같아야 통과(q_score·success·steps), 거리 지표는 상대 1e-6.
--pixels-report-only 면 영상 해시·채널 평균(RTX 잡음으로 실행마다 다름, 5.1 노이즈 바닥)은 표에만 적고 통과/실패에서 뺀다
  -- 물리·판정·지표·JSON 만으로 판정할 때(tools/exp_run.py compare).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

# 임계 (단위: 각 양의 단위). 2026-09-29 노이즈 바닥 측정 뒤 정함 -- docs\평가기_가속설계.md 4절.
TOL = {
    "robot_pose": 1e-6,     # m, 쿼터니언
    "robot_qpos": 1e-6,     # rad / m
    "robot_qvel": 1e-5,     # rad/s
    "obj::": 1e-6,          # 과제 물체 pose
    "obs::": 1e-6,          # proprio·카메라 상대 pose 등 작은 관측
    "val::": 1e-6,          # 서버 쪽 작은 관측
    "obs_mean::": 1e-3,     # 영상 채널 평균 (해시가 다를 때 얼마나 다른지)
    "agent_delta": 1e-7,    # m
    "action": 0.0,
}
EXACT_PREFIX = ("obs_hash::", "hash::", "goal_satisfied", "terminated", "truncated", "active")


def _tol(k):
    for p, v in TOL.items():
        if k == p or (p.endswith("::") and k.startswith(p)):
            return v
    return 1e-6


def load_run(d):
    out = {}
    for name in ("trace.npz", "server_log.npz"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            z = np.load(p, allow_pickle=False)
            pre = "" if name == "trace.npz" else "srv|"
            out.update({pre + k: z[k] for k in z.files})
    js = {}
    for f in glob.glob(os.path.join(d, "json", "*.json")):
        js[os.path.basename(f)] = json.load(open(f, encoding="utf-8"))
    return out, js


def compare_key(k, a, b, ea, eb, off_a=None, off_b=None):
    """반환: (판정, 최대|차이|, 처음 비트가 다른 스텝, 처음 임계 넘은 스텝, 비교 스텝 수)"""
    t = min(len(a), len(b))
    if t == 0:
        return ("비교 불가", np.nan, None, None, 0)
    x, y = a[:t, ea], b[:t, eb]
    base = k.split("|", 1)[-1]
    if x.shape != y.shape:  # 예: 포팅 평가기 dummy 의 proprio 차원이 다름 -> 모양부터 다르면 다름(스텝 0)
        return ("다름", np.inf, 0, 0, t)
    if x.dtype.kind in "US" or base.startswith(EXACT_PREFIX) or x.dtype == bool:
        neq = np.array([not np.array_equal(x[i], y[i]) for i in range(t)])
        first = int(np.argmax(neq)) if neq.any() else None
        return ("비트 동일" if first is None else "다름", float(neq.sum()), first, first, t)
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if off_a is not None and base.startswith(("obj::", "robot_pose")):
        x = x.copy(); y = y.copy()
        x[..., :3] -= off_a
        y[..., :3] -= off_b
    both_nan = np.isnan(x) & np.isnan(y)
    diff = np.where(both_nan, 0.0, np.abs(x - y))
    diff = np.where(np.isnan(diff), np.inf, diff)
    per_step = diff.reshape(t, -1).max(axis=1) if diff.size else np.zeros(t)
    bitdiff = np.array([not np.array_equal(np.where(both_nan[i], 0, x[i]), np.where(both_nan[i], 0, y[i]))
                        for i in range(t)])
    tol = _tol(base)
    first_bit = int(np.argmax(bitdiff)) if bitdiff.any() else None
    over = per_step > tol
    first_over = int(np.argmax(over)) if over.any() else None
    m = float(per_step.max()) if t else 0.0
    verdict = "비트 동일" if first_bit is None else ("허용오차 안" if first_over is None else "다름")
    return (verdict, m, first_bit, first_over, t)


def img_report(da, db, ea, eb):
    """두 실행 모두 trace_images.npz 가 있으면 같은 스텝·카메라 영상의 픽셀 차이를 적는다 (렌더 잡음 크기)."""
    pa, pb = os.path.join(da, "trace_images.npz"), os.path.join(db, "trace_images.npz")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return
    za, zb = np.load(pa), np.load(pb)
    ka = {k.split("|", 2)[0] + "|" + k.split("|", 2)[2]: k for k in za.files if k.split("|")[1] == str(ea)}
    kb = {k.split("|", 2)[0] + "|" + k.split("|", 2)[2]: k for k in zb.files if k.split("|")[1] == str(eb)}
    print("\n영상 원본 비교 (스텝|카메라: 검은 화면 여부 A/B, 다른 픽셀 비율, 평균|차|, 최대|차|, 0~255)")
    for key in sorted(set(ka) & set(kb), key=lambda s: (int(s.split("|")[0]), s)):
        x = za[ka[key]].astype(np.int16)[..., :3]
        y = zb[kb[key]].astype(np.int16)[..., :3]
        bx, by = x.max() == 0, y.max() == 0
        d = np.abs(x - y)
        cam = key.split("|")[1].split("::")[1].split(":")[1] if "::" in key else key
        print(f"  {key.split('|')[0]:>4}|{cam:<22} 검정 {'O' if bx else '-'}/{'O' if by else '-'}  "
              f"다른 픽셀 {(d.max(axis=-1) > 0).mean() * 100:5.1f}%  평균 {d.mean():6.3f}  최대 {d.max():3d}")


def compare_json(ja, jb):
    rows = []
    for name in sorted(set(ja) | set(jb)):
        a, b = ja.get(name), jb.get(name)
        if a is None or b is None:
            rows.append((name, "한쪽에 없음", "", ""))
            continue
        for key in ("success", "steps"):
            rows.append((name, key, a.get(key), b.get(key)))
        rows.append((name, "q_score", a["q_score"]["final"], b["q_score"]["final"]))
        for part in a.get("agent_distance", {}):
            rows.append((name, f"agent_distance.{part}", a["agent_distance"][part], b["agent_distance"].get(part)))
    bad = 0
    out = []
    for name, key, va, vb in rows:
        if key == "한쪽에 없음":  # 결과 JSON 이 한쪽에만 있으면 다름(판이 빠졌거나 도중에 죽음)
            ok = same = False
        elif key.startswith("agent_distance"):
            ok = va is not None and vb is not None and abs(va - vb) <= 1e-6 * max(1.0, abs(va))
            same = va == vb
        else:
            ok = same = va == vb
        verdict = "비트 동일" if same else ("허용오차 안" if ok else "다름")
        bad += verdict == "다름"
        out.append((name, key, va, vb, verdict))
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--env-a", type=int, default=0)
    ap.add_argument("--env-b", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="다름이 하나라도 있으면 exit 1")
    ap.add_argument("--strict", action="store_true", help="비트 동일만 통과")
    ap.add_argument("--negative", action="store_true", help="B 를 일부러 조금 바꿔 도구가 잡는지 본다")
    ap.add_argument("--show", default="", help="이 항목의 스텝별 차이를 몇 스텝 찍는다 (쉼표로 여러 개)")
    ap.add_argument("--pixels-report-only", action="store_true", help="영상 해시·채널 평균은 판정에서 뺀다(표에는 적음)")
    a = ap.parse_args()

    ra, ja = load_run(a.a)
    rb, jb = load_run(a.b)
    if a.negative:
        rb = {k: v.copy() for k, v in ra.items()}
        jb = json.loads(json.dumps(ja))
        mid = None
        for k in list(rb):
            v = rb[k]
            if mid is None:
                mid = len(v) // 2
            base = k.split("|", 1)[-1]
            if base.startswith("obj::") and v.dtype.kind == "f":
                v[mid:, a.env_b, 0] += 1e-5  # 과제 물체 한 개 x 를 10 um 옮김
                break
        for k in list(rb):
            if k.split("|", 1)[-1].startswith(("obs_hash::", "hash::")):
                rb[k][mid, a.env_b] = "0" * 32  # 영상 한 장 바꿈
                break
        print(f"[음성 대조] B = A 복사본에 스텝 {mid} 부터 과제 물체 x +1e-5 m, 영상 해시 1 개 변경")

    off_a = ra.get("static::scene_offset")
    off_b = rb.get("static::scene_offset")
    oa = off_a[a.env_a] if off_a is not None else None
    ob = off_b[a.env_b] if off_b is not None else None
    keys = [k for k in ra if k in rb and not k.startswith("static::")]
    print(f"A = {a.a} (env {a.env_a})\nB = {a.b} (env {a.env_b})")
    print(f"{'항목':<58}{'판정':<10}{'최대|차이|':>12}{'첫 비트차':>9}{'첫 초과':>8}{'스텝':>6}")
    counts = {"비트 동일": 0, "허용오차 안": 0, "다름": 0, "비교 불가": 0}
    pix = {"비트 동일": 0, "허용오차 안": 0, "다름": 0, "비교 불가": 0}
    for k in sorted(keys):
        v, m, fb, fo, t = compare_key(k, ra[k], rb[k], a.env_a, a.env_b, oa, ob)
        if a.pixels_report_only and k.split("|", 1)[-1].startswith(("obs_hash::", "hash::", "obs_mean::")):
            pix[v] += 1
            v = v + "(픽셀)"
        else:
            counts[v] += 1
        fb_s = "-" if fb is None else str(fb)
        fo_s = "-" if fo is None else str(fo)
        name = k if len(k) <= 57 else "…" + k[-56:]
        print(f"{name:<58}{v:<10}{m:>12.3e}{fb_s:>9}{fo_s:>8}{t:>6}")
    for k in [s for s in a.show.split(",") if s]:
        if k in ra and k in rb:
            x, y = ra[k][:, a.env_a].astype(np.float64), rb[k][:, a.env_b].astype(np.float64)
            t = min(len(x), len(y))
            d = np.abs(x[:t] - y[:t]).reshape(t, -1).max(axis=1)
            idx = sorted({0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, t - 1} & set(range(t)))
            print(f"  {k}: " + ", ".join(f"스텝{i} {d[i]:.2e}" for i in idx))
    img_report(a.a, a.b, a.env_a, a.env_b)
    jrows, jbad = compare_json(ja, jb)
    print("\n결과 JSON")
    for name, key, va, vb, v in jrows:
        print(f"  {name:<34}{key:<24}{str(va):>22}{str(vb):>22}  {v}")
    print(f"\n요약: 비트 동일 {counts['비트 동일']}, 허용오차 안 {counts['허용오차 안']}, 다름 {counts['다름']}"
          f", 비교 불가 {counts['비교 불가']}, JSON 다름 {jbad}")
    if a.pixels_report_only:
        print(f"픽셀(판정에서 뺌): 비트 동일 {pix['비트 동일']}, 허용오차 안 {pix['허용오차 안']}, 다름 {pix['다름']}")
    bad = counts["다름"] + jbad + (counts["허용오차 안"] if a.strict else 0)
    if a.negative:
        print("음성 대조: " + ("잡았다 (정상)" if bad else "못 잡았다 -- 비교가 무디다"))
        return 0 if bad else 1
    if a.check:
        print("통과" if not bad else f"실패 -- {bad} 개 항목")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
