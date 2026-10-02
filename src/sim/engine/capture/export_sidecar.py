"""physx_capture.py 가 뜬 기록을 C++ 재생기(ovd_replay)가 읽는 보조 파일로 바꾼다.

    python export_sidecar.py <dump_dir>
      convex.npz            -> convex.bin   (볼록 메시 원본: 꼭짓점·인덱스·다각형 평면식)
      sidelog.npz + views.json -> sidelog.bin (OVD 에 안 남는 호출: applyCache 계열·wake/sleep·힘)
      filters.json          -> filters.txt  (omni.physx 거르개 표 재료: 그룹·거른 쌍 경로, 접촉 보고 유무)

형식은 src/sim/engine/replay/sidecar.h 주석과 같다. 결과 파일도 에셋 파생물이라 git 에 올리지 않는다.
"""
import json
import os
import struct
import sys

import numpy as np

KIND = {"art": 0, "batch": 0, "rb": 1, "psi": 2}
# OmniGibson 묶음 제어가 부르는 내부 이름 -> tensors API 이름
METHOD = {"_set_dof_position_targets": "set_dof_position_targets",
          "_set_dof_velocity_targets": "set_dof_velocity_targets",
          "_set_dof_actuation_forces": "set_dof_actuation_forces"}


def export_convex(d):
    p = os.path.join(d, "convex.npz")
    if not os.path.exists(p):
        return 0
    z = np.load(p)
    vc, ic, pc = z["vert_counts"], z["idx_counts"], z["poly_counts"]
    verts, idx, polys = z["verts"].astype(np.float32), z["indices"].astype(np.uint8), z["polygons"]
    out = bytearray(b"CVX1" + struct.pack("<I", len(vc)))
    vo = io_ = po = 0
    for nv, ni, npo in zip(vc, ic, pc):
        out += struct.pack("<III", nv, ni, npo)
        out += verts[vo:vo + nv].tobytes(); vo += nv
        out += idx[io_:io_ + ni].tobytes(); io_ += ni
        for q in polys[po:po + npo]:
            out += np.array(q[:4], np.float32).tobytes() + struct.pack("<HH", int(q[4]), int(q[5]))
        po += npo
    open(os.path.join(d, "convex.bin"), "wb").write(bytes(out))
    return len(vc)


def export_sidelog(d):
    p, pv = os.path.join(d, "sidelog.npz"), os.path.join(d, "views.json")
    if not os.path.exists(p):
        return 0
    z = np.load(p, allow_pickle=False)
    views = json.load(open(pv, encoding="utf-8")) if os.path.exists(pv) else {}
    order = list(views)
    vindex = {k: i for i, k in enumerate(order)}

    def wstr(s):
        b = s.encode("utf-8")
        return struct.pack("<I", len(b)) + b

    out = bytearray(b"SLG1" + struct.pack("<I", len(order)))
    for k in order:
        v = views[k]
        prims = v.get("prims", [])
        out += struct.pack("<II", KIND.get(v.get("kind"), 0), len(prims))
        for s in prims:
            out += wstr(s)
        out += struct.pack("<I", int(v.get("max_dofs", 0)))
        signs = v.get("signs") or []
        for i in range(len(prims)):
            sg = signs[i] if i < len(signs) else []
            out += struct.pack("<I", len(sg)) + np.array(sg, np.int8).tobytes()
    n = len(z["post"])
    skipped_eff = [0]
    out += struct.pack("<I", n)
    io_ = do = 0
    for i in range(n):
        ni, nd = int(z["idx_len"][i]), int(z["data_len"][i])
        method = METHOD.get(str(z["method"][i]), str(z["method"][i]))
        # 기록 도구가 "효과 없음"(다시 읽은 값이 쓴 값과 다름)으로 표시한 쓰기는 재생기가 건너뛰도록 이름만 바꾼다 (번호는 그대로)
        if "eff" in z.files and int(z["eff"][i]) == 0:
            method = "noeffect:" + method
            skipped_eff[0] += 1
        out += struct.pack("<QI", int(z["post"][i]), vindex.get(str(z["view"][i]), 0)) + wstr(method)
        out += struct.pack("<I", ni) + z["idx"][io_:io_ + ni].astype(np.uint32).tobytes(); io_ += ni
        out += struct.pack("<I", nd) + z["data"][do:do + nd].astype(np.float32).tobytes(); do += nd
    open(os.path.join(d, "sidelog.bin"), "wb").write(bytes(out))
    if skipped_eff[0]:
        print(f"곁기록: 효과 없던 쓰기 {skipped_eff[0]} 건은 재생에서 건너뜀 표시")
    return n


def export_filters(d):
    p = os.path.join(d, "filters.json")
    if not os.path.exists(p):
        return 0
    f = json.load(open(p, encoding="utf-8"))
    tags = [t for t in ("episode_start", "end") if t in f]
    lines, seen = [], set()
    for t in tags:
        for g in f[t]["groups"]:
            for x in g["filtered"]:
                lines.append(f"groupfilter {g['path']} {x}")
            for x in g["includes"]:
                lines.append(f"groupinclude {g['path']} {x}")
        for a, b in f[t]["rels"]:
            lines.append(f"rel {a} {b}")
    uniq = [x for x in lines if not (x in seen or seen.add(x))]
    reports = max(f[t]["contact_report_prims"] for t in tags) if tags else 0
    uniq += ["inverted 0", f"contact_report {1 if reports else 0}"]
    if len(tags) == 2 and f["episode_start"]["rels"] != f["end"]["rels"]:
        print("[주의] 롤아웃 중에 거른 쌍이 바뀌었다 (시작/끝 합집합으로 씀)")
    open(os.path.join(d, "filters.txt"), "w", encoding="utf-8").write("\n".join(uniq) + "\n")
    return len(uniq)


if __name__ == "__main__":
    d = sys.argv[1]
    print(f"convex {export_convex(d)} 개, sidelog {export_sidelog(d)} 건, filters {export_filters(d)} 줄 -> {d}")
