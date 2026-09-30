"""S3(판정) 입력 내보내기: 기록 폴더의 toggle.pkl(physx_capture --record-toggle) + trace.npz + scope.json + BDDL 문제
-> 재생기(ovd_replay --s3 <폴더>)가 읽는 평평한 파일.

  python3 export_s3.py <기록 폴더> [--bddl <problem.bddl>]

s3_setup.txt (줄 단위, 공백 구분)
  so S O                            ToggledOn 판 수·물체 수 (VALUES 는 (S,O) 행 우선)
  obj k name bddl_name              물체 칸 k 의 OmniGibson 이름과 BDDL 이름(없으면 -)
  marker k parent_path ox oy oz r   표식 k: 부모 링크 경로, 국소 위치(float32 비트 16진), 반지름(비트)
  finger f path n_pts n_tri         손가락 링크 f (메시는 s3_meshes.bin 에 이 순서로)
  pair k f                          (표식, 손가락) 쌍 — 공식 _marker_finger_pair 순서
  rowcol nr nc                      접촉 행렬 부분(손가락 행 × 켜짐 물체 열) 크기
  with k c0 c1 ...                  물체 k 의 열 마스크(부분 행렬 열 번호)
  qrow r0 r1 ...                    질의 행 마스크(부분 행렬 행 번호)
  rowpath i path / colpath j path   부분 행렬 행(손가락 몸체)·열(켜짐 물체 몸체) 경로
  episode_start P substeps N steps T bddl <경로>
s3_meshes.bin   손가락마다 pts(float32 n_pts*3), tri(int32 n_tri*3)
s3_rows.bin     u32 행 수, 행마다: u64 post, u8 value[S*O], f32 time[S*O], u8 cm[nr*nc], u8 ccm[nr*nc], u8 has_cm (cm·ccm 은 행 우선)
s3_goal.txt     스텝마다 공식 trace 의 goal_satisfied 문자열 한 줄 (예: [] 또는 [0])
설정이 기록 중 여러 벌이면 마지막 벌(에피소드 쪽)만 쓴다. 에피소드 전 행은 그 설정과 모양이 같을 때만 쓴다.
"""
import argparse
import json
import os
import pickle
import struct
import sys

import numpy as np


def fhex(x):
    return "%08x" % struct.unpack("<I", struct.pack("<f", float(np.float32(x))))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--bddl", default="/mnt/c/behavior-2026/BEHAVIOR-1K/bddl3/bddl/activity_definitions/turning_on_radio/problem0.bddl")
    a = ap.parse_args()
    d = a.dir
    T = pickle.load(open(os.path.join(d, "toggle.pkl"), "rb"))
    st = T["static"][-1]
    # 설정은 update_handles 때마다 새로 적히지만(잡기 조인트 등) 행·열 경로가 같으면 한 벌로 본다
    same = [i for i, x in enumerate(T["static"]) if x.get("rows") == st.get("rows") and x.get("cols") == st.get("cols")
            and len(x["pairs"]) == len(st["pairs"])]
    rows = [r for r in T["rows"] if r["static"] in same]
    S, O = rows[0]["value"].shape
    scope = json.load(open(os.path.join(d, "scope.json")))
    name2bddl = {}
    for sc in scope:
        for bn, o in sc["objects"].items():
            name2bddl[o.get("name")] = bn
    s1 = {}
    s1p = os.path.join(d, "s1_setup.txt")
    if os.path.exists(s1p):
        for line in open(s1p):
            k = line.split()
            if len(k) >= 2 and k[0] in ("episode_start_post", "substeps"):
                s1[k[0]] = int(k[1])
    tr = np.load(os.path.join(d, "trace.npz"), allow_pickle=True)
    goals = [str(x) for x in tr["goal_satisfied"][:, 0]]

    rb = st["rb_paths"]
    pairs = st["pairs"]
    fingers = sorted({int(f) for f in pairs[:, 1]})
    fidx = {f: i for i, f in enumerate(fingers)}
    q = st["query_mask"].any(0)
    w = st["with_mask"]
    rows_sel = np.where(q)[0]
    cols_sel = np.where(w.any(0))[0]
    out = []
    out.append(f"so {S} {O}")
    for k, row in enumerate(np.array(st["objs"], dtype=object).reshape(-1)):
        out.append(f"obj {k} {row if row else '-'} {name2bddl.get(row, '-') if row else '-'}")
    n_markers = len(st["marker_parent"])
    for k in range(n_markers):
        o = st["marker_offset"][k]
        out.append(f"marker {k} {rb[int(st['marker_parent'][k])]} {fhex(o[0])} {fhex(o[1])} {fhex(o[2])} {fhex(st['marker_radius'][k])}")
    mesh_bin = bytearray()
    for f in fingers:
        path = rb[f]
        pts, tri = st["finger_meshes"].get(path, (np.zeros((0, 3), np.float32), np.zeros(0, np.int32)))
        tri = np.asarray(tri, np.int32).reshape(-1)
        out.append(f"finger {fidx[f]} {path} {len(pts)} {len(tri) // 3}")
        mesh_bin += np.ascontiguousarray(pts, np.float32).tobytes() + tri.tobytes()
    for k, f in pairs:
        out.append(f"pair {int(k)} {fidx[int(f)]}")
    out.append(f"rowcol {len(rows_sel)} {len(cols_sel)}")
    for k in range(S * O):
        cw = [int(i) for i, c in enumerate(cols_sel) if w[k % w.shape[0], c]] if k < w.shape[0] else []
        out.append("with %d %s" % (k, " ".join(map(str, cw))))
    out.append("qrow " + " ".join(str(i) for i in range(len(rows_sel))))
    # S3 v1(접촉 행렬을 우리 PhysX 접촉 보고로): 부분 행렬의 행·열 몸체 경로
    for i, pth in enumerate(st.get("rows", [])):
        out.append(f"rowpath {i} {pth}")
    for j, pth in enumerate(st.get("cols", [])):
        out.append(f"colpath {j} {pth}")
    out.append(f"episode_start {s1.get('episode_start_post', -1)} substeps {s1.get('substeps', 4)} steps {len(goals)} bddl {a.bddl}")
    open(os.path.join(d, "s3_setup.txt"), "w").write("\n".join(out) + "\n")
    open(os.path.join(d, "s3_meshes.bin"), "wb").write(bytes(mesh_bin))
    rb_bin = bytearray(struct.pack("<I", len(rows)))
    nrc = len(rows_sel) * len(cols_sel)
    for r in rows:
        rb_bin += struct.pack("<Q", int(r["post"]))
        rb_bin += np.ascontiguousarray(r["value"], np.uint8).tobytes() + np.ascontiguousarray(r["time"], np.float32).tobytes()
        has = "cm" in r and r["cm"].size == nrc
        cm = r["cm"] if has else np.zeros(nrc, np.uint8)
        ccm = r["ccm"] if has else np.zeros(nrc, np.uint8)
        rb_bin += np.ascontiguousarray(cm, np.uint8).tobytes() + np.ascontiguousarray(ccm, np.uint8).tobytes() + struct.pack("<B", int(has))
    open(os.path.join(d, "s3_rows.bin"), "wb").write(bytes(rb_bin))
    open(os.path.join(d, "s3_goal.txt"), "w").write("\n".join(goals) + "\n")
    print(f"S3 입력: 판 {S} 물체 {O}, 표식 {n_markers}, 손가락 {len(fingers)}, 쌍 {len(pairs)}, 접촉 부분행렬 {len(rows_sel)}x{len(cols_sel)}, "
          f"행 {len(rows)}, 스텝 {len(goals)} -> {d}")


if __name__ == "__main__":
    sys.exit(main())
