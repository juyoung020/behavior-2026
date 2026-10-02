# capture_slice.py 의 slice_events.json → test_slice_capture 가 읽는 npy.
# 행(반쪽마다): 자를 것 pos3 orn4 scale3 | 부분 bb_pos3 bb_orn4 bb_size3 | 공식 bb 중심3 방향4 bounding_box3 |
#              반쪽 nativeBB3 offset3 척도3 set 인자(bb_pos3 bb_orn4) | set 뒤 PhysX 자세 pos3 orn4
#   python3 slice_events_to_npy.py <기록 폴더>
import json
import os
import sys

import numpy as np

d = sys.argv[1]
r = json.load(open(os.path.join(d, "slice_events.json")))
sets = {s["name"]: s for s in r["setpose"]}
V = lambda v: [float(x) for x in v.strip("[]").replace(",", " ").split()] if isinstance(v, str) else list(v)  # 메타데이터는 float32 최단 표기 문자열
rows = []
for ev in r["slice"]:
    k = 0
    for src in ev["ins"]:
        for p in src["parts"].values():
            o = ev["outs"][k]
            k += 1
            s = sets.get(o["name"])
            if s is None:
                print("set 기록 없음", o["name"])
                continue
            rows.append(src["pos"] + src["orn"] + src["scale"] + V(p["bb_pos"]) + V(p["bb_orn"]) + V(p["bb_size"]) + o["bb_pos"] + o["bb_orn"]
                        + o["bbox"] + s["native_bbox"] + s["offset"] + s["scale"] + s["bb_pos"] + s["bb_orn"] + s["after_pos"] + s["after_orn"])
np.save(os.path.join(d, "slice_rows.npy"), np.array(rows, np.float32).reshape(-1, 53))
print("반쪽", len(rows), "자르기", len(r["slice"]), "다지기", len(r["dice"]))
