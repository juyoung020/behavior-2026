# 수확 기록 → test_spawn_exec 입력 (한 줄 = 넣을 행위자 하나, 공백 구분, float 는 float32 왕복 정확 %.9g)
#   python3 spawn_to_txt.py <수확 기록 폴더>   (spawn_table.json·harvest_map.json·harvest_assets.json·harvest_dice.json 을 읽음)
# 줄: S <틀 파일> <틀 안 행위자> <원본 pos3 orn4 scale3> <부분 bb_pos3 bb_orn4 bb_size3> <nativeBB3 offset3>
#     D <틀 파일> <틀 안 행위자> <원점 pos3 orn4>   (다진 입자: 공식 generate_particles 가 넣은 원점, test_dice_capture 로 이미 비트 동일)
import json
import os
import sys

import numpy as np

d = sys.argv[1]
st = json.load(open(os.path.join(d, "spawn_table.json")))
hm = json.load(open(os.path.join(d, "harvest_map.json")))
assets = json.load(open(os.path.join(d, "harvest_assets.json"))) if os.path.exists(os.path.join(d, "harvest_assets.json")) else {}
dice = json.load(open(os.path.join(d, "harvest_dice.json"))) if os.path.exists(os.path.join(d, "harvest_dice.json")) else []
F = lambda v: " ".join("%.9g" % np.float32(x) for x in v)
V = lambda v: [float(x) for x in v.strip("[]").replace(",", " ").split()] if isinstance(v, str) else list(v)
src_of = {e["src"]: e for e in hm if e["rule"] == "SlicingRule"}
out = []
for s in st["slices"]:
    e = src_of[s["src"]]
    for p in s["parts"]:
        meta = e["parts"][p["part"]]
        a = assets[p["name"]]
        out.append(f"S {p['template']} {p['actor']} {F(e['src_pos'])} {F(e['src_orn'])} {F(e['src_scale'])} "
                   f"{F(V(meta['bb_pos']))} {F(V(meta['bb_orn']))} {F(V(meta['bb_size']))} {F(a['native_bbox'])} {F(a['offset'])}")
gi = 0
for dd in st["dices"]:
    while gi < len(dice) and dice[gi]["system"] != dd["system"]:
        gi += 1
    g = dice[gi]
    gi += 1
    n0 = g["n_before"]
    for k, act in enumerate(dd["actors"]):
        fr = g["frames"][n0 + k]
        out.append(f"D {dd['template']} {act} {F(fr[:3])} {F(fr[3:])}")
open(os.path.join(d, "spawn_rows.txt"), "w").write("\n".join(out) + "\n")
print("넣을 행위자", len(out))
