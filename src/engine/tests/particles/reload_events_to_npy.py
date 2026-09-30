# harvest_reload.json (harvest_spawn.py 의 재적재 기록: dump 순간 원점 자세 → load 가 set 에 준 값) → test_reload 가 읽는 npy
#   python3 reload_events_to_npy.py <수확 기록 폴더> <출력 폴더>;  ./test_reload <출력 폴더>
import json
import os
import sys

import numpy as np

d, o = sys.argv[1], sys.argv[2]
ev = [e for e in json.load(open(os.path.join(d, "harvest_reload.json"))) if "load_set" in e]
os.makedirs(o, exist_ok=True)
for e in ev:
    assert len(e["dump_tfs"]) == len(e["load_set"]), (e["system"], e["step"])
    print("재적재", e["system"], "dump 스텝", e["step"], "load 스텝", e["load_step"], "입자", len(e["dump_tfs"]), "offset", e["offset"])
np.save(os.path.join(o, "n.npy"), np.array([len(e["dump_tfs"]) for e in ev], np.int32))
np.save(os.path.join(o, "tfs.npy"), np.ascontiguousarray(np.array([t for e in ev for t in e["dump_tfs"]], np.float32).reshape(-1, 7)))
np.save(os.path.join(o, "off.npy"), np.array([e["offset"] for e in ev], np.float32).reshape(-1, 3))
np.save(os.path.join(o, "scene_pose.npy"), np.array([[e["pose"], e["pose_inv"]] for e in ev], np.float32).reshape(-1, 2, 4, 4))
np.save(os.path.join(o, "out.npy"), np.ascontiguousarray(np.array([t for e in ev for t in e["load_set"]], np.float32).reshape(-1, 7)))
print("사건", len(ev))
