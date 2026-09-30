# harvest_dice.json (harvest_spawn.py 의 다지기 기록) → test_dice_capture 가 읽는 npy + scipy Delaunay (메시 점에서, 공식과 같은 scipy)
#   python3 dice_events_to_npy.py <수확 기록 폴더>
import json
import os
import sys

import numpy as np
from scipy.spatial import Delaunay

d = sys.argv[1]
ev = json.load(open(os.path.join(d, "harvest_dice.json")))
o = os.path.join(d, "dice_npy")
os.makedirs(o, exist_ok=True)
n = 0
for i, e in enumerate(ev):
    if "frames" not in e or any(m["type"] != "Mesh" for m in e["meshes"]):
        print("건너뜀", i, e["system"], [m["type"] for m in e["meshes"]])
        continue
    c = os.path.join(o, f"ev_{n:03d}")
    os.makedirs(c, exist_ok=True)
    np.save(os.path.join(c, "lo.npy"), np.array(e["lo"], np.float32))
    np.save(os.path.join(c, "hi.npy"), np.array(e["hi"], np.float32))
    np.save(os.path.join(c, "r.npy"), np.array([e["radius"]], np.float32))
    np.save(os.path.join(c, "off.npy"), np.array(e["offset"], np.float32))
    np.save(os.path.join(c, "rng.npy"), np.array(e["rng"], np.uint8))
    np.save(os.path.join(c, "nbefore.npy"), np.array([e["n_before"]], np.int32))
    np.save(os.path.join(c, "centers.npy"), np.ascontiguousarray(np.array(e["centers"], np.float32).reshape(-1, 3)))
    np.save(os.path.join(c, "frames.npy"), np.ascontiguousarray(np.array(e["frames"], np.float32).reshape(-1, 7)))
    np.save(os.path.join(c, "mesh_tf.npy"), np.ascontiguousarray(np.array([m["tf"] for m in e["meshes"]], np.float32)))
    for k, m in enumerate(e["meshes"]):
        tri = Delaunay(np.array(m["points"], np.float32).astype(np.float64))
        np.save(os.path.join(c, f"dl{k}_transform.npy"), np.ascontiguousarray(tri.transform.reshape(-1, 12)))
        np.save(os.path.join(c, f"dl{k}_neighbors.npy"), np.ascontiguousarray(tri.neighbors.astype(np.int32)))
        np.save(os.path.join(c, f"dl{k}_equations.npy"), np.ascontiguousarray(tri.equations))
        np.save(os.path.join(c, f"dl{k}_misc.npy"), np.r_[tri.paraboloid_scale, tri.paraboloid_shift, tri.min_bound, tri.max_bound])
    n += 1
print("다지기 사건", n)
