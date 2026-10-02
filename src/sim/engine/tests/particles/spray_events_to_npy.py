# capture_spray.py 의 spray_events.json → test_spray_capture 가 읽는 npy (같은 폴더에)
#   python3 spray_events_to_npy.py ~/engine-data/particles/spray_real/<과제>_<인스턴스>
import json
import os
import sys

import numpy as np

d = sys.argv[1]
ev = json.load(open(os.path.join(d, "spray_events.json")))
bodies = sorted({r[2][3] for e in ev for r in e["rays"] if r[2] is not None} | {n["link"] for e in ev for n in e["new"]})
E = len(ev)
rng = np.stack([np.array(e["rng"], np.uint8) for e in ev]) if E else np.zeros((0, 5056), np.uint8)
f = lambda k, w: np.array([e.get(k, [0] * w) for e in ev], np.float32).reshape(E, w)
misc = np.concatenate([f("link_p", 3), f("link_q", 4), f("obj_scale", 3), f("ext", 3), f("tmpl", 3), f("grp_min", 3), f("grp_max", 3),
                       np.array([[float(e["rel"]), float(e["ptype"] == "Cone")] for e in ev], np.float32).reshape(E, 2)], 1)
ray_off, rays = [0], []
new_off, new = [0], []
for e in ev:
    for s, t, h in e["rays"]:
        rays.append(list(s) + list(t) + ([1] + list(h[0]) + list(h[1]) + [h[2], bodies.index(h[3])] if h else [0] * 9))
    ray_off.append(len(rays))
    for n in e["new"]:
        new.append(sum(n["lm"], []) + sum(n["link_tf"], []) + [bodies.index(n["link"])])
    new_off.append(len(new))
np.save(os.path.join(d, "ev_rng.npy"), rng)
np.save(os.path.join(d, "ev_misc.npy"), misc)
np.save(os.path.join(d, "ev_ray_off.npy"), np.array(ray_off, np.int64))
np.save(os.path.join(d, "ev_rays.npy"), np.array(rays, np.float64).reshape(-1, 15))
np.save(os.path.join(d, "ev_new_off.npy"), np.array(new_off, np.int64))
np.save(os.path.join(d, "ev_new.npy"), np.array(new, np.float32).reshape(-1, 33))
print("사건", E, "광선", len(rays), "새 입자", len(new), "몸체", len(bodies))
