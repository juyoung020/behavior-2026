# 층 0(particles) 정답지: 뿌리기 표본 sampling_utils.sample_cuboid_on_object (도포기 _modify_particles, particle_modifier.py:1267 이 부르는 꼴)
# 를 공식 그대로 돌리고, 장면 질의만 가짜 신탁(회전된 상자들, double 로 교차 → float32 값)으로 바꾼다. 엔진 C++ 은 같은 광선 요청을 내는지
# (요청 비트 대조) 보고 신탁의 적중을 받아 같은 결과를 내야 한다. 중간값(평면·회전·법선 유사도)도 적는다.
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_spray_ref.py --out ~/engine-data/particles/spray
import argparse
import os
import pickle

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.sampling_utils as SU
import omnigibson.utils.transform_utils as T


class Hit:
    def __init__(self, pos, nrm, dist, body):
        self.position, self.normal, self.distance = pos, nrm, dist
        self.rigid_body, self.collision = body, body + "/collisions"


class Boxes:
    """장면: 회전된 상자들. raycast_all 은 상자 순서로 들어가는 면 하나씩 알린다 (PhysX 볼록 모양 광선과 같은 꼴)"""

    def __init__(self, rng):
        self.b = []
        for i in range(int(rng.integers(2, 6))):
            q = rng.normal(0, 1, 4)
            q /= np.linalg.norm(q)
            if rng.random() < 0.5:
                q = np.array([0, 0, np.sin(rng.uniform(0, 3)), np.cos(rng.uniform(0, 3))])
                q /= np.linalg.norm(q)
            x, y, z, w = q
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            self.b.append((f"/World/scene_0/obj{i}/base_link", rng.uniform(-0.5, 0.5, 3), R, rng.uniform(0.05, 0.4, 3)))
        self.calls = []

    def raycast_all(self, origin, dir, distance, reportFn):
        o, d = np.array(origin, float), np.array(dir, float)
        rep = []
        for path, c, R, h in self.b:
            lo, ld = R.T @ (o - c), R.T @ d
            t0, t1, ax, sg = -np.inf, np.inf, -1, 0
            ok = True
            for k in range(3):
                if abs(ld[k]) < 1e-12:
                    if abs(lo[k]) > h[k]:
                        ok = False
                    continue
                ta, tb = (-h[k] - lo[k]) / ld[k], (h[k] - lo[k]) / ld[k]
                s = -1
                if ta > tb:
                    ta, tb, s = tb, ta, 1
                if ta > t0:
                    t0, ax, sg = ta, k, s
                t1 = min(t1, tb)
            if not ok or t0 > t1 or t0 < 0 or t0 > distance:
                continue
            n = np.zeros(3)
            n[ax] = sg
            p = o + d * t0
            rep.append(Hit(tuple(float(np.float32(v)) for v in p), tuple(float(np.float32(v)) for v in R @ n), float(np.float32(t0)), path))
        self.calls.append((list(origin), list(dir), float(distance), [(h.position, h.normal, h.distance, h.rigid_body) for h in rep]))
        for h in rep:
            reportFn(h)
        return len(rep) > 0


class FakeSim:
    psqi = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(43)
    for shp in [(4,), (3, 4), (5, 4), (2, 5), (3, 6)]:
        T.quat2mat(th.rand(*shp))
    og.sim = FakeSim()
    rec = {"fit": [], "sim": [], "rot": []}
    of, os_, orr = SU.fit_plane, SU.check_normal_similarity, SU.compute_rotation_from_grid_sample

    def fit_w(points, log):
        r = of(points, log)
        rec["fit"].append((points.clone(), None if r[0] is None else (r[0].clone(), r[1].clone())))
        return r

    def sim_w(c, h, tol, log):
        r = os_(c, h, tol, log)
        rec["sim"].append((c.clone(), h.clone(), bool(r)))
        return r

    def rot_w(*args):
        r = orr(*args)
        rec["rot"].append(None if r is None else r.clone())
        return r

    SU.fit_plane, SU.check_normal_similarity, SU.compute_rotation_from_grid_sample = fit_w, sim_w, rot_w
    cases = []
    th.manual_seed(7)
    for c in range(a.n):
        scene = Boxes(rng)
        og.sim.psqi = scene
        n = 2
        # 원뿔 꼭짓점(분무기) 과 상자 쪽 끝점
        apex = rng.uniform(-1, 1, 3) + np.array([0, 0, 1.0])
        tgt = scene.b[int(rng.integers(len(scene.b)))][1] + rng.normal(0, 0.1, (n, 3))
        start = th.tensor(np.repeat(apex[None], n, 0), dtype=th.float32).reshape(n, 1, 3)
        end = th.tensor(apex + (tgt - apex) * 1.5, dtype=th.float32).reshape(n, 1, 3)
        dims = th.tensor(rng.uniform(0.01, 0.35, (n, 3)) if rng.random() < 0.6 else rng.uniform(0.2, 1.4, (n, 3)), dtype=th.float32)
        rng_state = th.get_rng_state().numpy().copy()
        for k in rec:
            rec[k] = []
        ignore = [type("O", (), {"links": {"b": type("L", (), {"prim_path": "/World/scene_0/atomizer/base_link"})()}})()]
        res = SU.sample_cuboid_on_object(obj=None, start_points=start, end_points=end, cuboid_dimensions=dims, ignore_objs=ignore,
                                         hit_proportion=0.0, cuboid_bottom_padding=SU.m.DEFAULT_CUBOID_BOTTOM_PADDING,
                                         undo_cuboid_bottom_padding=True, verify_cuboid_empty=False)
        out = [(None if r[0] is None else (r[0].numpy(), r[1].numpy(), r[2].numpy(), r[3])) for r in res]
        cases.append(dict(start=start.numpy(), end=end.numpy(), dims=dims.numpy(), rng=rng_state, rays=scene.calls,
                          boxes=[b[0] for b in scene.b], out=out,
                          fit=[(p.numpy(), None if r is None else (r[0].numpy(), r[1].numpy())) for p, r in rec["fit"]],
                          sim=[(x.numpy(), y.numpy(), z) for x, y, z in rec["sim"]],
                          rot=[None if r is None else r.numpy() for r in rec["rot"]]))
    with open(os.path.join(a.out, "cases.pkl"), "wb") as f:
        pickle.dump(cases, f)
    # C++ 용 평탄화
    ray_off, rays, hit_off, hits, outs, se = [0], [], [0], [], [], []
    for cs in cases:
        for o, d, dist, hl in cs["rays"]:
            rays.append(list(o) + list(d) + [dist])
            for p, nm, dd, body in hl:
                hits.append(list(p) + list(nm) + [dd, cs["boxes"].index(body)])
            hit_off.append(len(hits))
        ray_off.append(len(rays))
        for s in range(2):
            o = cs["out"][s]
            outs.append([0] * 11 + [-1] if o is None else [1] + list(o[0]) + list(o[1]) + list(o[2]) + [cs["boxes"].index(o[3])])
        se.append(np.concatenate([cs["start"].ravel(), cs["end"].ravel(), cs["dims"].ravel()]))
    np.save(os.path.join(a.out, "rng.npy"), np.stack([cs["rng"] for cs in cases]))
    np.save(os.path.join(a.out, "se.npy"), np.stack(se).astype(np.float32))
    np.save(os.path.join(a.out, "ray_off.npy"), np.array(ray_off, np.int64))
    np.save(os.path.join(a.out, "rays.npy"), np.array(rays, np.float64))
    np.save(os.path.join(a.out, "hit_off.npy"), np.array(hit_off, np.int64))
    np.save(os.path.join(a.out, "hits.npy"), np.array(hits, np.float64).reshape(-1, 8))
    np.save(os.path.join(a.out, "outs.npy"), np.array(outs, np.float64))
    ok = sum(o is not None for cs in cases for o in cs["out"])
    print("done", a.n, "표본 성공", ok, "/", 2 * a.n, "광선", sum(len(cs["rays"]) for cs in cases))


if __name__ == "__main__":
    main()
