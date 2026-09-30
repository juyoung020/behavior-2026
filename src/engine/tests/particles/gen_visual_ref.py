# 층 0(particles) 정답지: 시각 입자(MacroVisualParticleSystem) 세계 위치 + 입자 제거기(ParticleRemover) 한 스텝 갱신.
# 공식 함수를 가짜 물체에 붙여 그대로 부른다 (Kit·GPU 안 씀, Linux 평가기처럼 T.* 는 torch.compile).
#   - 국소 행렬: MacroVisualParticleSystem._modify_batch_particles_position_orientation(local=True)  (macro_particle_system.py:749)
#   - 세계 위치: MacroVisualParticleSystem._compute_batch_particles_position_orientation(local=False) (:664)
#   - 제거기 갱신: ParticleModifier._update (particle_modifier.py:697) → ParticleRemover._modify_particles (:927)
#       ADJACENCY: RigidPrim.visual_aabb (rigid_prim.py:572, T.transform_points) + 여유 0.02 (_check_in_mesh :524)
#       PROJECTION: RigidPrim.check_points_in_volume → GeomPrim.check_points_in_volume (geom_prim.py:229, th.linalg.inv)
#                   → check_points_in_cylinder / cone / cube / sphere (utils/geometry_utils.py)
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_visual_ref.py --out ~/engine-data/particles/visual
import argparse
import os

import numpy as np
import torch as th

import omnigibson.object_states.particle_modifier as PM
import omnigibson.systems.macro_particle_system as MPS
import omnigibson.utils.transform_utils as T
from omnigibson.object_states.saturated import ModifiedParticles, Saturated
from omnigibson.object_states.toggle import ToggledOn
from omnigibson.prims.geom_prim import GeomPrim
from omnigibson.prims.rigid_prim import RigidPrim
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.constants import ParticleModifyMethod, PrimType

Vis = MPS.MacroVisualParticleSystem
PR = PM.ParticleRemover
MESH_KIND = {0: None, 1: "Cylinder", 2: "Cone", 3: "Cube", 4: "Sphere"}


class FakeLink:
    """RigidPrim 대역: 공식 property/메서드를 그대로 붙이고 자세만 바꿔 끼운다"""

    transform_local_points_to_world = XFormPrim.transform_local_points_to_world
    visual_boundary_points_world = RigidPrim.__dict__["visual_boundary_points_world"]
    visual_aabb = RigidPrim.__dict__["visual_aabb"]
    check_points_in_volume = RigidPrim.check_points_in_volume

    def __init__(self, hull=None, meshes=None):
        self.scaled_transform = None
        self._hull = hull
        self.visual_meshes = meshes or {}

    @property
    def visual_boundary_points_local(self):
        return self._hull


class FakeMesh:
    """GeomPrim 대역 (투영 메시: 원기둥·원뿔·상자·구)"""

    check_points_in_volume = GeomPrim.check_points_in_volume
    check_local_points_in_volume = GeomPrim.check_local_points_in_volume

    def __init__(self, kind, attrs):
        self._mesh_type = kind
        self._attrs = attrs
        self.scaled_transform = None

    def get_attribute(self, k):
        return self._attrs[k]


class FakeSys:
    _compute_batch_particles_position_orientation = Vis._compute_batch_particles_position_orientation
    get_particles_position_orientation = Vis.get_particles_position_orientation
    _modify_batch_particles_position_orientation = Vis._modify_batch_particles_position_orientation
    remove_particles = MPS.MacroParticleSystem.remove_particles
    n_particles = MPS.MacroParticleSystem.__dict__["n_particles"]
    initialized = True
    name = "dust"

    def __init__(self):
        self.particles = {}
        self._particles_info = {}
        self._particles_local_mat = {}
        self._group_particles = {}

    def _is_cloth_obj(self, obj):
        return False

    def _modify_particle_local_mat(self, name, mat):
        self._particles_local_mat[name] = mat

    def remove_particle_by_name(self, name):  # macro_particle_system.py:255 + :491 (prim 삭제만 뺌)
        assert name in self.particles
        self.particles.pop(name)
        g = self._particles_info[name]["group"]
        self._group_particles[g].pop(name)
        self._particles_local_mat.pop(name)
        self._particles_info.pop(name)


class FakeScene:
    def __init__(self, sys):
        self.sys = sys
        self.active_systems = {sys.name: sys}

    def is_visual_particle_system(self, system_name):
        return True

    def is_physical_particle_system(self, system_name):
        return False

    def get_system(self, name, force_init=True):
        return self.sys


class FakeModified:  # saturated.py:14 ModifiedParticles 와 같은 뜻
    def __init__(self):
        self.particle_counts = {}

    def get_value(self, system):
        return self.particle_counts.get(system.name, 0)

    def set_value(self, system, v):
        if v == 0 and system.name in self.particle_counts:
            self.particle_counts.pop(system.name)
        else:
            self.particle_counts[system.name] = v


class FakeSaturated:  # saturated.py:150 _get_value (count == limit)
    def __init__(self, obj, limit):
        self.obj, self.limit = obj, limit

    def get_value(self, system):
        c = self.obj.states[ModifiedParticles].get_value(system)
        assert c <= self.limit
        return c == self.limit


class FakeToggle:
    def __init__(self):
        self.v = False

    def get_value(self):
        return self.v


class FakeObj:
    prim_type = PrimType.RIGID

    def __init__(self, scene, limit):
        self.scene = scene
        self.states = {ModifiedParticles: FakeModified(), ToggledOn: FakeToggle()}
        self.states[Saturated] = FakeSaturated(self, limit)

    def state_updated(self):
        pass


class FakeRemover:
    _check_in_mesh = PR._check_in_mesh
    _modify_particles = PR._modify_particles
    _update = PM.ParticleModifier._update
    _generate_condition = PM.ParticleModifier._generate_condition
    _generate_limit_condition = PM.ParticleModifier._generate_limit_condition
    check_conditions_for_system = PM.ParticleModifier.check_conditions_for_system
    supports_system = PM.ParticleModifier.supports_system
    conditions = PM.ParticleModifier.__dict__["conditions"]
    systems_to_check = PM.ParticleModifier.__dict__["systems_to_check"]
    n_steps_per_modification = PR.__dict__["n_steps_per_modification"]

    def __init__(self, obj, link, method, limit, needs_toggle):
        self.obj, self.link, self.method = obj, link, method
        self.visual_particle_modification_limit = limit
        self.physical_particle_modification_limit = 400
        self._current_step = 0
        conds = [self._generate_condition(PM.ParticleModifyCondition.TOGGLEDON, True)] if needs_toggle else []
        self._conditions = {"dust": conds + [self._generate_limit_condition("dust")]}


def rand_quat(rng, n):
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # 저장값처럼 float32 로 반올림 (정규화가 끝비트만큼 어긋난 입력)
    return q.astype(np.float32)


def scaled_tf(rng, pos, scale, kind):
    """Fabric 세계 행렬(double, 행 벡터 S*R*T) → float32 → .T  (usd_utils.py:2168 get_world_pose_with_scale 와 같은 모양)"""
    if kind == "ident":
        return th.eye(4, dtype=th.float32)
    if kind == "trans":
        M = np.eye(4)
        M[3, :3] = pos
    else:
        x, y, z, w = rand_quat(rng, 1)[0].astype(np.float64)
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (z * x - y * w)],
                      [2 * (x * y - z * w), 1 - 2 * (z * z + x * x), 2 * (y * z + x * w)],
                      [2 * (z * x + y * w), 2 * (y * z - x * w), 1 - 2 * (y * y + x * x)]])
        M = np.eye(4)
        M[:3, :3] = np.diag(scale) @ R
        M[3, :3] = pos
    return th.tensor(M, dtype=th.float32).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    th.manual_seed(0)
    tot_rm = 0
    for c in range(a.cases):
        d = os.path.join(a.out, f"case_{c:03d}")
        os.makedirs(d, exist_ok=True)
        L = int(rng.integers(1, 4))  # 입자가 붙은 링크 수 (링크 = 무리 하나)
        R = int(rng.integers(1, 4))
        n_per = rng.integers(1, 41, L)
        N = int(n_per.sum())
        H = 8
        sys = FakeSys()
        scene = FakeScene(sys)
        p_link = np.repeat(np.arange(L), n_per).astype(np.int32)
        order = rng.permutation(N) if rng.random() < 0.5 else np.arange(N)  # 시스템 순서가 무리 순서와 섞인 경우
        p_link = p_link[order]
        lpos = (rng.uniform(-0.3, 0.3, (N, 3))).astype(np.float32)
        lquat = rand_quat(rng, N)
        links = [FakeLink() for _ in range(L)]
        names = [f"dustParticle{i}" for i in range(N)]
        for i, nm in enumerate(names):
            g = f"obj{p_link[i]}"
            sys.particles[nm] = None
            sys._group_particles.setdefault(g, {})[nm] = None
            sys._particles_info[nm] = dict(obj=None, link=links[p_link[i]], group=g)
        # 국소 행렬: 무리마다 set_group_particles_local_pose (_load_state 경로)
        for g in sorted(sys._group_particles):
            idx = [names.index(nm) for nm in sys._group_particles[g]]
            sys._modify_batch_particles_position_orientation(
                particles=sys._group_particles[g], positions=th.from_numpy(lpos[idx]),
                orientations=th.from_numpy(lquat[idx]), local=True)
        local_mat = np.stack([sys._particles_local_mat[nm].numpy() for nm in names]).astype(np.float32)

        # 제거기
        meth = rng.integers(0, 5, R).astype(np.int32)
        meth[rng.random(R) < 0.5] = 0
        limit = np.where(rng.random(R) < 0.4, rng.integers(1, 12, R), 200).astype(np.int32)
        need_tog = (meth != 0) & (rng.random(R) < 0.8)
        hull = rng.uniform(-0.15, 0.15, (R, H, 3)).astype(np.float32)
        mattr = np.zeros((R, 3), np.float64)  # radius height size (USD double → 파이썬 float)
        rems = []
        for r in range(R):
            obj = FakeObj(scene, int(limit[r]))
            if meth[r] == 0:
                link = FakeLink(hull=th.from_numpy(hull[r]))
                method = ParticleModifyMethod.ADJACENCY
            else:
                kind = MESH_KIND[int(meth[r])]
                mattr[r] = [0.5, 1.0, 1.0] if rng.random() < 0.5 else rng.uniform(0.2, 1.2, 3)
                mesh = FakeMesh(kind, {"radius": float(mattr[r, 0]), "height": float(mattr[r, 1]), "size": float(mattr[r, 2])})
                link = FakeLink(meshes={"mesh_0": mesh})
                method = ParticleModifyMethod.PROJECTION
            rems.append(FakeRemover(obj, link, method, int(limit[r]), bool(need_tog[r])))

        S = a.steps
        link_tf = np.zeros((S, L, 4, 4), np.float32)
        rem_tf = np.zeros((S, R, 4, 4), np.float32)
        toggle = np.zeros((S, R), np.uint8)
        world = np.full((S, N, 3), np.nan, np.float32)
        aabb = np.full((S, R, 2, 3), np.nan, np.float32)
        alive_after = np.zeros((S, R, N), np.uint8)
        count_after = np.zeros((S, R), np.int32)
        proj_pts = np.zeros((S, R, 96, 3), np.float32)
        proj_inv = np.zeros((S, R, 4, 4), np.float32)
        proj_loc = np.zeros((S, R, 96, 3), np.float32)
        proj_in = np.zeros((S, R, 96), np.uint8)
        centers = rng.uniform(-1, 1, (L, 3))
        for s in range(S):
            for l in range(L):
                kind = rng.choice(["rot", "rot", "rot", "trans", "ident"], p=[0.3, 0.3, 0.2, 0.15, 0.05])
                tf = scaled_tf(rng, centers[l] + rng.normal(0, 0.02, 3), rng.uniform(0.5, 2.0, 3), kind)
                links[l].scaled_transform = tf
                link_tf[s, l] = tf.numpy()
            alive = [nm for nm in names if nm in sys.particles]
            if alive:
                wp = sys.get_particles_position_orientation()[0].numpy()
                for k, nm in enumerate(alive):
                    world[s, names.index(nm)] = wp[k]
            for r, rm in enumerate(rems):
                # 제거기를 살아 있는 입자 근처로 (가끔은 경계 몇 ulp 에 맞춤)
                tgt = world[s][~np.isnan(world[s][:, 0])]
                base = tgt[rng.integers(len(tgt))] if len(tgt) else rng.uniform(-1, 1, 3)
                if meth[r] == 0 and len(tgt) and rng.random() < 0.3:
                    ax = int(rng.integers(3))
                    side = rng.integers(2)
                    hmin, hmax = hull[r, :, ax].min(), hull[r, :, ax].max()
                    pos = base.astype(np.float64) + rng.normal(0, 0.05, 3)
                    t = np.float32(base[ax]) + np.float32(0.02) - np.float32(hmin) if side == 0 else \
                        np.float32(base[ax]) - np.float32(0.02) - np.float32(hmax)
                    t = np.nextafter(t, np.float32(np.inf) * (1 if rng.random() < 0.5 else -1), dtype=np.float32) \
                        if rng.random() < 0.7 else t
                    for _ in range(int(rng.integers(0, 3))):
                        t = np.nextafter(t, np.float32(np.inf), dtype=np.float32)
                    pos[ax] = float(t)
                    tf = scaled_tf(rng, pos, None, "trans")
                else:
                    kind = rng.choice(["rot", "trans", "ident"], p=[0.7, 0.25, 0.05])
                    sc = rng.uniform(0.05, 0.4, 3) if meth[r] else rng.uniform(0.5, 2.0, 3)
                    if meth[r] in (3, 4) or rng.random() < 0.5:
                        sc = np.full(3, sc[0])
                    tf = scaled_tf(rng, base + rng.normal(0, 0.08, 3), sc, kind)
                if meth[r] == 0:
                    rm.link.scaled_transform = tf
                else:
                    rm.link.visual_meshes["mesh_0"].scaled_transform = tf
                rem_tf[s, r] = tf.numpy()
                rm.obj.states[ToggledOn].v = bool(rng.random() < 0.7)
                toggle[s, r] = rm.obj.states[ToggledOn].v
                if meth[r] == 0:
                    lo, hi = rm.link.visual_aabb
                    aabb[s, r, 0], aabb[s, r, 1] = lo.numpy(), hi.numpy()
                else:
                    # 투영 부피 판정만 따로: 살아 있는 입자 + 부피 경계 근처 점 (geom_prim.py:229 식 그대로)
                    mesh = rm.link.visual_meshes["mesh_0"]
                    P = proj_pts.shape[2]
                    alive_w = world[s][~np.isnan(world[s][:, 0])][:P]
                    loc = rng.uniform(-0.75, 0.75, (P, 3))
                    loc[:, 2] = rng.uniform(-0.6, 0.6, P)
                    pw = (np.c_[loc, np.ones(P)] @ tf.numpy().astype(np.float64).T)[:, :3].astype(np.float32)
                    pw[: len(alive_w)] = alive_w
                    pt = th.from_numpy(pw)
                    inv = th.linalg.inv(mesh.scaled_transform)
                    hom = th.cat((pt, th.ones((P, 1))), dim=1)
                    proj_pts[s, r], proj_inv[s, r] = pw, inv.numpy()
                    proj_loc[s, r] = (hom @ inv.T)[:, :3].numpy()
                    proj_in[s, r] = mesh.check_points_in_volume(pt).numpy()
                before = len(sys.particles)
                rm._update()  # 공식 ParticleModifier._update
                tot_rm += before - len(sys.particles)
                alive_after[s, r] = [nm in sys.particles for nm in names]
                count_after[s, r] = rm.obj.states[ModifiedParticles].get_value(sys)
        for k, v in dict(p_link=p_link, lpos=lpos, lquat=lquat, local_mat=local_mat, link_tf=link_tf, rem_meth=meth,
                         rem_limit=limit, rem_need_toggle=need_tog.astype(np.uint8), rem_hull=hull, rem_attr=mattr,
                         rem_tf=rem_tf, toggle=toggle, world=world, aabb=aabb, alive_after=alive_after,
                         count_after=count_after, proj_pts=proj_pts, proj_inv=proj_inv, proj_loc=proj_loc,
                         proj_in=proj_in).items():
            np.save(os.path.join(d, k + ".npy"), v)
    # 부피 판정 함수만 (utils/geometry_utils.py): 노름·높이 경계 몇 ulp 점. 원기둥·원뿔의 th.norm(p[:, :-1]) 과 구의 th.norm(p) 순서를 가린다
    import omnigibson.utils.geometry_utils as GU
    vd = os.path.join(a.out, "volume")
    os.makedirs(vd, exist_ok=True)
    M = 200000
    kinds = rng.integers(1, 5, M).astype(np.int32)
    attr = np.where(rng.random((M, 3)) < 0.3, np.array([0.5, 1.0, 1.0]), rng.uniform(0.05, 1.5, (M, 3)))
    pts = rng.uniform(-1, 1, (M, 3))
    ang = rng.uniform(0, 2 * np.pi, M)
    th3 = rng.uniform(0, np.pi, M)
    for i in range(M):
        r, h = attr[i, 0], attr[i, 1]
        z = rng.uniform(-h / 2, h / 2)
        if kinds[i] == 1:
            rr = r
        elif kinds[i] == 2:
            rr = r * (1 - (z + h / 2) / h)
        else:
            rr = r
        if kinds[i] == 4:
            pts[i] = rr * np.array([np.sin(th3[i]) * np.cos(ang[i]), np.sin(th3[i]) * np.sin(ang[i]), np.cos(th3[i])])
        elif kinds[i] == 3:
            pts[i] = rng.uniform(-attr[i, 2] / 2, attr[i, 2] / 2, 3)
            pts[i, rng.integers(3)] = attr[i, 2] / 2 * rng.choice([-1, 1])
        else:
            pts[i] = [rr * np.cos(ang[i]), rr * np.sin(ang[i]), z if rng.random() < 0.8 else h / 2 * rng.choice([-1, 1])]
    pts = pts.astype(np.float32)
    k = rng.integers(-4, 5, (M, 3))
    for _ in range(4):  # 몇 ulp 흔들기
        m = k > 0
        pts = np.where(m, np.nextafter(pts, np.float32(np.inf)), pts)
        k = k - m
        m = k < 0
        pts = np.where(m, np.nextafter(pts, np.float32(-np.inf)), pts)
        k = k + m
    res = np.zeros(M, np.uint8)
    # 점마다 크기 1 배치로 부른다 (속성이 점마다 다름). 행마다 독립인지는 아래 큰 배치 하나로 확인
    P = th.from_numpy(pts)
    for i in range(M):
        kd, (r, h, sz) = kinds[i], attr[i]
        p = P[i : i + 1]
        if kd == 1:
            res[i] = GU.check_points_in_cylinder([r, h], p)[0]
        elif kd == 2:
            res[i] = GU.check_points_in_cone([r, h], p)[0]
        elif kd == 3:
            res[i] = GU.check_points_in_cube(sz, p)[0]
        else:
            res[i] = GU.check_points_in_sphere(r, p)[0]
    Pb = th.from_numpy(pts[:4096])  # 큰 배치 하나도 (배치 크기에 따라 노름 커널이 다른지)
    res_b = np.array(GU.check_points_in_cylinder([float(attr[0, 0]), float(attr[0, 1])], Pb), np.uint8)
    for k2, v in dict(kinds=kinds, attr=attr, pts=pts, inside=res, batch_inside=res_b).items():
        np.save(os.path.join(vd, k2 + ".npy"), v)
    print("done cases", a.cases, "removed", tot_rm, "volume pts", M, "inside", int(res.sum()))


if __name__ == "__main__":
    main()
