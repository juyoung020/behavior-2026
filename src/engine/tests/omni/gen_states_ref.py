# 층 0(omni) 정답지: OmniGibson 물체 상태의 warp 커널(공식 코드, CUDA)을 물리 없이 가짜 장면 입력으로 직접 돌려 입·출력을 적는다.
# 평가기와 같은 커널을 같은 방식(wp.launch, device="cuda")으로 부른다. GPU 잠금 규칙을 지킬 것(.gpu_lock).
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_states_ref.py --out ~/engine-data/omni/states --frames 40
# 커널 (원본 위치):
#   utils/usd_utils.py  _poses_to_matrices_kernel, _aabb_reduce_kernel, _aabb_baselink_fallback_kernel,
#                       _update_contact_matrices_kernel, _update_body_transforms_kernel
#   object_states/aabb.py _aabb_init_kernel, open_state.py _open_update_kernel, touching.py (3), inside.py (5),
#   adjacency.py (2), on_top.py, under.py, next_to.py
# 출력: frame_XXXX/<이름>.npy (입력·출력 배열 그대로). C++ 시험 test_states 가 읽는다.
import argparse
import math
import os

import numpy as np
import torch as th
import warp as wp
from scipy.spatial import ConvexHull

import omnigibson.object_states.adjacency as ADJ
import omnigibson.object_states.aabb as AABBM
import omnigibson.object_states.inside as INS
import omnigibson.object_states.next_to as NXT
import omnigibson.object_states.on_top as ONT
import omnigibson.object_states.open_state as OPN
import omnigibson.object_states.touching as TCH
import omnigibson.object_states.toggle as TOG
import omnigibson.object_states.cooked as CKD
import omnigibson.object_states.frozen as FRZ
import omnigibson.object_states.heat_source_or_sink as HSS
import omnigibson.object_states.max_temperature as MXT
import omnigibson.object_states.on_fire as ONF
import omnigibson.object_states.temperature as TMP
import omnigibson.object_states.under as UND
import omnigibson.utils.usd_utils as UU

D = "cuda"


def f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def rand_quat(rng, axis_aligned_p=0.2, tilt=0.4):
    if rng.random() < axis_aligned_p:
        q = np.array([0.0, 0.0, 0.0, 1.0])
        if rng.random() < 0.5:  # z 축 90도 단위
            k = rng.integers(0, 4)
            q = np.array([0.0, 0.0, math.sin(k * math.pi / 4), math.cos(k * math.pi / 4)])
    else:
        yaw = rng.uniform(-math.pi, math.pi)
        ax = rng.standard_normal(2) * tilt
        q = np.array([ax[0], ax[1], math.sin(yaw / 2), math.cos(yaw / 2)])
        q = q / np.linalg.norm(q)
    q = f32(q)
    if rng.random() < 0.3:  # PhysX 가 주는 것처럼 단위에서 끝비트 벗어남
        q = f32(q * np.float32(1.0 + rng.uniform(-3e-7, 3e-7)))
    return q


def rand_hull(rng, half):
    """무작위 볼록 껍질 (꼭짓점, 삼각형 인덱스)."""
    while True:
        n = int(rng.integers(8, 24))
        pts = rng.uniform(-1, 1, (n, 3)) * half
        if rng.random() < 0.3:  # 상자
            pts = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * half
        try:
            h = ConvexHull(pts)
        except Exception:
            continue
        used = np.unique(h.simplices)
        remap = -np.ones(len(pts), int)
        remap[used] = np.arange(len(used))
        return f32(pts[used]), np.ascontiguousarray(remap[h.simplices].astype(np.int32))


def save(dirp, **kw):
    for k, v in kw.items():
        np.save(os.path.join(dirp, k + ".npy"), np.ascontiguousarray(v))


def wpa(x, dtype=None):
    return wp.array(np.ascontiguousarray(x), dtype=dtype, device=D) if dtype is not None else wp.array(
        np.ascontiguousarray(x), device=D)


def frame(rng, out):
    os.makedirs(out, exist_ok=True)
    S = int(rng.integers(1, 4))
    O = int(rng.integers(4, 13))
    # ---- 물체·링크 (판끼리 같은 물체 구성, 자세만 다름)
    n_links = rng.integers(1, 3, O)
    obj_half = rng.uniform(0.05, 0.6, (O, 3))
    link_meshes = []  # 물체별 링크 메시 (없으면 None)
    for o in range(O):
        ms = []
        for l in range(n_links[o]):
            if rng.random() < 0.08 and l > 0:
                ms.append(None)
            elif rng.random() < 0.05:
                ms.append(None)
            else:
                ms.append(rand_hull(rng, obj_half[o] * rng.uniform(0.5, 1.0, 3)))
        link_meshes.append(ms)
    # 링크 평평 번호 (판 순서, 물체 순서, 링크 순서)
    link_scene, link_obj, link_mesh = [], [], []
    for s in range(S):
        for o in range(O):
            for l in range(n_links[o]):
                link_scene.append(s)
                link_obj.append(o)
                link_mesh.append(link_meshes[o][l])
    L = len(link_scene)
    # ---- 자세: 격자 + 일부는 다른 물체 위/안
    base_pos = np.zeros((S, O, 3))
    for s in range(S):
        for o in range(O):
            base_pos[s, o] = [rng.uniform(-3, 3), rng.uniform(-3, 3), obj_half[o, 2]]
        for o in range(O):
            r = rng.random()
            t = int(rng.integers(0, O))
            if t == o:
                continue
            if r < 0.3:  # t 위에 올려놓기 (닿거나 살짝 뜸)
                base_pos[s, o, :2] = base_pos[s, t, :2] + rng.uniform(-0.1, 0.1, 2)
                base_pos[s, o, 2] = base_pos[s, t, 2] + obj_half[t, 2] + obj_half[o, 2] + rng.choice([0.0, 1e-3, 0.02])
            elif r < 0.45:  # t 안 (중심 겹침)
                base_pos[s, o] = base_pos[s, t] + rng.uniform(-0.05, 0.05, 3)
            elif r < 0.6:  # t 옆
                base_pos[s, o] = base_pos[s, t] + np.array([obj_half[t, 0] + obj_half[o, 0] + rng.uniform(0, 0.2), 0, 0])
    poses = np.zeros((L, 7), np.float32)
    for i in range(L):
        s, o = link_scene[i], link_obj[i]
        p = base_pos[s, o] + (rng.uniform(-0.05, 0.05, 3) if rng.random() < 0.5 else 0)
        poses[i, :3] = p
        poses[i, 3:] = rand_quat(rng)
    # ---- wp.Mesh 만들기 (RigidBodyViewAPI.initialize_view 와 같은 방식)
    meshes = []
    mesh_ids = np.zeros(L, np.uint64)
    vcount = np.zeros(L, np.int32)
    pts_all, tri_all, pts_off, tri_off = [], [], [0], [0]
    for i in range(L):
        m = link_mesh[i]
        if m is None:
            pts_off.append(pts_off[-1])
            tri_off.append(tri_off[-1])
            continue
        pts, tri = m
        wm = wp.Mesh(points=wp.array(pts, dtype=wp.vec3, device=D), indices=wp.array(tri.reshape(-1), dtype=wp.int32, device=D))
        meshes.append(wm)
        mesh_ids[i] = wm.id
        vcount[i] = len(pts)
        pts_all.append(pts)
        tri_all.append(tri)
        pts_off.append(pts_off[-1] + len(pts))
        tri_off.append(tri_off[-1] + len(tri))
    save(out, S=np.array([S, O, L], np.int32), poses=poses, link_scene=np.array(link_scene, np.int32),
         link_obj=np.array(link_obj, np.int32), mesh_pts=f32(np.concatenate(pts_all)) if pts_all else np.zeros((0, 3), np.float32),
         mesh_tri=np.concatenate(tri_all).astype(np.int32) if tri_all else np.zeros((0, 3), np.int32),
         pts_off=np.array(pts_off, np.int32), tri_off=np.array(tri_off, np.int32), has_mesh=(mesh_ids != 0).astype(np.uint8))
    link_mesh_ids = wp.array(mesh_ids, dtype=wp.uint64, device=D)

    # ---- 1. 자세 → 행렬
    poses_gpu = wp.array(poses, dtype=wp.float32, device=D)
    mats = wp.zeros(L, dtype=wp.mat44, device=D)
    wp.launch(UU._poses_to_matrices_kernel, dim=L, inputs=[poses_gpu, mats], device=D)
    save(out, pose_matrices=mats.numpy().reshape(L, 16))

    # ---- 2. AABB (aabb.py initialize_view + prepare_aabb_kernel_inputs + _update_values)
    prim_body_idx, link_idx, base_link_body, base_link_rows = [], [], [], []
    first_link = {}
    for i in range(L):
        key = (link_scene[i], link_obj[i])
        if key not in first_link:
            first_link[key] = i
            base_link_body.append(i)
            base_link_rows.append(link_scene[i] * O + link_obj[i])
        if vcount[i] > 0:
            prim_body_idx.append(i)
            link_idx.append(link_scene[i] * O + link_obj[i])
    aabb_links, aabb_verts, aabb_objs = [], [], []
    for b, row in zip(prim_body_idx, link_idx):
        for v in range(vcount[b]):
            aabb_links.append(b)
            aabb_verts.append(v)
            aabb_objs.append(row)
    aabb = wp.zeros((S * O, 6), dtype=wp.float32, device=D)
    wp.launch(AABBM._aabb_init_kernel, dim=S * O, inputs=[aabb], device=D)
    if aabb_links:
        wp.launch(UU._aabb_reduce_kernel, dim=len(aabb_links),
                  inputs=[mats, link_mesh_ids, wpa(np.array(aabb_links, np.int32)), wpa(np.array(aabb_verts, np.int32)),
                          wpa(np.array(aabb_objs, np.int32)), aabb], device=D)
    wp.launch(UU._aabb_baselink_fallback_kernel, dim=len(base_link_body),
              inputs=[poses_gpu, wpa(np.array(base_link_body, np.int32)), wpa(np.array(base_link_rows, np.int32)), aabb], device=D)
    aabb_np = aabb.numpy()
    save(out, aabb_base_link=np.array(base_link_body, np.int32), aabb=aabb_np)
    aabb3 = wp.array(aabb_np.reshape(S, O, 6), dtype=wp.float32, device=D)

    # ---- 3. Open (open_state.py _update_values)
    n_art = S * O
    max_dof = int(rng.integers(1, 5))
    jp = f32(rng.uniform(-2, 2, (n_art, max_dof)))
    rows = np.arange(S * O, dtype=np.int32).reshape(S, O)
    rng.shuffle(rows.reshape(-1))
    mask = (rng.random((S, O, max_dof)) < 0.6).astype(np.uint8)
    lo = rng.uniform(-2, 0, (S, O, max_dof))
    hi = lo + rng.uniform(0.1, 2.5, (S, O, max_dof))
    th1 = f32(0.95 * lo + 0.05 * hi)
    th2 = f32(0.95 * hi + 0.05 * lo)
    d1 = f32(np.where(hi > th1, 1.0, -1.0))
    d2 = f32(np.where(lo > th2, 1.0, -1.0))
    both = (rng.random((S, O)) < 0.3).astype(np.uint8)
    open_out = wp.zeros((S, O), dtype=wp.uint8, device=D)
    wp.launch(OPN._open_update_kernel, dim=(S, O),
              inputs=[wpa(jp), wpa(rows), wpa(mask), wpa(th1), wpa(d1), wpa(th2), wpa(d2), wpa(both), wp.int32(max_dof),
                      open_out], device=D)
    save(out, open_jp=jp, open_rows=rows, open_mask=mask, open_th1=th1, open_d1=d1, open_th2=th2, open_d2=d2,
         open_both=both, open_out=open_out.numpy())

    # ---- 4. 접촉 행렬 + 몸체 자세 (RigidContactAPIImpl.update) — 판 0 만
    B = int(rng.integers(3, 20))
    R = int(rng.integers(1, B + 1))
    C = int(rng.integers(1, B + 3))
    nst = int(rng.integers(1, 5))
    npend = 4
    prev_tf = np.zeros((B, 7), np.float32)
    for b in range(B):
        prev_tf[b, :3] = rng.uniform(-2, 2, 3)
        prev_tf[b, 3:] = rand_quat(rng)
    all_tf = np.zeros((npend, B, 7), np.float32)
    for i in range(npend):
        for b in range(B):
            base = prev_tf[b] if i == 0 else all_tf[i - 1, b]
            r = rng.random()
            if r < 0.4:
                all_tf[i, b] = base  # 잠: 그대로
            elif r < 0.55:
                all_tf[i, b] = base
                all_tf[i, b, int(rng.integers(0, 3))] += np.float32(rng.choice([5e-7, 1e-6, 2e-6, 1e-5]))
            elif r < 0.7:
                all_tf[i, b] = base
                q = base[3:].astype(np.float64)
                q = q + rng.standard_normal(4) * rng.choice([1e-5, 1e-4, 1e-2])
                all_tf[i, b, 3:] = f32(q / np.linalg.norm(q))
            else:
                all_tf[i, b, :3] = base[:3] + f32(rng.standard_normal(3) * 1e-3)
                all_tf[i, b, 3:] = rand_quat(rng)
    net = f32(np.where(rng.random((npend, R, 3)) < 0.2, rng.standard_normal((npend, R, 3)), 0.0))
    rows_b = rng.permutation(B)[:R].astype(np.int32)
    b2r = np.full(B, -1, np.int32)
    b2r[rows_b] = np.arange(R, dtype=np.int32)
    col_b = np.array([rng.integers(-1, B) if rng.random() < 0.8 else -1 for _ in range(C)], np.int32)
    imp = f32(np.where(rng.random((npend, R, C, 3)) < 0.25, rng.standard_normal((npend, R, C, 3)), 0.0))
    cm0 = (rng.random((R, C)) < 0.3).astype(np.uint8)
    ccm0 = (rng.random((R, C)) < 0.3).astype(np.uint8)
    body_tf = wpa(prev_tf)
    cm = wpa(cm0)
    ccm = wpa(ccm0)
    nsteps = wpa(np.array([nst], np.int32))
    wp.launch(UU._update_contact_matrices_kernel, dim=(R, C),
              inputs=[wpa(all_tf), body_tf, wpa(net), wpa(b2r), wpa(imp), wpa(rows_b), wpa(col_b), nsteps,
                      wp.float32(1e-6), wp.float32(1e-4), cm, ccm], device=D)
    wp.launch(UU._update_body_transforms_kernel, dim=B,
              inputs=[wpa(all_tf), body_tf, wpa(net), wpa(b2r), nsteps, wp.float32(1e-6), wp.float32(1e-4), body_tf],
              device=D)
    save(out, cm_all_tf=all_tf, cm_prev_tf=prev_tf, cm_net=net, cm_b2r=b2r, cm_imp=imp, cm_rows=rows_b, cm_cols=col_b,
         cm_nsteps=np.array([nst], np.int32), cm_in=cm0, ccm_in=ccm0, cm_out=cm.numpy(), ccm_out=ccm.numpy(),
         body_tf_out=body_tf.numpy())

    # ---- 5. Touching (판마다 접촉 행렬 + 행/열 마스크)
    N = O
    tvals = wp.zeros((S, N, N), dtype=wp.uint8, device=D)
    pair = wp.zeros((S, N, N), dtype=wp.int32, device=D)
    t_rows, t_cols, t_cm = [], [], []
    for s in range(S):
        Rs = int(rng.integers(1, 2 * N))
        Cs = int(rng.integers(1, 2 * N))
        rm = (rng.random((N, Rs)) < 0.15).astype(np.uint8)
        cmk = (rng.random((N, Cs)) < 0.15).astype(np.uint8)
        cmat = (rng.random((Rs, Cs)) < 0.2).astype(np.uint8)
        o2c = wp.zeros((N, Cs), dtype=wp.int32, device=D)
        wp.launch(TCH._touching_obj_to_col_kernel, dim=(N, Rs, Cs), inputs=[wpa(rm), wpa(cmat), o2c], device=D)
        wp.launch(TCH._touching_pair_kernel, dim=(N, N, Cs), inputs=[o2c, wpa(cmk), pair[s]], device=D)
        t_rows.append(rm)
        t_cols.append(cmk)
        t_cm.append(cmat)
    wp.launch(TCH._touching_finalize_kernel, dim=(S, N, N), inputs=[pair, tvals], device=D)
    for s in range(S):
        save(out, **{f"t_rows_{s}": t_rows[s], f"t_cols_{s}": t_cols[s], f"t_cm_{s}": t_cm[s]})
    touching_np = tvals.numpy()
    save(out, touching=touching_np)

    # ---- 6. Inside
    aidx = np.arange(N, dtype=np.int32)
    if rng.random() < 0.3:
        aidx[rng.integers(0, N)] = -1
    # 담는 메시: 무작위 물체의 첫 링크에 볼록 껍질 (링크 좌표), inv_local_w_scale = (링크→메시 척도 변환)^-1
    mesh_container, mesh_scene, mesh_parent, mesh_inv, fc, fn, f2m = [], [], [], [], [], [], []
    for s in range(S):
        for o in range(O):
            if rng.random() < 0.4:
                continue
            parent = first_link[(s, o)]
            for _ in range(int(rng.integers(1, 3))):
                pts, tri = rand_hull(rng, obj_half[o] * rng.uniform(0.6, 1.2, 3))
                tp = th.from_numpy(pts)
                hv = tp[th.from_numpy(tri).long()]
                cen = hv.mean(dim=1)
                nrm = th.cross(hv[:, 1] - hv[:, 0], hv[:, 2] - hv[:, 0], dim=1)
                nrm = nrm / th.clamp(th.linalg.vector_norm(nrm, dim=1, keepdim=True), min=1e-8)
                nrm = INS._orient_face_normals_outward(tp, cen, nrm)
                sc = rng.uniform(0.7, 1.3, 3)
                Tm = np.eye(4)
                Tm[:3, :3] = np.diag(sc)
                Tm[:3, 3] = rng.uniform(-0.05, 0.05, 3)
                inv = np.linalg.inv(Tm).astype(np.float32)
                mi = len(mesh_container)
                mesh_container.append(o)
                mesh_scene.append(s)
                mesh_parent.append(parent)
                mesh_inv.append(inv)
                fc.append(cen.numpy().astype(np.float32))
                fn.append(nrm.numpy().astype(np.float32))
                f2m.extend([mi] * len(tri))
    M = len(mesh_container)
    prefilter = wp.zeros((S, N, N), dtype=wp.int32, device=D)
    ipair = wp.zeros((S, N, N), dtype=wp.int32, device=D)
    ivals = wp.zeros((S, N, N), dtype=wp.uint8, device=D)
    aidx_wp = wpa(aidx)
    wp.launch(INS._inside_aabb_prefilter_kernel, dim=(S, N, N), inputs=[aabb3, aidx_wp, prefilter], device=D)
    inv_world_np = np.zeros((0, 16), np.float32)
    outside_np = np.zeros((S, N, 0), np.int32)
    if M > 0:
        inv_world = wp.zeros(M, dtype=wp.mat44, device=D)
        wp.launch(INS._inside_inv_world_kernel, dim=M,
                  inputs=[mats, wpa(np.array(mesh_parent, np.int32)), wp.array(np.stack(mesh_inv), dtype=wp.mat44, device=D),
                          inv_world], device=D)
        F = len(f2m)
        outside = wp.zeros((S, N, M), dtype=wp.int32, device=D)
        wp.launch(INS._inside_halfspace_test_kernel, dim=(S, N, F),
                  inputs=[aabb3, aidx_wp, prefilter, inv_world, wpa(np.array(f2m, np.int32)),
                          wpa(np.array(mesh_container, np.int32)), wpa(np.array(mesh_scene, np.int32)),
                          wp.array(np.concatenate(fc), dtype=wp.vec3, device=D),
                          wp.array(np.concatenate(fn), dtype=wp.vec3, device=D), outside], device=D)
        wp.launch(INS._inside_mesh_reduce_kernel, dim=(S, N, M),
                  inputs=[aidx_wp, prefilter, wpa(np.array(mesh_container, np.int32)), wpa(np.array(mesh_scene, np.int32)),
                          outside, ipair], device=D)
        inv_world_np = inv_world.numpy().reshape(M, 16)
        outside_np = outside.numpy()
        save(out, in_mesh_container=np.array(mesh_container, np.int32), in_mesh_scene=np.array(mesh_scene, np.int32),
             in_mesh_parent=np.array(mesh_parent, np.int32), in_mesh_inv=np.stack(mesh_inv).reshape(M, 16),
             in_fc=np.concatenate(fc), in_fn=np.concatenate(fn), in_f2m=np.array(f2m, np.int32))
    wp.launch(INS._inside_finalize_kernel, dim=(S, N, N), inputs=[ipair, ivals], device=D)
    save(out, in_aidx=aidx, in_M=np.array([M], np.int32), in_prefilter=prefilter.numpy(), in_inv_world=inv_world_np,
         in_outside=outside_np, inside=ivals.numpy())

    # ---- 7. Adjacency (+ OnTop / Under / NextTo)
    adj_aabb = aidx.copy()
    dirs, maxd = ADJ._build_adjacency_axis_tables()
    K = dirs.shape[0]
    l2o = np.array(link_obj, np.int32)
    l2s = np.array(link_scene, np.int32)
    if rng.random() < 0.3:
        l2o[rng.integers(0, L)] = -1
    adj_scr = wp.zeros((S, N, N, K), dtype=wp.int32, device=D)
    adj_vals = wp.zeros((S, N, N, K), dtype=wp.uint8, device=D)
    wp.launch(ADJ._adjacency_ray_cast_kernel, dim=(S, N, L, K),
              inputs=[aabb3, wpa(adj_aabb), wp.array(dirs.numpy(), dtype=wp.float32, device=D),
                      wp.array(maxd.numpy(), dtype=wp.float32, device=D), link_mesh_ids, mats, wpa(l2o), wpa(l2s), adj_scr],
              device=D)
    wp.launch(ADJ._adjacency_finalize_kernel, dim=(S, N, N, K), inputs=[adj_scr, adj_vals], device=D)
    adj_np = adj_vals.numpy()
    save(out, adj_dirs=dirs.numpy().astype(np.float32), adj_maxd=maxd.numpy().astype(np.float32), adj_l2o=l2o,
         adj_l2s=l2s, adjacency=adj_np)
    # OnTop / Under / NextTo: 상태별 번호 → Touching/Adjacency/AABB 번호 (일부 -1)
    tidx = np.arange(N, dtype=np.int32)
    jdx = np.arange(N, dtype=np.int32)
    if rng.random() < 0.3:
        tidx[rng.integers(0, N)] = -1
    if rng.random() < 0.3:
        jdx[rng.integers(0, N)] = -1
    ot = wp.zeros((S, N, N), dtype=wp.uint8, device=D)
    un = wp.zeros((S, N, N), dtype=wp.uint8, device=D)
    nt = wp.zeros((S, N, N), dtype=wp.uint8, device=D)
    wp.launch(ONT._on_top_kernel, dim=(S, N, N), inputs=[tvals, adj_vals, wpa(tidx), wpa(jdx), ot], device=D)
    wp.launch(UND._under_kernel, dim=(S, N, N), inputs=[adj_vals, wpa(jdx), un], device=D)
    wp.launch(NXT._next_to_kernel, dim=(S, N, N), inputs=[aabb3, adj_vals, aidx_wp, wpa(jdx), nt], device=D)
    save(out, st_tidx=tidx, st_jdx=jdx, on_top=ot.numpy(), under=un.numpy(), next_to=nt.numpy())

    # ---- 8. ToggledOn (toggle.py ToggledOn._update_values 순서 그대로)
    Rt = int(rng.integers(2, 12))
    Ct = int(rng.integers(2, 12))
    tg_cm = [(rng.random((Rt, Ct)) < 0.3).astype(np.uint8) for _ in range(S)]
    tg_q = [(rng.random((1, Rt)) < 0.4).astype(np.uint8) for _ in range(S)]
    tg_with = [(rng.random((O, Ct)) < 0.3).astype(np.uint8) for _ in range(S)]
    tvals0 = (rng.random((S, O)) < 0.5).astype(np.uint8)
    ttime0 = f32(rng.choice([0.0, 0.03333334, 0.06666667, 0.1, 0.13333334, 0.14999999], (S, O)))
    tvals_wp = wpa(tvals0)
    ttime_wp = wpa(ttime0)
    mask = wp.zeros((S, O), dtype=wp.int32, device=D)
    mask_flat = mask.reshape((S * O,))
    for s in range(S):
        wp.launch(UU._is_in_contact_batch_kernel, dim=(O, Rt, Ct),
                  inputs=[wpa(tg_q[s]), wpa(tg_cm[s]), wpa(tg_with[s]), wp.int32(0), mask[s]], device=D)
    mask_after_contact = mask.numpy().copy()
    # requires_closed: 일부 물체, Open 값 (S*O)
    rc_this = [int(k) for k in range(S * O) if rng.random() < 0.3]
    open_flat = (rng.random(S * O) < 0.4).astype(np.uint8)
    rc_open = [int(rng.integers(0, S * O)) for _ in rc_this]
    if rc_this:
        wp.launch(TOG._check_requires_closed_kernel, dim=len(rc_this),
                  inputs=[wpa(np.array(rc_this, np.int32)), wpa(np.array(rc_open, np.int32)), wpa(open_flat),
                          tvals_wp.reshape((S * O,)), ttime_wp.reshape((S * O,)), mask_flat], device=D)
    # 표식: 물체마다 (판 s, 물체 o) 부모 링크 = 첫 링크, 손가락 링크 = 메시 있는 링크 무작위 2~4 개 (판 s 안)
    mk_parent = np.zeros(S * O, np.int32)
    mk_off = np.zeros((S * O, 3), np.float32)
    mk_rad = np.zeros(S * O, np.float32)
    pairs = []
    matsn = mats.numpy()
    for s in range(S):
        fl = [l for l in range(L) if link_scene[l] == s and mesh_ids[l] != 0]
        fingers = list(rng.choice(fl, min(len(fl), int(rng.integers(2, 5))), replace=False)) if fl else []
        for o in range(O):
            k = s * O + o
            par = first_link[(s, o)]
            mk_parent[k] = par
            mk_rad[k] = np.float32(rng.uniform(0.01, 0.1))
            if fingers and rng.random() < 0.8:
                # 표식 중심을 손가락 메시 꼭짓점 근처(반지름 안팎)에 둔다: 세계 → 부모 링크 좌표 offset
                fg = int(rng.choice(fingers))
                v = link_mesh[fg][0][rng.integers(0, len(link_mesh[fg][0]))].astype(np.float64)
                Mf = matsn[fg].astype(np.float64)
                wpt = Mf[:3, :3] @ v + Mf[:3, 3]
                wpt = wpt + rng.standard_normal(3) / np.sqrt(3) * mk_rad[k] * rng.uniform(0.5, 1.5)
                Mp = matsn[par].astype(np.float64)
                mk_off[k] = f32(Mp[:3, :3].T @ (wpt - Mp[:3, 3]))
            else:
                mk_off[k] = f32(rng.uniform(-0.3, 0.3, 3))
            for fg in fingers:
                pairs.append((k, int(fg)))
    if pairs:
        wp.launch(TOG._check_overlap_kernel, dim=len(pairs),
                  inputs=[mats, link_mesh_ids, wpa(mk_parent), wp.array(mk_off, dtype=wp.vec3, device=D), wpa(mk_rad),
                          wp.array(np.array(pairs, np.int32), dtype=wp.vec2i, device=D),
                          wpa(np.arange(S * O, dtype=np.int32)), mask_flat], device=D)
    mask_before_set = mask.numpy().copy()
    dt = wp.array(np.array([np.float32(1.0 / 30.0)], np.float32), dtype=wp.float32, device=D)
    wp.launch(TOG._set_toggle_value_kernel, dim=(S, O),
              inputs=[tvals_wp, mask_flat, ttime_wp, wp.int32(O), wp.float32(TOG.m.CAN_TOGGLE_SECONDS), dt], device=D)
    save(out, tg_R=np.array([Rt, Ct], np.int32), tg_vals0=tvals0, tg_time0=ttime0, tg_rc_this=np.array(rc_this, np.int32),
         tg_rc_open=np.array(rc_open, np.int32), tg_open=open_flat, tg_mk_parent=mk_parent, tg_mk_off=mk_off, tg_mk_rad=mk_rad,
         tg_pairs=np.array(pairs, np.int32).reshape(-1, 2), tg_mask_contact=mask_after_contact, tg_mask_set=mask_before_set,
         tg_vals=tvals_wp.numpy(), tg_time=ttime_wp.numpy(), tg_mask_end=mask.numpy())
    for s in range(S):
        save(out, **{f"tg_cm_{s}": tg_cm[s], f"tg_q_{s}": tg_q[s], f"tg_with_{s}": tg_with[s]})

    # ---- 9. 온도 사슬 (HeatSourceOrSink → Temperature(열 모으기·감쇠·불 유지) → MaxTemperature → OnFire), 4 스텝
    H = int(rng.integers(1, min(O, 5) + 1))
    src_obj = rng.choice(O, H, replace=False)
    hs_self_temp = np.array([int(o) if rng.random() < 0.85 else -1 for o in src_obj], np.int32)
    hs_self_in = np.array([int(o) if rng.random() < 0.85 else -1 for o in src_obj], np.int32)
    hs_req_in = (rng.random(H) < 0.3).astype(np.uint8)
    hs_req_tg = (rng.random(H) < 0.4).astype(np.uint8)
    hs_req_cl = (rng.random(H) < 0.3).astype(np.uint8)
    hs_req_fi = (rng.random(H) < 0.25).astype(np.uint8)
    hs_tg_idx = np.array([int(o) if rng.random() < 0.9 else -1 for o in src_obj], np.int32)
    hs_op_idx = np.array([int(o) if rng.random() < 0.9 else -1 for o in src_obj], np.int32)
    hs_fi_idx = np.array([int(o) if rng.random() < 0.9 else -1 for o in src_obj], np.int32)
    hs_temp = f32(rng.choice([200.0, 250.0, 1000.0, -20.0, 180.5], H))
    hs_rate = f32(rng.choice([0.04, 0.1, 0.5, 0.037], H))
    hs_thr = f32(rng.uniform(0.05, 0.4, H))
    hs_ign = f32(rng.choice([250.0, 200.0, 99.9], H))
    hs_link = np.full((S, H), -1, np.int32)
    for s_ in range(S):
        for h in range(H):
            ls_ = [l for l in range(L) if link_scene[l] == s_ and link_obj[l] == int(src_obj[h])]
            if ls_ and rng.random() < 0.95:
                hs_link[s_, h] = int(rng.choice(ls_))
    # 열 요소 위치: 판 0 기준으로 대상 메시 면 위 한 점에서 바깥(또는 안)으로 문턱 ± 몇 ulp 떨어진 곳 (경계 시험)
    hs_off = np.zeros((H, 3), np.float32)
    for h in range(H):
        li = hs_link[0, h]
        tl = [l for l in range(L) if link_scene[l] == 0 and link_obj[l] != int(src_obj[h]) and mesh_ids[l] != 0]
        if li < 0 or not tl or rng.random() < 0.2:
            hs_off[h] = f32(rng.uniform(-0.5, 0.5, 3))
            continue
        tgt = int(rng.choice(tl))
        pts_t, tri_t = link_mesh[tgt]
        f_ = tri_t[rng.integers(0, len(tri_t))]
        bc = rng.dirichlet([1, 1, 1])
        ploc = bc @ pts_t[f_].astype(np.float64)
        nrm = np.cross(pts_t[f_[1]] - pts_t[f_[0]], pts_t[f_[2]] - pts_t[f_[0]]).astype(np.float64)
        nrm /= np.linalg.norm(nrm) + 1e-30
        if np.dot(nrm, ploc - pts_t.mean(0)) < 0:
            nrm = -nrm
        r_ = rng.random()
        if r_ < 0.5:
            dist = float(hs_thr[h]) * (1.0 + float(rng.integers(-4, 5)) * 1.2e-7)
        elif r_ < 0.7:
            dist = -float(rng.uniform(0.001, 0.05))
        else:
            dist = float(rng.uniform(0.0, 2.0)) * float(hs_thr[h])
        Mt = matsn[tgt].astype(np.float64)
        wpt = Mt[:3, :3] @ (ploc + nrm * dist) + Mt[:3, 3]
        Ms = matsn[li].astype(np.float64)
        hs_off[h] = f32(Ms[:3, :3].T @ (wpt - Ms[:3, 3]))
    t2a = np.arange(O, dtype=np.int32)
    t2i = np.arange(O, dtype=np.int32)
    if rng.random() < 0.3:
        t2a[rng.integers(0, O)] = -1
    if rng.random() < 0.3:
        t2i[rng.integers(0, O)] = -1
    tl_off, tl_idx = [0], []
    for s_ in range(S):
        for n in range(O):
            if rng.random() < 0.9:
                tl_idx.extend([l for l in range(L) if link_scene[l] == s_ and link_obj[l] == n])
            tl_off.append(len(tl_idx))
    tl_off = np.array(tl_off, np.int32)
    tl_idx = np.array(tl_idx if tl_idx else [0], np.int32)
    tg_v = (rng.random((S, O)) < 0.6).astype(np.uint8)
    op_v = (rng.random((S, O)) < 0.4).astype(np.uint8)
    fi_v = (rng.random((S, O)) < 0.4).astype(np.uint8)
    n_sc = [S if rng.random() < 0.8 else max(0, S - 1) for _ in range(3)]
    temp = f32(np.where(rng.random((S, O)) < 0.3, rng.choice([250.0, 23.0, 70.0, 0.0, 99.9], (S, O)), rng.uniform(-30, 300, (S, O))))
    mx_idx = np.arange(O, dtype=np.int32)
    mx = f32(np.where(rng.random((S, O)) < 0.3, -np.inf, rng.uniform(-30, 300, (S, O))))
    of_idx = np.array([o if rng.random() < 0.9 else -1 for o in range(O)], np.int32)
    of_ign = f32(rng.choice([250.0, 200.0, 99.9], O))
    save(out, hs_n=np.array([H] + n_sc, np.int32), hs_self_temp=hs_self_temp, hs_self_in=hs_self_in, hs_req_in=hs_req_in,
         hs_req_tg=hs_req_tg, hs_req_cl=hs_req_cl, hs_req_fi=hs_req_fi, hs_tg_idx=hs_tg_idx, hs_op_idx=hs_op_idx,
         hs_fi_idx=hs_fi_idx, hs_temp=hs_temp, hs_rate=hs_rate, hs_thr=hs_thr, hs_ign=hs_ign, hs_link=hs_link, hs_off=hs_off,
         tp_t2a=t2a, tp_t2i=t2i, tp_tl_off=tl_off, tp_tl_idx=tl_idx, tp_tg_v=tg_v, tp_op_v=op_v, tp_mx_idx=mx_idx,
         tp_of_idx=of_idx, tp_of_ign=of_ign)
    ph = wp.zeros((1, 1), dtype=wp.uint8, device=D)
    temp_wp = wpa(temp)
    mx_wp = wpa(mx)
    fi_wp = wpa(fi_v)
    dt = wp.array(np.array([np.float32(1.0 / 30.0)], np.float32), dtype=wp.float32, device=D)
    for k in range(4):
        fi_in = fi_wp.numpy().copy()
        hv = wp.zeros((S, H), dtype=wp.uint8, device=D)
        wp.launch(HSS._heatsource_is_active_kernel, dim=(S, H),
                  inputs=[wpa(hs_req_tg), wpa(hs_req_cl), wpa(hs_req_fi), wpa(hs_tg_idx), wpa(hs_op_idx), wpa(hs_fi_idx),
                          wpa(tg_v), wpa(op_v), fi_wp, wp.int32(n_sc[0]), wp.int32(n_sc[1]), wp.int32(n_sc[2]), hv], device=D)
        infl = wp.zeros((S, H, O), dtype=wp.uint8, device=D)
        inc = wp.zeros((S, O), dtype=wp.float32, device=D)
        t_in = temp_wp.numpy().copy()
        wp.launch(TMP._incoming_heat_kernel, dim=(S, H, O),
                  inputs=[hv, wpa(hs_req_in), wpa(hs_temp), wpa(hs_rate), wpa(hs_thr), wpa(hs_self_temp), wpa(hs_self_in),
                          wpa(hs_link), wp.array(hs_off, dtype=wp.vec3, device=D), wpa(t2a), wpa(t2i), wpa(tl_off), wpa(tl_idx),
                          link_mesh_ids, wp.int32(O), mats, aabb3, ivals, temp_wp, wp.int32(S), infl, inc], device=D)
        inc_np = inc.numpy().copy()
        wp.launch(TMP._temperature_decay_kernel, dim=(S, O),
                  inputs=[temp_wp, inc, wp.float32(TMP.m.DEFAULT_TEMPERATURE), wp.float32(TMP.m.TEMPERATURE_DECAY_SPEED), dt],
                  device=D)
        t_dec = temp_wp.numpy().copy()
        wp.launch(TMP._self_heating_clamp_kernel, dim=(S, H),
                  inputs=[hv, wpa(hs_req_fi), wpa(hs_temp), wpa(hs_ign), wpa(hs_self_temp), temp_wp], device=D)
        mx_in = mx_wp.numpy().copy()
        wp.launch(MXT._max_temperature_kernel, dim=(S, O), inputs=[mx_wp, temp_wp, wpa(mx_idx)], device=D)
        wp.launch(ONF._on_fire_kernel, dim=(S, O), inputs=[wpa(of_idx), wpa(of_ign), temp_wp, fi_wp], device=D)
        save(out, **{f"tp_fi_in_{k}": fi_in, f"tp_hv_{k}": hv.numpy(), f"tp_t_in_{k}": t_in, f"tp_infl_{k}": infl.numpy(),
                     f"tp_inc_{k}": inc_np, f"tp_t_dec_{k}": t_dec, f"tp_t_out_{k}": temp_wp.numpy(), f"tp_mx_in_{k}": mx_in,
                     f"tp_mx_out_{k}": mx_wp.numpy(), f"tp_fi_out_{k}": fi_wp.numpy()})
    del ph
    # Cooked / Frozen: 공식 _get_value (MaxTemperature/Temperature.get_value = float32 .item()) 를 가짜 물체로 부른다
    class _F:
        pass
    mxn, tn = mx_wp.numpy(), temp_wp.numpy()
    ck_t = np.array([float(rng.choice([70.0, 58.7, 100.0 / 3.0, float(mxn[s_, o]) if np.isfinite(mxn[s_, o]) else 70.0,
                                       np.nextafter(float(mxn[s_, o]), np.inf) if np.isfinite(mxn[s_, o]) else 70.0]))
                     for s_ in range(S) for o in range(O)], np.float64)
    fz_t = np.array([float(rng.choice([0.0, -0.1, float(tn[s_, o]), np.nextafter(float(tn[s_, o]), -np.inf)]))
                     for s_ in range(S) for o in range(O)], np.float64)
    ck_o = np.zeros(S * O, np.uint8)
    fz_o = np.zeros(S * O, np.uint8)
    for s_ in range(S):
        for o in range(O):
            k = s_ * O + o
            ob, ms_, ts_ = _F(), _F(), _F()
            ms_.get_value = (lambda v=mxn[s_, o]: th.tensor(v).to(th.float32).item())
            ts_.get_value = (lambda v=tn[s_, o]: th.tensor(v).to(th.float32).item())
            ob.states = {MXT.MaxTemperature: ms_, TMP.Temperature: ts_}
            c = CKD.Cooked.__new__(CKD.Cooked)
            c.obj, c.cook_temperature = ob, ck_t[k]
            z = FRZ.Frozen.__new__(FRZ.Frozen)
            z.obj, z.freeze_temperature = ob, fz_t[k]
            ck_o[k] = bool(c._get_value())
            fz_o[k] = bool(z._get_value())
    save(out, ck_temp=ck_t, ck_out=ck_o, fz_temp=fz_t, fz_out=fz_o)
    wp.synchronize()
    del meshes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    wp.init()
    rng = np.random.default_rng(a.seed)
    for f in range(a.frames):
        frame(rng, os.path.join(a.out, f"frame_{f:04d}"))
    with open(os.path.join(a.out, "index.txt"), "w") as fo:
        fo.write("\n".join(f"frame_{f:04d}" for f in range(a.frames)) + "\n")
    print("done", a.frames)


if __name__ == "__main__":
    main()
