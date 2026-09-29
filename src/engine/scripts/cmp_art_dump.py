"""S0 로봇 상태 비교: 공식 텐서 뷰 덤프(physx_capture --dump-at-pre N 의 state_pre<N>.npz) vs 재생 덤프(ovd_replay --dump-art 텍스트).
링크·dof·모양을 이름으로 맞춰 비트 비교한다 (문서 15절).  python3 cmp_art_dump.py state_preN.npz ours.txt
"""
import sys, numpy as np
off = np.load(sys.argv[1], allow_pickle=True)
lp = [str(x) for x in np.asarray(off["attr_link_paths"]).ravel()]
dp = [str(x) for x in np.asarray(off["attr_dof_paths"]).ravel()]
print("공식 링크", len(lp), "dof", len(dp), "예", lp[:2], dp[:2])
links, dofs, shapes = {}, [], []
cur = None
for l in open(sys.argv[2]):
    t = l.split()
    if t[0] == "link_name": cur = t[2]; links[cur] = {}
    elif t[0] in ("mass", "com", "inertia_diag", "pose", "body"): links[cur][t[0]] = [float(x) if not x.startswith("0x") else int(x, 16) for x in t[2:]]
    elif t[0] == "dof":
        kv = {}
        i = 1
        # dof N link L axis A motion M lim lo hi k .. c .. maxf .. dtype .. env a b c d arm .. maxdofvel .. fr a b c pos .. vel .. tgt .. tvel ..
        kv["link"] = int(t[3]); kv["lim"] = (float(t[9]), float(t[10])); kv["k"] = float(t[12]); kv["c"] = float(t[14]); kv["maxf"] = float(t[16])
        kv["arm"] = float(t[t.index("arm") + 1]); kv["maxdofvel"] = float(t[t.index("maxdofvel") + 1]); fi = t.index("fr"); kv["fr"] = [float(x) for x in t[fi+1:fi+4]]
        kv["pos"] = float(t[t.index("pos") + 1]); kv["vel"] = float(t[t.index("vel") + 1]); kv["tgt"] = float(t[t.index("tgt") + 1]); kv["tvel"] = float(t[t.index("tvel") + 1])
        dofs.append(kv)
    elif t[0] == "shape": shapes.append(t)
link_order = list(links.keys())
# 우리 dof -> 자식 링크 이름
dof_link = [link_order[d["link"]] for d in dofs]
def bits(a): return np.asarray(a, np.float32).view(np.uint32)
def cmp(name, off_arr, ours_arr):
    a = np.asarray(off_arr, np.float32); b = np.asarray(ours_arr, np.float32)
    if a.shape != b.shape: print(f"{name}: 모양 {a.shape} vs {b.shape}"); return
    d = np.nonzero((bits(a) != bits(b)).reshape(a.shape[0], -1).any(1))[0]
    msg = f"{name}: {a.shape[0]} 개 중 다름 {len(d)}"
    for i in d[:4]: msg += f"\n    [{i}] 공식 {a[i]} 우리 {b[i]}"
    print(msg)
# 링크: 공식 순서로 우리 값을 줄 세움
lk = [links[p] for p in lp]
if "masses" in off.files: cmp("질량", off["masses"][0], [x["mass"][0] for x in lk])
if "inertias" in off.files: print("  (inertias 는 3x3 전역/국소 틀이라 대각 비교 생략)")
if "coms" in off.files: cmp("질량중심(국소 p,q)", off["coms"][0], [x["com"] for x in lk])
# dof: 공식 dof 경로의 마지막 이름(조인트) -> 자식 링크를 scope 로 못 찾으니, 우리 dof 를 자식 링크 순서로 맞춘다: 공식 dof 순서 = 링크 순서의 풀린 축
our_by_link = {}
for d, ln in zip(dofs, dof_link): our_by_link.setdefault(ln, []).append(d)
od = []
for p in lp: od += our_by_link.get(p, [])
print("dof 수: 공식", len(dp), "우리", len(od))
for key, name in (("k", "dof_stiffnesses"), ("c", "dof_dampings"), ("maxf", "dof_max_forces"), ("arm", "dof_armatures"), ("maxdofvel", "dof_max_velocities"), ("pos", "dof_positions"), ("vel", "dof_velocities"), ("tgt", "dof_position_targets"), ("tvel", "dof_velocity_targets")):
    if name in off.files: cmp(name, off[name][0], [d[key] for d in od])
if "dof_limits" in off.files: cmp("dof_limits", off["dof_limits"][0], [d["lim"] for d in od])
if "dof_friction_properties" in off.files: cmp("dof_friction_properties", off["dof_friction_properties"][0], [d["fr"] for d in od])
for k in sorted(off.files):
    try: print(f"  공식 {k}: 모양 {np.asarray(off[k]).shape}")
    except Exception as e: print(k, e)
print("---- 추가")
cmp("링크 자세 (p,q)", off["link_transforms"][0], [x["pose"] for x in lk])
lv = np.asarray(off["link_velocities"][0]); print("  공식 링크 속도 최대|v|", np.abs(lv).max())
dg = np.asarray(off["disable_gravities"][0]); print("  공식 중력 끔 링크 수", int(dg.sum()), "우리 actor_flags 는 덤프 참조")
I = np.asarray(off["inertias"][0]).reshape(44, 3, 3)
print("  공식 관성 대각 예(국소?)", I[6].diagonal(), "우리 질량공간 대각", lk[6]["inertia_diag"])
# 모양: 공식 순서 = 링크 순서(공식) x 모양 순서
by_link = {}
for s in shapes: by_link.setdefault(link_order[int(s[3])], []).append(s)
osh = []
for p in lp: osh += by_link.get(p, [])
print("모양 수: 공식", off["contact_offsets"].shape[1], "우리", len(osh))
cmp("접촉 거리", off["contact_offsets"][0], [float(s[5]) for s in osh])
cmp("쉼 거리", off["rest_offsets"][0], [float(s[6]) for s in osh])
cmp("재질 (정지·운동 마찰·반발)", off["material_properties"][0], [[float(s[s.index("mat") + 1 + i]) for i in range(3)] for s in osh])
print("  dof_types", np.asarray(off["dof_types"][0]).ravel()[:10], "motions", np.asarray(off["dof_motions"][0]).ravel()[:10], "drive_types", np.asarray(off["drive_types"][0]).ravel()[:10])
print("  drive_model", np.asarray(off["dof_drive_model_properties"][0])[:3], "friction_coef", np.asarray(off["dof_friction_coefficients"][0]).ravel()[:6])
print("  root", off["root_transforms"][0], off["root_velocities"][0])
print("---- 관성·중력")
def qmat(x, y, z, w):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
bad = 0
for i, x in enumerate(lk):
    R = qmat(*x["com"][3:7]); Ours = R @ np.diag(x["inertia_diag"]) @ R.T
    d = np.abs(Ours - I[i]).max(); rel = d / max(np.abs(I[i]).max(), 1e-30)
    if rel > 1e-5: bad += 1; print(f"  링크 {i} {lp[i].split('/')[-1]}: 상대차 {rel:.2e}\n    공식 {I[i].ravel()}\n    우리 {Ours.ravel()}")
print("관성 텐서(질량공간 대각 회전 복원, 상대 1e-5): 다른 링크", bad)
