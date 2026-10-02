"""장면 추출 한 단계 (particles): 리드의 physx_capture.py 를 그대로 돌리면서, 판을 불러온 직후 입자 모듈 입력을 읽기 전용으로 적는다.
extract_instance.sh 가 `export ENGINE_CAPTURE_PY=<이 파일>` 한 줄로 부른다 (run_capture_linux.sh 가 이 변수로 기록 스크립트를 고름).
물리·난수·평가 결과는 바꾸지 않는다:
  - 값만 읽는다. torch 연산은 쓰지 않는다 (torch.compile 커널 이력을 바꾸지 않으려고 — docs 12.7). 예외: 제거기 링크의
    visual_boundary_points_local(cached_property) 은 공식도 스텝 중 같은 값을 계산해 캐시하는 것이라 먼저 부른다 (T.quat2mat 모양 (4,) — 이미 컴파일된 모양).
  - 실패해도 평가를 멈추지 않는다 (예외를 삼키고 적기만 한다).
출력 (기록 폴더 안, extract_instance.sh 가 결과 폴더로 옮김):
  particles_spec.json : 활성 계(순서), 제거기(이름·범주·방식·링크 경로·능력 인자 원문·한도·투영 메시 종류·속성), 도포기 목록, 자를 것·자르개·다질 것 목록
  particles_arrays.npz: 제거기별 볼록 껍질 점(링크 국소), 시각 입자(계·무리 물체·링크 경로·국소 4x4, 공식 dict 순서)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.normpath(os.path.join(HERE, "..", "..", "capture"))
sys.path.insert(0, CAP)

import numpy as np  # noqa: E402

import physx_capture as PC  # noqa: E402


def dump_particle_spec(evaluator, out_dir):
    import omnigibson as og
    import omnigibson.object_states.particle_modifier as PM
    import omnigibson.systems.macro_particle_system as MPS

    scene = og.sim.scenes[0]
    spec = dict(active_systems=list(scene.active_systems.keys()), removers=[], appliers=[], sliceable=[], slicer=[], diceable=[])
    arrays = {}
    for o in scene.objects:
        ab = getattr(o, "_abilities", {}) or {}
        for k in ("sliceable", "slicer", "diceable"):
            if k in ab:
                spec[k].append(o.name)
        if PM.ParticleRemover in o.states:
            st = o.states[PM.ParticleRemover]
            r = dict(name=o.name, category=o.category, method=str(st.method), link=None, link_path=None,
                     ability=json.loads(json.dumps(ab.get("particleRemover", {}), default=str)),
                     systems=list(st.conditions.keys()), limit_visual=int(st.visual_particle_modification_limit),
                     limit_physical=int(st.physical_particle_modification_limit))
            if st.link is not None:
                r["link"], r["link_path"] = st.link.name, st.link.prim_path
                if st.method == PM.ParticleModifyMethod.ADJACENCY:
                    hull = st.link.visual_boundary_points_local
                    if hull is not None:
                        arrays[f"hull__{o.name}"] = hull.numpy().astype(np.float32)
                else:
                    ms = list(st.link.visual_meshes.values())
                    if ms:
                        m = ms[0]
                        names = set(m.prim.GetPropertyNames())
                        r["mesh_type"] = m._mesh_type
                        r["mesh_path"] = m.prim_path
                        r["mesh_attr"] = [float(m.get_attribute(k)) if k in names else 0.0 for k in ("radius", "height", "size")]
            spec["removers"].append(r)
        if PM.ParticleApplier in o.states:
            spec["appliers"].append(dict(name=o.name, category=o.category,
                                         ability=json.loads(json.dumps(ab.get("particleApplier", {}), default=str))))
    for s in scene.active_systems.values():
        if isinstance(s, MPS.MacroVisualParticleSystem) and s.particles:
            names = list(s.particles.keys())
            arrays[f"vis__{s.name}__obj"] = np.array([s._particles_info[n]["obj"].name for n in names])
            arrays[f"vis__{s.name}__link"] = np.array([s._particles_info[n]["link"].prim_path for n in names])
            arrays[f"vis__{s.name}__local_mat"] = np.stack([s._particles_local_mat[n].numpy() for n in names]).astype(np.float32)
    # 입자 익히기 레시피 (등록 순서 = 실행 우선순위, 용기마다 첫 실행 가능한 하나만)
    import omnigibson.transition_rules as TR

    spec["cooking_particle_recipes"] = [
        dict(name=k, input_systems=list(v["input_systems"]), output_systems=list(v["output_systems"]),
             fillable_categories=None if v.get("fillable_categories") is None else list(v["fillable_categories"]))
        for k, v in TR.CookingPhysicalParticleRule._recipes.items()]
    spec["disabled_rules"] = [r.__name__ for r in TR.RULES_REGISTRY.objects if not r.ENABLED]
    # 판 시작 시 난수 생성기 상태 (다지기 방향·뿌리기 표본·레시피 모델 고르기가 소비). 읽기만 한다 (소비 없음)
    import random

    import torch as th

    # base 창 물체 표 (core/particles/base_state.h): 등록부 차례 = dump_state/load_state 차례. 계 등록부가 물체 등록부보다 먼저다.
    from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim

    base = []
    for o in scene.objects:
        try:
            links = [l.prim_path for l in o.links.values() if isinstance(l, RigidDynamicPrim)]
            base.append(dict(name=o.name, articulated=bool(o.articulated), n_joints=int(o.n_joints), kinematic_only=bool(o.kinematic_only),
                             fixed_base=bool(o.fixed_base), prim_type=int(o.prim_type), root_link=o.root_link.prim_path, dynamic_links=links,
                             articulation_root=o.articulation_root_path if o.articulated else None,
                             robot=o in scene.robots))
        except Exception as e:
            base.append(dict(name=getattr(o, "name", "?"), error=repr(e)))
    spec["base_objects"] = base
    spec["system_registry"] = [s.name for s in scene.system_registry.objects]
    arrays["torch_rng_state"] = th.get_rng_state().numpy()
    st = random.getstate()
    spec["py_random_state"] = [st[0], list(st[1]), st[2]]
    with open(os.path.join(out_dir, "particles_spec.json"), "w") as f:
        json.dump(spec, f, indent=1, ensure_ascii=False)
    np.savez(os.path.join(out_dir, "particles_arrays.npz"), **arrays)
    n_vis = sum(len(v) for k, v in arrays.items() if k.endswith("__obj"))
    print(f"[particles] 추출: 제거기 {len(spec['removers'])}, 도포기 {len(spec['appliers'])}, 시각 입자 {n_vis}, "
          f"자를 것 {len(spec['sliceable'])}, 자르개 {len(spec['slicer'])}, 다질 것 {len(spec['diceable'])}", flush=True)


orig_install = PC.install


def install(cap):
    orig_install(cap)  # 리드의 겉싸개를 먼저 건다 (load_batch 도 이미 감싸짐)
    from omnigibson.eval import evaluator as E

    inner = E.BatchedEvaluator.load_batch

    def load_batch(self, *a, **kw):
        r = inner(self, *a, **kw)
        try:
            dump_particle_spec(self, cap.dump_dir)
        except Exception as e:  # 추출 실패가 평가를 바꾸면 안 된다
            print(f"[particles] 추출 실패: {e!r}", flush=True)
        return r

    E.BatchedEvaluator.load_batch = load_batch


PC.install = install

if __name__ == "__main__":
    PC.main()
