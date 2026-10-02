"""Linux(WSL2) 에서 Isaac Sim 5.1 Kit 이 RTX 없이 뜨는지, PhysX 가 도는지 확인하는 최소 시험.

    python linux_boot_test.py [kit|og] [--/설정=값 ...]
      kit : SimulationApp(headless) 만 띄우고 PhysX 로 상자 하나 떨어뜨린다
      og  : OmniGibson 을 띄우고 빈 장면에 물체 하나로 몇 스텝
"""
import os
import sys
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
mode = sys.argv[1] if len(sys.argv) > 1 else "kit"
extra = [a for a in sys.argv[2:] if a.startswith("--/")]
t0 = time.time()

if mode == "kit":
    sys.argv = [sys.argv[0]] + extra
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    print(f"[boot] SimulationApp 뜸 {time.time() - t0:.1f}s", flush=True)
    import carb
    import omni.physx
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    cs = carb.settings.get_settings()
    for k in ("/renderer/active", "/renderer/enabled", "/app/renderer/enabled", "/rtx/rendermode", "/physics/cudaDevice"):
        print(f"[boot] {k} = {cs.get(k)!r}", flush=True)
    stage = omni.usd.get_context().get_stage()
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    cube = UsdGeom.Cube.Define(stage, "/World/cube")
    cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 2))
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    ph = omni.physx.get_physx_interface()
    ph.start_simulation()
    for i in range(60):
        ph.update_simulation(1.0 / 60.0, i / 60.0)
    pos = UsdGeom.Xformable(cube.GetPrim()).ComputeLocalToWorldTransform(0).ExtractTranslation()
    tr = ph.get_rigidbody_transformation("/World/cube")
    print(f"[boot] 1초 뒤 상자 위치 {tr['position'] if tr else pos}", flush=True)
    app.close()
else:
    import omnigibson as og
    from omnigibson.macros import gm

    gm.HEADLESS = True
    gm.USE_GPU_DYNAMICS = False
    cfg = {"scene": {"type": "Scene"}, "objects": [{"type": "PrimitiveObject", "name": "box", "primitive_type": "Cube",
                                                     "size": 0.1, "position": [0, 0, 1.0]}]}
    env = og.Environment(configs=cfg)
    print(f"[boot] OmniGibson 환경 뜸 {time.time() - t0:.1f}s", flush=True)
    for _ in range(30):
        og.sim.step()
    print(f"[boot] 30 스텝 뒤 상자 {env.scene.object_registry('name', 'box').get_position_orientation()[0]}", flush=True)
    og.shutdown()
