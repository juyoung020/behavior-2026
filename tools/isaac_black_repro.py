"""OmniGibson 없이 Isaac Sim 5.1 만으로 카메라 RGB 가 검게(RGBA 전부 0) 나오는지 센다 -- 검은 프레임 원인 가르기용.

    conda activate behavior
    python C:\\behavior-2026\\tools\\isaac_black_repro.py [--kit omnigibson|default] [--cams 3] [--res 224] [--frames 300]
                                                        [--renderer RaytracedLighting|PathTracing] [--no-play] [--set /키=값 ...]

- 공식 평가기와 같은 구성: 카메라마다 render product 하나(비타일), replicator 'rgb' annotator(호스트 복사, device=cpu),
  'distance_to_image_plane' annotator, 헤드리스, 실시간 RTX. 한 프레임 = simulation_app.update() 한 번(OmniGibson 의 og.sim.step 과 같음).
- --kit omnigibson : OmniGibson 이 쓰는 앱 설정(omnigibson_5_1_0.kit — Windows 에서도 Vulkan 강제)으로 띄운다.
  --kit default    : Isaac Sim 기본 앱(isaacsim.exp.base.python.kit, Windows 기본 그래픽 API).
- 장면: 바닥 + 색 상자 몇 개 + 조명(검은 화면이 '진짜로 어두운' 것과 헷갈리지 않게 밝게).
- 출력: 카메라별 검은 프레임 수와 처음 몇 개 위치, depth 가 전부 0 인 프레임 수.
"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--kit", choices=("omnigibson", "default"), default="omnigibson")
ap.add_argument("--cams", type=int, default=3)
ap.add_argument("--res", type=int, default=224)
ap.add_argument("--frames", type=int, default=300)
ap.add_argument("--renderer", default="RaytracedLighting")
ap.add_argument("--no-play", action="store_true", help="타임라인을 재생하지 않는다 (replicator 는 재생 중에만 찍는다 -- 진단용)")
ap.add_argument("--set", action="append", default=[], help="/키=값 (시작 뒤 carb 설정)")
a = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

exp = ""
if a.kit == "omnigibson":
    exp = os.path.join(os.environ.get("EXP_PATH", ""), "omnigibson_5_1_0.kit")
    if not os.path.exists(exp):
        import isaacsim

        exp = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "omnigibson_5_1_0.kit")
    assert os.path.exists(exp), f"OmniGibson kit 없음: {exp} (OmniGibson 을 한 번 띄우면 Isaac apps 폴더로 복사된다)"
cfg = {"headless": True, "renderer": a.renderer, "width": 640, "height": 480}
# 캐시·로그를 평가기와 같은 곳(OmniGibson portable root)에 쓰게 한다 -- conda 환경 안 isaacsim\kit\ 에 새 잔재를 안 남기려고
PORTABLE = os.path.join("C:" + os.sep, "behavior-2026", "BEHAVIOR-1K", "OmniGibson", "appdata", "local")
if os.path.isdir(PORTABLE):
    sys.argv += ["--portable-root", PORTABLE]
app = SimulationApp(cfg, experience=exp) if exp else SimulationApp(cfg)

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

cs = carb.settings.get_settings()
for kv in a.set:
    k, v = kv.split("=", 1)
    val = {"true": True, "false": False}.get(v.lower(), v)
    try:
        val = int(val) if not isinstance(val, bool) else val
    except ValueError:
        pass
    cs.set(k, val)
watch = ["/app/gatherRenderResults", "/app/settings/fabricDefaultStageFrameHistoryCount", "/omni/replicator/captureOnPlay",
         "/rtx/rendermode", "/rtx/post/aa/op", "/rtx-transient/dlssg/enabled", "/app/vulkan", "/app/asyncRendering",
         "/app/hydraEngine/waitIdle", "/renderer/multiGpu/enabled"]
print("[repro] 설정:", {k: cs.get(k) for k in watch}, flush=True)

stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World")
ground = UsdGeom.Cube.Define(stage, "/World/ground")
ground.AddScaleOp().Set(Gf.Vec3f(10, 10, 0.05))
ground.CreateDisplayColorAttr([(0.6, 0.6, 0.6)])
for i, col in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]):
    c = UsdGeom.Cube.Define(stage, f"/World/box{i}")
    c.AddTranslateOp().Set(Gf.Vec3f(-1.5 + i, 0, 0.5))
    c.AddScaleOp().Set(Gf.Vec3f(0.3, 0.3, 0.3))
    c.CreateDisplayColorAttr([col])
UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(1000)
UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(3000)

rps, rgbs, deps = [], [], []
for i in range(a.cams):
    cam = rep.create.camera(position=(3 + i, -3, 2), look_at=(0, 0, 0.5))
    rp = rep.create.render_product(cam, (a.res, a.res))
    r = rep.AnnotatorRegistry.get_annotator("rgb")
    r.attach([rp])
    d = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    d.attach([rp])
    rps.append(rp)
    rgbs.append(r)
    deps.append(d)

if not a.no_play:  # OmniGibson 처럼 타임라인 재생 (omni.replicator.captureOnPlay = true 라 재생 중에만 annotator 가 찍힌다)
    import omni.timeline

    omni.timeline.get_timeline_interface().play()
for _ in range(20):  # 데우기 (OmniGibson 도 장면 로드 뒤 여러 번 렌더)
    app.update()

black = [[] for _ in range(a.cams)]
dblack = [0] * a.cams
t0 = time.time()
for f in range(a.frames):
    app.update()
    for i in range(a.cams):
        x = rgbs[i].get_data(device="cpu")
        x = x["data"] if isinstance(x, dict) else x
        if x.size == 0 or x.max() == 0:
            black[i].append(f)
        y = deps[i].get_data(device="cpu")
        y = y["data"] if isinstance(y, dict) else y
        if y.size == 0 or not np.any(y):
            dblack[i] += 1
dt = time.time() - t0
print(f"[repro] kit={a.kit} cams={a.cams} res={a.res} renderer={a.renderer} play={not a.no_play} set={a.set} "
      f"frames={a.frames} ({a.frames / dt:.1f} fps)", flush=True)
for i in range(a.cams):
    print(f"[repro] 카메라 {i}: RGB 검정 {len(black[i])}/{a.frames} 처음 {black[i][:12]}  depth 0 {dblack[i]}", flush=True)
print(f"[repro] 합계 RGB 검정 {sum(len(b) for b in black)}/{a.frames * a.cams}", flush=True)
app.close()
