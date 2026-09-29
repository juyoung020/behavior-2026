"""OmniGibson + 빈 장면 + 공식 R1Pro 로봇(카메라 3 대)만으로 카메라 RGB 가 검게 나오는지 센다 -- 검은 프레임 원인 가르기용.

    conda activate behavior  (KMP_DUPLICATE_LIB_OK=TRUE, OMNI_KIT_ACCEPT_EULA=YES)
    python C:\\behavior-2026\\tools\\og_black_repro.py [--steps 600] [--res 224] [--scene empty|radio]

- 평가기와 같게: 헤드리스, 뷰어 카메라 없음(gm.RENDER_VIEWER_CAMERA=False), 물리 120 / 렌더 30 / 행동 30 Hz,
  공식 eval\\r1pro.yaml(eval 블록만 뺌) + 카메라별 해상도(평가기 load_env 와 같은 sensor_config 덮어쓰기), 0 행동.
- --scene empty : 빈 장면(Scene) -- 큰 집 장면·과제·물체 상태가 없는 조건.  --scene radio : 평가기와 같은 radio 장면(부분 로드, 과제 없음).
- 스텝마다 env.step -> 관측의 카메라 RGB 가 전부 0 인지 센다.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--res", type=int, default=224)
ap.add_argument("--scene", choices=("empty", "radio"), default="empty")
a = ap.parse_args()

import omnigibson as og  # noqa: E402
import torch as th  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

gm.HEADLESS = True
gm.RENDER_VIEWER_CAMERA = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = False

robot = OmegaConf.to_container(OmegaConf.load(os.path.join(os.path.dirname(og.__file__), "eval", "r1pro.yaml")))
cams = robot.pop("eval")["camera_sensor_names"]
robot["obs_modalities"] = ["proprio", "rgb"]
robot["sensor_config"] = robot.get("sensor_config", {})
for sensor_name in cams.values():
    robot["sensor_config"][sensor_name.split(":", 1)[1]] = {"sensor_kwargs": {"image_height": a.res, "image_width": a.res}}
cfg = {"env": {"action_frequency": 30, "rendering_frequency": 30, "physics_frequency": 120}, "robots": [robot]}
if a.scene == "empty":
    cfg["scene"] = {"type": "Scene"}
else:
    cfg["scene"] = {"type": "InteractiveTraversableScene", "scene_model": "house_double_floor_lower",
                    "load_room_instances": ["corridor_0", "garden_0", "kitchen_0", "living_room_0"]}
env = og.Environment(configs=cfg)
env.reset()
r = env.robots[0]
keys = [k for k in r.sensors if "Camera" in k]
black = {k: [] for k in keys}
zero = th.zeros(r.action_dim)
t0 = time.time()
for s in range(a.steps):
    obs_list = env.step(zero)[0]  # v3.9.3: 환경마다 관측 딕셔너리 목록
    ro = obs_list[0][r.name]
    for k in keys:
        rgb = ro[k]["rgb"]
        if int(rgb.max()) == 0:
            black[k].append(s)
dt = time.time() - t0
print(f"[og-repro] scene={a.scene} res={a.res} steps={a.steps} ({a.steps / dt:.1f} steps/s)", flush=True)
for k in keys:
    print(f"[og-repro] {k}: RGB 검정 {len(black[k])}/{a.steps} 처음 {black[k][:12]}", flush=True)
og.shutdown()
