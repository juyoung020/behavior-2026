"""OmniGibson + 빈 장면/radio 장면 + 공식 R1Pro 로봇(카메라 3 대)만으로 카메라 RGB 가 검게 나오는지 센다 -- 검은 프레임 원인 가르기용.

    conda activate behavior  (KMP_DUPLICATE_LIB_OK=TRUE, OMNI_KIT_ACCEPT_EULA=YES)
    python C:\\behavior-2026\\tools\\og_black_repro.py [--steps 600] [--res 224] [--scene empty|radio]
        [--diag OUT.json] [--reread] [--set /carb/key=val ...] [--kit-arg=--/key=val ...]

- 평가기와 같게: 헤드리스, 뷰어 카메라 없음(gm.RENDER_VIEWER_CAMERA=False), 물리 120 / 렌더 30 / 행동 30 Hz,
  공식 eval\\r1pro.yaml(eval 블록만 뺌) + 카메라별 해상도(평가기 load_env 와 같은 sensor_config 덮어쓰기), 0 행동.
- --scene empty : 빈 장면(Scene) -- 큰 집 장면·과제·물체 상태가 없는 조건.  --scene radio : 평가기와 같은 radio 장면(부분 로드, 과제 없음).
- 스텝마다 env.step -> 관측의 카메라 RGB 가 전부 0 인지 센다.
- --diag OUT.json : 스텝·카메라마다 [검정, 알파 최대, ReferenceTime(이 데이터가 몇 시각 프레임인지), 시뮬레이터 시각,
  Kit 갱신 번호, 영상 crc32] 를 적는다(시간 동기화 확인). ReferenceTime annotator 를 카메라마다 하나 더 붙인다(진단 전용).
- --reread : 검은 스텝에서 (1) 렌더 없이 annotator 를 다시 읽고 (2) og.sim.render() 한 번 더 한 뒤 읽는다 -- 버퍼가 늦게 채워지는지 가르기.
  (렌더를 한 번 더 하므로 그 뒤 순서가 바뀐다. 진단 전용.)
- --reattach N : N 스텝에서 카메라마다 rgb annotator 를 OmniGibson 자체 함수로 떼었다 다시 붙인다 -- 규칙적 검정이 annotator(합성 데이터 그래프) 상태인지
  렌더 결과(RTX) 상태인지 가르기. 다시 붙인 직후 몇 스텝은 데이터가 비어 있을 수 있다(빈 배열은 따로 센다).
- --set KEY=VAL : 환경을 만든 뒤 carb 설정을 바꾼다.  --kit-arg=ARG : Kit 시작 인자를 덧붙인다(평가기 도구와 같은 방법).
"""
import argparse
import json
import os
import time
import zlib

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--res", type=int, default=224)
ap.add_argument("--scene", choices=("empty", "radio"), default="empty")
ap.add_argument("--diag", default="")
ap.add_argument("--reread", action="store_true")
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--reattach", type=int, default=-1, help="이 스텝에서 카메라마다 rgb annotator 를 떼었다 다시 붙인다(진단)")
a, rest = ap.parse_known_args()
kit_args = [x[len("--kit-arg="):] for x in rest if x.startswith("--kit-arg=")]
if [x for x in rest if not x.startswith("--kit-arg=")]:
    ap.error(f"모르는 인자: {rest}")

import omnigibson as og  # noqa: E402
import torch as th  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

gm.HEADLESS = True
gm.RENDER_VIEWER_CAMERA = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = False

if kit_args:
    import omnigibson.simulator as S

    orig_launch = S._launch_app

    def launch_with_args(*la, **lkw):
        import isaacsim

        orig_init = isaacsim.SimulationApp.__init__

        def init(self, launch_config=None, *ia, **ikw):
            cfg = dict(launch_config or {})
            cfg["extra_args"] = list(cfg.get("extra_args", [])) + kit_args
            print(f"[og-repro] kit extra_args={cfg['extra_args']}", flush=True)
            return orig_init(self, cfg, *ia, **ikw)

        isaacsim.SimulationApp.__init__ = init
        try:
            return orig_launch(*la, **lkw)
        finally:
            isaacsim.SimulationApp.__init__ = orig_init

    S._launch_app = launch_with_args

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

import carb  # noqa: E402
import omni.kit.app  # noqa: E402

cs = carb.settings.get_settings()
for kv in a.set:
    k, v = kv.split("=", 1)
    cur = cs.get(k)
    val = {"true": True, "false": False}.get(v.lower(), v)
    if not isinstance(val, bool):
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
    cs.set(k, val)
    print(f"[og-repro] 설정 {k}: {cur!r} -> {cs.get(k)!r}", flush=True)

env.reset()
r = env.scenes[0].robots[0]  # v3.9.3: env.robots 는 환경마다 목록
keys = [k for k in r.sensors if "Camera" in k]
black = {k: [] for k in keys}
zero = th.zeros(r.action_dim)
app = omni.kit.app.get_app()

reft = {}
if a.diag:
    import omni.replicator.core as rep

    with og.sim.editing_usd():  # render var 추가 = USD 편집 (OmniGibson 의 USD 편집 감시에 걸리지 않게)
        for k in keys:
            ann = rep.AnnotatorRegistry.get_annotator("ReferenceTime")
            ann.attach([r.sensors[k].render_product])
            reft[k] = ann


def _reftime(k):
    try:
        d = reft[k].get_data()
        num, den = d.get("referenceTimeNumerator"), d.get("referenceTimeDenominator")
        return float(num) / float(den) if den else None
    except Exception as e:  # 진단 실패는 기록만
        return f"{type(e).__name__}: {e}"[:60]


def _raw(k):
    d = r.sensors[k]._annotators["rgb"].get_data()
    return d["data"] if isinstance(d, dict) else d


rows = {k: [] for k in keys}
reread = []
empty = {k: [] for k in keys}
t0 = time.time()
for s in range(a.steps):
    if s == a.reattach:
        for k in keys:
            r.sensors[k]._remove_modality_from_backend("rgb")
            r.sensors[k]._add_modality_to_backend("rgb")
        # USD 편집(render var 추가)이 PhysX 텐서 뷰를 무효로 만든다 -> OmniGibson 이 물체를 새로 넣을 때 쓰는 함수로 다시 잡는다
        og.sim.update_handles()
        print(f"[og-repro] 스텝 {s}: rgb annotator 다시 붙임", flush=True)
    obs_list = env.step(zero)[0]  # v3.9.3: 환경마다 관측 딕셔너리 목록
    ro = obs_list[0][r.name]
    upd = app.get_update_number()
    simt = float(og.sim.current_time)
    for k in keys:
        rgb = ro[k]["rgb"]
        if rgb.numel() == 0:  # 다시 붙인 직후 등 데이터 없음
            empty[k].append(s)
            continue
        b = int(rgb.max()) == 0
        if b:
            black[k].append(s)
        if a.diag:
            ch = rgb.shape[-1] if rgb.ndim == 3 else 0
            amax = int(rgb[..., 3].max()) if ch == 4 else -1
            crc = zlib.crc32(rgb.cpu().contiguous().numpy().tobytes())
            rows[k].append([s, b, amax, _reftime(k), simt, upd, crc])
        if b and a.reread:
            again = _raw(k)
            b1 = int(again.max()) == 0
            og.sim.render()
            after = _raw(k)
            b2 = int(after.max()) == 0
            reread.append([k, s, b1, b2, _reftime(k) if a.diag else None])
dt = time.time() - t0
print(f"[og-repro] scene={a.scene} res={a.res} steps={a.steps} ({a.steps / dt:.1f} steps/s) set={a.set} kit={kit_args}", flush=True)
for k in keys:
    print(f"[og-repro] {k}: RGB 검정 {len(black[k])}/{a.steps} 처음 {black[k][:12]}", flush=True)
    if a.reattach >= 0:
        pre = [x for x in black[k] if x < a.reattach]
        post = [x for x in black[k] if x >= a.reattach]
        print(f"[og-repro]   다시 붙이기 전 {len(pre)}/{a.reattach} (나머지3 {sorted(set(x % 3 for x in pre))}) · 뒤 {len(post)}/{a.steps - a.reattach}"
              f" (처음 {post[:8]}) · 빈 데이터 스텝 {empty[k][:8]}", flush=True)
if a.reread:
    n = len(reread)
    print(f"[og-repro] 다시 읽기(렌더 없이) 검정 {sum(x[2] for x in reread)}/{n}, 렌더 한 번 더 뒤 검정 {sum(x[3] for x in reread)}/{n}", flush=True)
if a.diag:
    # 요약: 검은 스텝과 안 검은 스텝에서 (시뮬레이터 시각 - 프레임 기준 시각) 분포, 같은 crc 가 바로 앞 스텝과 같은 비율(영상 반복)
    for k in keys:
        lag = {True: [], False: []}
        rep_same = {True: 0, False: 0}
        prev = None
        for s, b, amax, ref, simt, upd, crc in rows[k]:
            if isinstance(ref, float):
                lag[b].append(round(simt - ref, 4))
            if prev is not None and crc == prev:
                rep_same[b] += 1
            prev = crc
        amaxb = sorted({x[2] for x in rows[k] if x[1]})
        print(f"[og-repro] {k}: 지연(시뮬시각-프레임시각) 검정 {sorted(set(lag[True]))[:6]} / 정상 {sorted(set(lag[False]))[:6]}"
              f" | 앞 스텝과 영상 같음 검정 {rep_same[True]} 정상 {rep_same[False]} | 검은 프레임 알파 최대 {amaxb}", flush=True)
    with open(a.diag, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"args": vars(a), "kit_args": kit_args, "cols": ["step", "black", "alpha_max", "ref_time", "sim_time", "kit_update", "crc32"],
                   "rows": rows, "reread": reread}, f)
    print(f"[og-repro] 진단 저장 {a.diag}", flush=True)
og.shutdown()
