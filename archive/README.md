# archive — 지금 안 쓰는 모듈

지금 파이프라인(공식 평가기, 물체 기억 scene_graph, 계획기·move_robot, 시뮬 엔진·fasteval)에서
쓰지 않게 된 모듈을 지우지 않고 옮겨 둔 곳이다. 폴더 구조는 원래 경로 그대로다(`archive/<원래 경로>`).
안의 코드·스크립트는 옮기기 전 그대로 두었다(안의 경로도 옛 경로). 빌드·시험에는 들어가지 않는다(`COLCON_IGNORE`, 상위 Cargo·CMake 없음).

지난 문서는 따로 [docs/archive/](../docs/archive/README.md) 에 있다.

## 옮긴 것 (2026-10-03)

| 모듈 | 이유 |
|---|---|
| `src/agent/probe/` (Rust), `tools/run_probe.sh`, `tools/probe_prep.py`, `tools/probe_infer.py` | 지시 형식 오프라인 실험 도구. 실험이 끝나고 결론이 문서로 남았다([docs/실험_지시형식_오프라인.md](../docs/실험_지시형식_오프라인.md)). 다시 학습한 뒤 같은 실험을 할 때 되살린다 |
| `src/vla/comet/` (patches), `tools/setup/setup_comet_wsl.sh`, `tools/run_comet_server.sh`, `tools/comet_reasoner_env.sh`, `tools/run_comet_offline_test.sh` | 2025 2위 openpi-comet 저장소(JAX 서버 + Comet 계획기 반복문)를 WSL 에서 돌리던 패치·실행기. Comet 가중치는 이제 네이티브 엔진으로 평가기 안에서 돈다(`data/pi05_native/comet_*.pi05w`, `src/vla/pi05_native/glue/run_eval_native.py`). 계획은 우리 계획기(`src/agent/planner`)가 맡는다 |
| `tools/isaac_black_repro.py`, `tools/og_black_repro.py`, `tools/gpu_hog.py`, `tools/gpu_mem_sampler.ps1`, `tools/linux_black_frame_repro.sh` | 검은 프레임 (B) 원인 가르기 도구(Windows). 리눅스 PC 에서 검은 프레임 0 으로 쟁점이 아니다(plan.md 4.2). 검출 `tools/black_frame_check.py` 는 그대로 쓴다 |
| `tools/setup/run_download_ckpts.ps1`, `run_install_isaacsim.ps1`, `run_stream_assets.ps1`, `install_isaacsim.py`, `fix_behavior_env.sh`, `setup_wsl.sh`, `extract_ckpts_wsl.sh` | Windows·WSL 설치 스크립트. 리눅스는 `tools/setup/linux_setup.sh`·`train_4090.sh`([docs/Linux_설치.md](../docs/Linux_설치.md))가 맡는다 |
| `tools/setup/fetch_cosmos_wsl.sh` | GR00T N1.7 백본(Cosmos-Reason2-2B) 받기. GR00T 는 계획에 없다 |
| `src/vla/pi05_native/build_windows.bat` | Windows(MSVC) 빌드. 리눅스는 `build_linux.sh` |
| `src/vla/pi05_native/glue/run_eval_native.ps1` | Windows 평가기 실행기. 리눅스는 같은 폴더 `run_eval_native.py` |
| `tools/exp_run.ps1` | 리눅스판 `tools/exp_run.py` 로 대체 |
| `tools/run_eval_radio.ps1` | 리눅스판 `tools/run_eval_radio.sh` 로 대체 |
| `tools/run_replay_eval.ps1` | 리눅스판 `tools/run_replay_eval.sh` 로 대체 |
| `tools/subpack.ps1` | 리눅스판 `tools/subpack.py` 로 대체 |
| `tools/gpu_lock.ps1` | 리눅스판 `tools/gpu_lock.sh` 로 대체 |
| `src/vla/pi05_native/exp/run_insurance.ps1` | 리눅스판 `src/vla/pi05_native/exp/run_insurance.sh` 로 대체 |
| `src/sim/integ/run_eval_integ.ps1` | 리눅스판 `src/sim/integ/run_eval_integ.sh` 로 대체 |
| `src/sim/integ/wsl_stack.sh` | 리눅스판 `src/sim/integ/simlink_stack.sh` 로 대체 |
| `src/sim/integ/wsl_cleanup.sh` | 리눅스판 `src/sim/integ/simlink_cleanup.sh` 로 대체 |
| `src/sim/engine/capture/run_capture.ps1` | 리눅스판 `src/sim/engine/capture/run_capture.sh` 로 대체 |
| `src/sim/engine/capture/render_scene_wsl.py` | 리눅스판 `src/sim/engine/capture/render_scene.py` 로 대체 |
| `src/sim/engine/capture/render_scene_wsl.sh` | 리눅스판 `src/sim/engine/capture/render_scene.sh` 로 대체 |
| `src/sim/engine/tests/render/gpu_session.ps1` | 리눅스판 `src/sim/engine/tests/render/gpu_session.sh` 로 대체 |
| `src/sim/engine/tests/render/capture/capture_with_lock.ps1` | 리눅스판 `src/sim/engine/tests/render/capture/capture_with_lock.sh` 로 대체 |
| `src/sim/engine/tests/render/capture/process_capture.ps1` | 리눅스판 `src/sim/engine/tests/render/capture/process_capture.sh` 로 대체 |
| `src/sim/engine/tests/render/capture/run_render_capture.ps1` | 리눅스판 `src/sim/engine/tests/render/capture/run_render_capture.sh` 로 대체 |
| `tools/setup/download_top_ckpts_wsl.sh` | 리눅스판 `tools/setup/download_top_ckpts.sh` 로 대체 |
| `tools/setup/fetch_qwen35_gguf_wsl.sh` | 리눅스판 `tools/setup/fetch_qwen35_gguf.sh` 로 대체 |
| `tools/setup/setup_llamacpp_cuda_wsl.sh` | 리눅스판 `tools/setup/setup_llamacpp_cuda.sh` 로 대체 |

## 옮긴 것 (2026-10-05, 검출기 결정 — ObjectSAM + SigLIP 2 + objprob)

YOLOE·YOLO26s 엔진을 `~/ovdet_models/archive` 로 보관하고 ObjectSAM(YOLO26n 학생)으로 정한 뒤, YOLOE 에만 쓰던 것.

| 모듈 | 이유 |
|---|---|
| `src/scene_graph/ovdet/tools/export_yoloe.py` | YOLOE ONNX 내보내기(글 프롬프트 임베딩을 머리에 굳힘). ObjectSAM 은 robot-agent `training/fastsam/export.sh`·`build_engine.py` |
| `src/scene_graph/ovdet/tools/ovdet_eval.py`, `src/scene_graph/ovdet/scripts/eval_linux.sh`, `eval_conf.sh` | YOLOE 머리·프롬프트 비교(데모 프레임). ObjectSAM 검출 단계 평가는 robot-agent `training/fastsam/eval_det.py`, 끝에서 끝은 `src/scene_graph/tools/realbag/objprob_eval.py` |
| `src/scene_graph/ovdet/tools/ref_check.py` | YOLOE Ultralytics FP32 기준 비교(돌린 적 없음) |
| `outputs/mem_yolo26s_20261003_044921/run.sh` | 보관한 yolo26s-seg 엔진 경로로 도는 옛 기억 판 실행기 |

남긴 것(아직 씀 또는 불분명): `ovdet/config/task_prompts.txt`(글루가 과제 이름을 낱말 표에 더함)·`make_task_prompts.py`·`vocab_all.txt`(그 표를 만든 도구 — 불분명, 남김), `ovdet/tools/build_engines.py`(엔진 빌드 일반), scenemap 옛 이름 규칙(`--no-objprob`·`SGRT_OBJPROB=0`·시험이 씀), libsgrt 프롬프트 길(`SGRT_PROMPT`, 보관 엔진을 고를 때), `runtime/tools/dom_bench_det.cpp`(기본 엔진을 ObjectSAM 으로 바꿔 계속 씀), `tools/realbag/detcmp_*`(objprob_eval 이 씀).

Windows·WSL 스크립트를 리눅스판으로 옮긴 표(쓰는 법·확인한 것)는 [tools/README.md](../tools/README.md).

## 지운 것 (2026-10-06, π0.5 버림)

π0.5 는 쓰지 않는다(VLA = RecallVLA, robot-agent `training/vla`). 옮기지 않고 **지웠다**(git 기록에는 남음). 위 표 중 지운 줄:
`src/agent/probe/`·`tools/run_probe.sh`·`probe_prep.py`·`probe_infer.py`(π0.5 지시 형식 실험), `src/vla/comet/`·`setup_comet_wsl.sh`·`run_comet_server.sh`·
`comet_reasoner_env.sh`·`run_comet_offline_test.sh`(openpi-comet), `src/vla/pi05_native/build_windows.bat`·`glue/run_eval_native.ps1`·`exp/run_insurance.ps1`,
`src/sim/integ/run_eval_integ.ps1`, `tools/setup/run_download_ckpts.ps1`·`extract_ckpts_wsl.sh`·`download_top_ckpts_wsl.sh`(π0.5 체크포인트 받기).
되살리려면 `git log --diff-filter=D --name-only -- <경로>` 로 지운 커밋을 찾아 그 앞 커밋에서 꺼낸다.

## 되살리기

```bash
git mv archive/<원래 경로> <원래 경로>      # 예: git mv archive/src/agent/probe src/agent/probe
```

그 뒤 빌드 파일·README 폴더 표·문서 경로를 다시 원래 자리로 고친다. 옮긴 커밋은 `git log --follow -- archive/<경로>` 로 찾는다.

## 새로 옮길 때

- 안 쓰는 근거(참조 검색, 빌드 파일, plan.md 5절 진행 표)를 확인하고 `git mv` 로 `archive/<원래 경로>` 에 옮긴다. 지우지 않는다.
- 남은 참조(빌드 파일·스크립트·README 폴더 표·문서)를 고친다. 위 표에 한 줄(모듈 | 이유)을 더한다.
