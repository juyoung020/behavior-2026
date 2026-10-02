# archive — 지금 안 쓰는 모듈

지금 파이프라인(공식 평가기 + 네이티브 π0.5 엔진, 물체 기억 scene_graph, 계획기·move_robot, 시뮬 엔진·fasteval, 학습 코드)에서
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

## 되살리기

```bash
git mv archive/<원래 경로> <원래 경로>      # 예: git mv archive/src/agent/probe src/agent/probe
```

그 뒤 빌드 파일·README 폴더 표·문서 경로를 다시 원래 자리로 고친다. 옮긴 커밋은 `git log --follow -- archive/<경로>` 로 찾는다.

## 새로 옮길 때

- 안 쓰는 근거(참조 검색, 빌드 파일, plan.md 5절 진행 표)를 확인하고 `git mv` 로 `archive/<원래 경로>` 에 옮긴다. 지우지 않는다.
- 남은 참조(빌드 파일·스크립트·README 폴더 표·문서)를 고친다. 위 표에 한 줄(모듈 | 이유)을 더한다.
