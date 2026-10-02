# 작업 PC 환경 — Ubuntu 22.04 + RTX 5070 Ti + Isaac Sim 5.1 + BEHAVIOR-1K v3.9.3-post1

프로젝트는 리눅스 PC 한 대에서 한다. 지금 작업 PC 는 `jy-desktop`(10-03 부터, 3.1 절). 그 전(10-02)에는 `ad17-MS-7E01`(RTX 4090)에서 했다 — 아래 1.1 절·3 절·4 절의 10-02 결과는 그 PC 의 기록이다.
주최 측 성능 측정 장비도 "Ubuntu 22.04.5 LTS"([raw/site_challenge_evaluation.md](raw/site_challenge_evaluation.md)).
지난 계획·기록은 [archive/windows/Linux_설치.md](archive/windows/Linux_설치.md).

## 1. PC

| 항목 | 값 |
|---|---|
| 호스트 | `jy-desktop`, 저장소 `~/robot-agent/src/behavior-2026` (`robot-agent` 의 서브모듈) |
| OS | Ubuntu 22.04 |
| GPU | RTX 5070 Ti 16 GB (Blackwell, sm_120) 한 장 |
| 드라이버 | **580.178.04 (`nvidia-driver-580-open`)** — 595.91.07 에서 내림(3.1 절) |
| CUDA 툴킷 | `/usr/local/cuda-12.8` 을 쓴다 (13.2 도 깔려 있으나 쓰지 않음 — 이유는 3.1 절) |
| conda `behavior` | OmniGibson 3.9.3 / Isaac Sim 5.1 |

### 1.1 앞 PC (`ad17-MS-7E01`, 10-02 까지) — 기록

| 항목 | 값 |
|---|---|
| 호스트 | `ad17-MS-7E01`, 사용자 `ad17`, 저장소 `~/behavior-2026` |
| OS | Ubuntu 22.04.5 LTS, 커널 6.8.0-138-generic, Secure Boot 꺼짐 |
| GPU | RTX 4090 24 GB (Ada, sm_89) 한 장 |
| 드라이버 | **580.178.04 (`nvidia-driver-580-open`)** — 10-02 에 535.309 에서 올림 |
| CUDA 툴킷 | `/usr/local/cuda-11.8` 만 있었음 |
| RAM · 디스크 | 31 GB · `/` 432 GB (10-02 남은 공간 약 140 GB) |
| BEHAVIOR-1K | `bd049de` (v3.9.3-post1, 공식 압축본과 파일 단위 동일 — `refs/BEHAVIOR-1K_v3.9.3-post1_official/`) |
| conda `behavior` | isaacsim 5.1.0.0, torch 2.7.0+cu128, warp-lang 1.12.0 |
| 재부팅 자동 시작 | crontab `@reboot ~/.local/bin/claude-boot.sh` → tmux 세션 `claude`(창 `main` = `claude -c`, `rc` = `claude rc`), 기록 `~/.local/state/claude-boot.log` |

## 2. 설치 순서

1. `bash tools/setup/linux_setup.sh driver` → 재부팅 → `nvidia-smi` 확인. 드라이버는 **580 계열**(535 는 아래 3 절의 오류).
2. `bash tools/setup/linux_setup.sh base` → 새 셸 → `gh auth login`.
3. `bash tools/setup/linux_setup.sh repos` (우리 저장소 + BEHAVIOR-1K `bd049de` = v3.9.3-post1).
4. `bash tools/setup/linux_setup.sh behavior` — 공식 `./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval`
   (+ 약관 동의 인자 `--accept-conda-tos --accept-nvidia-eula --accept-dataset-tos`). 이어서 `pip install av "numpy<2"`(검은 프레임 검사 도구가 씀).
5. `import omnigibson` 이 `warp` 를 요구하면 `pip install warp-lang==1.12.0`(OmniGibson `primitives` 옵션과 같은 버전).
6. `bash tools/setup/linux_setup.sh check` → `bash tools/setup/linux_first_check.sh 3` (4 절).
- git 신원: `juyoung020 <151780134+juyoung020@users.noreply.github.com>`. 커밋 훅: `git config core.hooksPath tools/git-hooks`.
- 복호화 키(`BEHAVIOR-1K/datasets/omnigibson.key`)·토큰(`~/.config/behavior-2026/`)은 커밋하지 않는다.

## 3. 겪은 것 (1~4 는 앞 PC `ad17` 기록; 기록: `outputs/linux_try_*`, `outputs/linux_first_*`, `logs/linux_dl/`)

1. 10-01: `import omnigibson` 이 `warp` 없음으로 실패 → `warp-lang==1.12.0` 설치.
2. 10-01~02, **드라이버 535**: 매번 Isaac Sim 이 "The currently installed NVIDIA graphics driver is unsupported or has known issues" 경고.
   판마다 실패 모양이 달랐다 — 판 시작 전 조용히 종료, 리셋에서 카메라 RGB 관측이 비어 `Observation space does not match`,
   평가기 생성 중 warp 오류(`wp.matrix(pos, rot, scale)` 가 warp 1.12 에서 없어짐, Isaac Sim fabric 코드).
3. 10-02 **드라이버 580.178.04-open 으로 올림**(apt, 535 패키지는 580 을 가리키는 전환용으로 바뀜) → 재부팅 뒤 **같은 warp 1.12 그대로 정상 동작**.
   위 2 의 오류들은 535 드라이버 탓으로 본다(warp 오류가 드라이버에 따라 갈린 이유는 확인 못 함 — 추정).
4. 10-02 π0.5 radio 체크포인트(Google Drive, [베이스라인.md](베이스라인.md))를 이 PC 로 받는 중 — Drive 속도 약 1 MB/s, DNS 일시 실패로 끊긴 적 있음(`logs/linux_dl/gdrive_pi05.log`).

### 3.1 지금 작업 PC (`jy-desktop`, RTX 5070 Ti 16 GB sm_120, 저장소 `~/robot-agent/src/behavior-2026`, 10-03)

- 저장소는 `robot-agent` 의 서브모듈. 옛 경로를 쓰는 스크립트를 위해 심볼릭 링크 `~/behavior-2026`, `/mnt/c/behavior-2026` → 서브모듈.
- CUDA 는 **12.8**(저장소 스크립트 기준). 13.2 로 바꿔 봤다가 되돌림: 13.2 는 PhysX 에 `cuCtxCreate_v4` 패치가 필요하고 `libPhysXGpu_64.so` 링크가 깨지며,
  드라이버 580(CUDA 13.0)이 13.2 PTX 를 JIT 못 해("PTX was compiled with an unsupported toolchain") SASS 없는 빌드가 GPU 에서 안 돈다. clang 14 + `libstdc++-12-dev` 가 있어야 clang 링크가 된다.
- 엔진 시험 CMake 는 `CMAKE_CUDA_ARCHITECTURES` 기본을 `project()` 앞에서 120 으로 정한다(CUDA 13 기본 75 이면 sm_75 만 빌드되던 문제).
- **드라이버 595.91.07 에서 Isaac Sim 5.1 이 시작 때 `librtx.scenedb.plugin.so` 세그폴트**(맨 SimulationApp 도, iGPU 막아도 같음 — 595 계열 알려진 문제) → `nvidia-driver-580-open` 580.178.04 로 내림.
- 엔진 시험(CUDA 12.8, sm_120, 10-03): PhysX 대조 CPU·GPU 시험 전부 비트 동일(관절체·접촉·조인트·풀이·섬·렌더 층1=층2·초월함수 전수). 녹화 데이터(`~/engine-data`)가 필요한 시험 30 개는 이 PC 에 데이터가 없어 못 돌림. `test_aos_diff_1lane` 은 문서대로 ±0 두 함수만 다름(예상). `test_render_math` 는 인자 없이(간격 1) 돌리면 fexp2 입력 수가 int 를 넘어 죽음 — `test_render_math 4` 로 통과. `test_pcm_convex_gpu` 는 커널 스택 50,416 B 가 한도 32 KB 를 넘어 실패 → 64 KB 로(`CX_STACK_KB`), 통과.
- `import omnigibson` 은 `warp-lang==1.12.0` 이 있어야 된다(위 3 의 1 과 같음). HF 토큰은 `hf auth login`(저장소 밖).

## 4. 첫 확인 (`tools/setup/linux_first_check.sh N`)

radio 인스턴스 0, **0 행동 150 스텝 × N 판**(영상 저장), 판마다 검은 프레임 수(`black_frames.json`) + `tools/black_frame_check.py --max-ratio 0`, 이어서 기준 행동열 재생 비교.

결과(10-02, 앞 PC `ad17`·RTX 4090, `outputs/linux_first_20261002_143938/`, `outputs/linux_first_20261002_145840/`)
- 0 행동 5 판: 카메라 3 대 모두 검은 프레임 **0/151**, 판 정상 종료(0 행동이라 success 0).
- 행동열 재생 2 판(501 스텝): 검은 프레임 **0/501**, `black_frame_check.py --max-ratio 0` 통과.
- **판정: 앞 PC(`ad17`)에서는 검은 프레임 (B) 가 나지 않았다**(`jy-desktop` 에서 다시 확인). 검출기(`--black-guard`, `black_frame_check.py`)는 제출 판 증거로 계속 켠다.
- GUI 실행(창 띄움, 10-02 `outputs/linux_gui_20261002_151040/`): 501 스텝 정상, 검은 프레임 0.

## 5. 확인할 것

- [x] `nvidia-smi` — RTX 5070 Ti, 580.178.04 (앞 PC 는 RTX 4090, 같은 드라이버)
- [ ] `vulkaninfo --summary`
- [x] torch 2.7.0+cu128
- [x] CUDA 12.8 툴킷 (`/usr/local/cuda-12.8`)
- [ ] 네이티브 π0.5 엔진 `src/vla/pi05_native/build_linux.sh` 를 sm_120 으로 빌드
- [ ] π0.5 radio 체크포인트 → 정책으로 radio 한 판(공식 제한시간)
