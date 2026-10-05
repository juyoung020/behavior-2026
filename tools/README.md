# tools — 실행·측정·검증 스크립트

이 문서는 2026-10-03 에 Windows·WSL 전용 스크립트를 리눅스(jy-desktop, Ubuntu 22.04, RTX 5070 Ti sm_120, 드라이버 580,
conda `behavior` = OmniGibson 3.9.3 / Isaac Sim 5.1, CUDA 12.8) 판으로 옮긴 내용을 적는다.
옛 스크립트는 지우지 않고 `archive/<원래 경로>` 로 옮겼다([archive/README.md](../archive/README.md)).

> **10-06**: π0.5 관련 스크립트(`ft_*`, `run_pi05_*`, `run_verify_chunk.sh`, `serve_b1k_agent.py`, `check_agent_hook.py`, `setup/train_4090.sh`,
> `setup/download_top_ckpts.sh`, `exp/native_*.json`, `src/vla/*`, `src/sim/integ/run_eval_integ.sh`)는 지웠다. VLA = RecallVLA (robot-agent `training/vla`).
> 아래 "확인한 것 (2026-10-03)" 표는 그때 기록이다.

## 공통

- 경로는 모두 저장소 기준이다(스크립트가 제 위치에서 저장소 뿌리를 찾는다). `/mnt/c`, Windows 파이썬, `wsl.exe` 를 쓰지 않는다.
- 시뮬레이터를 띄우는 스크립트는 conda `behavior` 환경을 스스로 켠다(`CONDA_BASE` 기본 `conda info --base`, 없으면 `~/miniconda3`).
- 재생 서버(`tools/replay_policy_server.py`)의 파이썬은 `REPLAY_PY`(기본 `~/miniconda3/envs/behavior/bin/python`). Rust 재생 서버·비교기는
  `~/cargo-target/{replaysrv,tracecmp}/release/` (빌드: `src/sim/fasteval/tracecmp/build.sh`, replaysrv 는
  `CARGO_TARGET_DIR=~/cargo-target/replaysrv cargo build --release`).
- 옛 행렬·메타 JSON 안의 `C:/behavior-2026/...` 경로는 읽을 때 이 저장소로 바꾼다. 저장소의 행렬은 저장소 기준 상대 경로로 고쳤다.
- 끝낼 때 남의 프로세스를 이름으로 끄지 않는다(`pkill -f` 없음). 스크립트가 띄운 서버만 pid·프로세스 그룹으로 끈다.
- 모든 셸 스크립트는 `set -euo pipefail`(렌더 장면 뜨기만 옛 판처럼 `set -uo pipefail`), `--help` 와 대부분 `--dry-run`(명령만 찍기)이 있다.

## 옛 스크립트 → 리눅스판

| 옛 (archive/…) | 리눅스판 | 하는 일 |
|---|---|---|
| `tools/gpu_lock.ps1` | `tools/gpu_lock.sh` | 공용 GPU 잠금 + 선착순 대기열 (명령·라이브러리 둘 다) |
| `tools/run_eval_radio.ps1` | `tools/run_eval_radio.sh` | 공식 평가기 한 번(계측·검은 프레임 검사 선택) |
| `tools/run_replay_eval.ps1` | `tools/run_replay_eval.sh` | 기록한 행동열을 재생 서버로 먹이며 평가기 |
| `tools/exp_run.ps1` | `tools/exp_run.py` | 과제 x 인스턴스 x 설정 실험 실행기 · 표 · 비교 |
| `tools/subpack.ps1` | `archive/tools/subpack.py`(10-06 보관) | 제출 패키지 모으기·검사·README·zip |
| `src/sim/integ/wsl_stack.sh` | `src/sim/integ/simlink_stack.sh` | simlink 띄우기 + VRAM 기록 |
| `src/sim/integ/wsl_cleanup.sh` | `src/sim/integ/simlink_cleanup.sh` | 비정상 종료 때 simlink 쪽 정리(적어 둔 pid 만) |
| `src/sim/engine/capture/run_capture.ps1` | `src/sim/engine/capture/run_capture.sh` | 재생하며 물리 층 기록(RTX 렌더 켬, 영상 저장) |
| `src/sim/engine/capture/render_scene_wsl.py` | `src/sim/engine/capture/render_scene.py` | 렌더 없이 장면 덤프(이름만 바꿈) |
| `src/sim/engine/capture/render_scene_wsl.sh` | `src/sim/engine/capture/render_scene.sh` | 과제 하나의 렌더 장면(scene.rsc) 뜨기 |
| `src/sim/engine/tests/render/gpu_session.ps1` | `src/sim/engine/tests/render/gpu_session.sh` | 잠금 잡고 render GPU 시험 묶음 |
| `src/sim/engine/tests/render/capture/run_render_capture.ps1` | `…/capture/run_render_capture.sh` | 렌더 기준 자료 뜨기(RGB-D 래퍼) |
| `src/sim/engine/tests/render/capture/capture_with_lock.ps1` | `…/capture/capture_with_lock.sh` | 잠금 + VRAM 확인 뒤 위 스크립트 |
| `src/sim/engine/tests/render/capture/process_capture.ps1` | `…/capture/process_capture.sh` | 기준 자료 → rsc 변환·잡음 폭(GPU 안 씀) |
| `tools/setup/fetch_qwen35_gguf_wsl.sh` | `tools/setup/fetch_qwen35_gguf.sh` | Qwen3.5-9B GGUF 받기 + sha256 대조 |
| `tools/setup/setup_llamacpp_cuda_wsl.sh` | `tools/setup/setup_llamacpp_cuda.sh` | llama.cpp CUDA 12.8(sm_120) 빌드 |

옛 PowerShell 의 `-Name` 선택지는 `--name` 으로 바꿨다(예: `-MaxSteps` → `--max-steps`, `-ChunkSize` → `--chunk-size`). 동작은 같다.

## 스크립트별

### tools/gpu_lock.sh — 공용 GPU 잠금

`<저장소>/.gpu_lock` 디렉터리를 만드는 것이 잠금, `<저장소>/.gpu_queue/` 가 선착순 대기열. `owner.txt`·대기표 형식은 옛 ps1 과
엔진 라이브러리(`src/sim/engine/scripts/gpu_lock.sh`)와 같다(이 PC 에서 `/mnt/c/behavior-2026` 은 저장소를 가리키는 링크라 같은 디렉터리다).

```bash
tools/gpu_lock.sh status
tools/gpu_lock.sh acquire --owner fasteval --purpose '평가 3 판' --minutes 10 --vram-gb 6 [--max-wait-min 120]
tools/gpu_lock.sh release --owner fasteval          # owner 가 같을 때만 푼다
# 라이브러리
source tools/gpu_lock.sh
gpu_lock_enter <owner> <purpose> <분(최대 30)> <vram_gb> [최대 대기 분=120]   # 잡으면 0, 못 잡으면 1
gpu_lock_exit <owner>;  gpu_lock_cancel   # (기다리다 끊을 때 내 표 지우기)
```

오래된 잠금(end_epoch + 900 초 지남)·오래된 표(같은 기준, 또는 `host=linux|wsl` 인데 pid 없음)는 지운다. 기다리는 동안 `GPU_LOCK_POLL`(기본 45) 초마다 표를 새로 적는다.

### tools/run_eval_radio.sh — 공식 평가기 한 번

```bash
tools/run_eval_radio.sh --policy local --max-steps 300 --trace --black-guard warn      # 서버 없이 0 행동
tools/run_eval_radio.sh --port 8000 --instances 0,1 --wrapper RGBD --tag x            # 떠 있는 정책 서버로
```

선택지: `--task --instance --instances --max-steps(0=공식 제한시간) --wrapper Default|RGBD --port --tag --out-dir --chunk-size --gui
--timing --trace --deep --black-diag --black-guard warn|abort --kit-set(여러 번) --kit-arg(여러 번) --robot-config(경로|none)
--policy websocket|local --render-iters --vk-nvidia-only --dry-run`. 계측 선택지가 하나라도 있으면 `tools/eval_instrumented.py` 로 감싼다.
결과 `outputs/eval_<과제>_<시각>[_태그]/`, 로그 `logs/<같은 이름>.log`. `--vk-nvidia-only` 는 Windows 레지스트리 대신
`/usr/share/vulkan/icd.d/nvidia_icd.json` 을 `VK_DRIVER_FILES` 로 준다.

### tools/run_replay_eval.sh — 행동열 재생 평가

```bash
tools/run_replay_eval.sh --actions outputs/eval_turning_on_radio_20260929_195500_nf_a/actions.npz --trace --tag nf_a
```

재생 서버(포트 8010, `--once`)를 띄우고 `/healthz` 가 뜨면 `run_eval_radio.sh --port 8010` 을 부른다. `--perturb STEP:DIM:DELTA`(음성 대조),
`--conda-server`(옛 `-WindowsServer`: 서버를 conda behavior 파이썬으로), 나머지 계측 선택지는 그대로 넘긴다.

### tools/exp_run.py — 실험 실행기

```bash
python3 tools/exp_run.py run --matrix tools/exp/smoke_radio.json [--backend original|ported] [--reuse] [--dry-run]
python3 tools/exp_run.py table --dir outputs/exp_<이름>_<시각>
python3 tools/exp_run.py compare -A <실험 또는 판 폴더> -B <...> [--out compare.md]
```

행렬 형식·결과(`status.jsonl`, `results.csv`, `summary.md`, `logs/`)는 옛 판과 같다. 판마다 GPU 잠금(owner `exp_run`)을 잡고,
다른 시뮬레이터(파이썬 + omnigibson)가 없고 VRAM 이 `gpu_busy_mib` 아래일 때만 돈다. 재생 서버 `server`: `rust`(기본) |
`python`(옛 `wsl`, `REPLAY_PY`) | `conda`(옛 `windows`). 포팅 평가기는 `src/sim/engine/eval/ported_eval.py`(행렬 `ported.python` 으로 파이썬 지정 가능,
옛 `host` 키는 무시). 비교기는 Rust `tracecmp` 가 있으면 그것, 없으면 `tools/trace_compare.py`.

### tools/subpack.py — 제출 패키지

챌린지 제출 도구(R1 설정 전제) — 10-06 에 [`archive/tools/`](../archive/README.md) 로 옮김(`archive/tools/subpack.py`, `archive/tools/exp/submission_meta.example.json`).

### src/sim/integ/simlink_stack.sh (+ simlink_cleanup.sh)

```bash
bash src/sim/integ/simlink_stack.sh <출력 폴더> [scene 1|0] [llm kau|oracle|none] [simlink 인자 ...]
```

simlink(계획기 + scenemap) 하나를 띄운다: VRAM 기록 → simlink `--once`(`127.0.0.1:7801`, `SIMLINK_LISTEN`) → `stack_ready` → 평가기 쪽
접착부(`glue/simlink_policy.py` `IntegPolicy`)가 붙어 한 판 → `stack_done`. 평가기와 π0.5 를 함께 띄우던 종단 실행기 `run_eval_integ.sh` 는
10-06 에 지웠다(안쪽 정책이 없음). 연결만 재려면 `glue/link_bench.py`(0 행동 정책).
simlink 바이너리 `SIMLINK`(기본 `~/cargo-target/simlink/release/simlink`, `src/sim/integ/build_simlink.sh`). 키는 `~/.config/behavior-2026/kau.env` 에서 환경변수로만.

### 엔진 스크립트 (src/sim/engine)

엔진 코드는 안 바꿨다. 아래 스크립트만 옮겼다.

- `capture/run_capture.sh --actions A --tag T [--task --instance --max-steps 500 --dry-run]` — 옛 Windows 판처럼 RTX 렌더를 켜고 영상까지,
  결과 `src/sim/engine/dumps/<tag>/`. (렌더 없는 옛 WSL 판 `capture/run_capture_linux.sh` 는 `~/behavior-linux` 를 쓰는 그대로 남아 있다.)
- `capture/render_scene.sh <과제> [모드] [번호] [텍스처 상한]` — 렌더 없이 장면 덤프 → `convert_scene.py` → `~/engine-data/render_scenes/<과제>/`.
  잠금 owner `engine-lead`. `DRY_RUN=1` 로 명령만.
- `tests/render/gpu_session.sh [--script gpu_tests.sh] [--minutes 5] [--log logs/render_gpu_session.log]` — 잠금 owner `render-B`.
- `tests/render/capture/capture_with_lock.sh --actions A --tag T [--wrapper RGBD] [--minutes 20] [--max-other-mib 3500]` — 잠금 owner
  `engine-render`(옛 판의 mkdir 되풀이 대신 공용 대기열), GPU 사용량이 문턱 아래일 때 `run_render_capture.sh`.
- `tests/render/capture/run_render_capture.sh --actions A --tag T [--steps 0,100] [--noise-renders 2] [--dry-run]` → `dumps/render_<tag>/`.
- `tests/render/capture/process_capture.sh [--tag radio_rgbd] [--tex-max 1024] [--ref224 ...]` — export → rsc → `rsc_check`·`render_compare`.

### tools/setup — 받기·빌드

```bash
bash tools/setup/fetch_qwen35_gguf.sh [--dry-run] [REPO] [QUANT_FILE]   # ~/models/Qwen3.5-9B-GGUF (5.68 + 0.92 GB)
bash tools/setup/setup_llamacpp_cuda.sh [--dry-run]     # ~/llama.cpp/build/bin/{llama-server,llama-mtmd-cli,llama-bench}
bash tools/run_qwen_server.sh [CTX] [PORT]              # 위 둘이 있어야 함, 127.0.0.1:8081 (QWEN_HOST 로 바꿈)
```

계획기 LLM 은 AI 에이전트 수업(최영식 교수)의 KAU API(`https://agent.kau.ac.kr/v1`, Qwen3.5-9B)가 기본이고, 아래 로컬 llama.cpp 는 선택이다.

llama.cpp 는 `/usr/local/cuda-12.8` 의 nvcc 로 `CMAKE_CUDA_ARCHITECTURES=120` 빌드한다(13.2 안 씀). 12.8 이 없으면 NVIDIA ubuntu2204 저장소에서
컴파일러·런타임·cuBLAS 만 apt 로 받는다(sudo).

## 확인한 것 (2026-10-03)

| 스크립트 | 확인 | 결과 |
|---|---|---|
| 모든 새 셸 스크립트 | `bash -n`, shellcheck 0.11 | 경고 0 |
| `gpu_lock.sh` | 임시 디렉터리에서 잡기·대기·포기(`--max-wait-min 0`)·남의 잠금 안 풀기·오래된 잠금/죽은 pid 표 지우기·라이브러리 모드; 저장소 잠금으로 status; 엔진 `gpu_lock.sh` 가 잡은 잠금을 같은 디렉터리로 봄 | 모두 기대대로 |
| `exp_run.py` | `table` 을 Windows 실험 폴더 사본에(경로 `C:\` 바꿔 읽기); `compare` 파이썬·Rust 비교기, 음성 대조(nf_neg); `run --dry-run` (smoke_radio, radio_native_full) | 표가 Windows 판 `summary.md` 와 글자까지 같음(제목 폴더 이름만 다름), nf_a~nf_b 통과·nf_neg 실패(종료 1) |
| `subpack.py` | 실제 결과 폴더 두 개 → 임시 폴더 패키지·zip; 제한시간 위반 판 → 오류·zip 안 만듦; 없는 출처 | zip 항목·README·검사 보고 정상, 종료 코드 0/1/2 |
| `run_eval_radio.sh`, `run_replay_eval.sh`, `run_insurance.sh`, `run_eval_integ.sh`, 엔진 capture·render 스크립트 | `--dry-run` 으로 만든 명령 확인 | 옛 판과 같은 인자 |
| 재생 서버 | openpi venv 파이썬판·Rust replaysrv 를 띄워 `/healthz` | 둘 다 200 |
| `simlink_stack.sh` + `simlink_cleanup.sh` | `--llm none --scene 0` 으로 simlink 띄움 → `stack_ready`·`vram.txt`·`stack_pids` → 정리 | 127.0.0.1:7801 에서 듣고, 정리 뒤 simlink 없음 |
| `gpu_session.sh` | 잠금 잡고 작은 시험 스크립트 실행 → 풂 | 정상(실제 `gpu_tests.sh` 는 `~/engine-build/render` 바이너리가 아직 없어 안 돌림) |
| `download_top_ckpts.sh --dry-run` | 원격 목록과 `~/checkpoints` 비교 | 1위 50.58 GB·Comet pt50 11.34 GB 모두 받아 둠, 받을 것 0 |
| `fetch_qwen35_gguf.sh --dry-run` | 원격 크기·sha256 | Q4_K_M 5.68 GB(sha256 03b74727…)·mmproj 0.92 GB, 아직 안 받음(디스크 여유 부족해 안 받음) |
| `setup_llamacpp_cuda.sh` | 실제 빌드(CUDA 12.8, sm_120, `-j 24`) | 약 4 분, `~/llama.cpp` 커밋 bed0a85, `llama-server`·`llama-mtmd-cli`·`llama-bench` 생김, nvcc 12.8.93, cudart·cuBLAS 가 cuda-12.8 에 링크, `llama-bench --list-devices` 가 RTX 5070 Ti(compute 12.0) 를 봄. GGUF 가 아직 없어 추론 시험은 못 함(로그 `logs/setup_llamacpp_cuda_20261003_0557.log`) |

Isaac Sim 을 띄우는 실제 평가 판(`run_eval_radio.sh` 등)은 이번에 돌리지 않았다 — 같은 때 다른 GPU 작업(trtexec·모델 내보내기)이 돌고 있었다.
