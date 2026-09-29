# 리눅스 듀얼 부팅 설치 — Ubuntu 22.04 + RTX 5070 Ti + Isaac Sim 5.1 + BEHAVIOR-1K v3.9.3-post1

작성: 2026-09-30 · 목적: 이 PC 의 Windows 에서 생기는 검은 프레임 (B)(RTX 가 카메라 영상 한 칸을 3 스텝 주기로 비움, [평가기_가속설계.md](평가기_가속설계.md) 5.2.2)를
피해, **렌더가 되는 리눅스에서 공식 평가기를 돌려** 제출용 결과를 뽑는다. 주최 측 성능 측정 장비도 "Ubuntu 22.04.5 LTS"([raw/site_challenge_evaluation.md](raw/site_challenge_evaluation.md)).
표기: (추정) = 설치 전이라 확인 못 한 것. 스크립트: `tools/setup/linux_setup.sh`, `linux_import_from_windows.sh`, `linux_first_check.sh`.

## 0. 먼저 막히는 것 — 디스크 공간 (사용자 결정 필요)

이 PC 디스크는 **Samsung 990 EVO Plus 1 TB 한 장**이고, Windows C: 가 거의 전부(998 GB)를 쓰며 **남은 공간 약 24 GB** 다(09-30 측정). 리눅스 파티션을 만들 자리가 없다.

리눅스에 필요한 공간(대략)

| 항목 | 크기 |
|---|---|
| Ubuntu 22.04 + 드라이버·도구 | 약 25 GB |
| conda `behavior` 환경(Isaac Sim 5.1 pip 휠 25 개 + torch 등) | 약 25 GB (추정) |
| BEHAVIOR-1K datasets(에셋 3.9.0·로봇 에셋·2026 인스턴스·키) | 34.7 + 1.2 GB |
| π0.5 체크포인트(radio) + 네이티브 가중치 | 17 + 6.3 GB |
| 결과(영상·기록, 1,000 판) + 여유 | 50 GB 이상 |
| **합계** | **최소 약 160 GB, 권장 250 GB** |

선택지
1. **Windows 쪽 공간을 비우고 C: 를 줄인다**(디스크 관리 > 볼륨 축소). 큰 것: WSL 디스크 `ext4.vhdx` **212 GB**(`C:\Users\user one\AppData\Local\wsl\{5e74…}`), Docker Desktop WSL 디스크, 다른 프로젝트 데이터.
   예: WSL 안의 안 쓰는 것을 지우고 `wsl --shutdown` 뒤 vhdx 압축(Optimize-VHD / diskpart compact) → C: 250 GB 축소. 축소가 막히면(움직일 수 없는 파일) 페이지 파일·시스템 복원을 잠시 끈다.
2. **외장 또는 두 번째 NVMe SSD** 에 설치(가장 안전 — Windows 파티션을 안 건드림). 메인보드 M.2 빈 슬롯이 있는지 확인.
3. 리눅스 USB 라이브 부팅으로 먼저 (B) 가 리눅스에서 안 나는지만 확인(설치 없이 1 판) — 공간·시간이 모자랄 때 판정만.

→ **어느 쪽으로 할지 사용자 결정**. 아래 절차는 파티션(또는 새 SSD)이 준비됐다고 본다.

Windows 쪽 준비(확인됨): 최대 절전 끔(`powercfg /a`), 빠른 시작 끔(`HiberbootEnabled` = 0) → 리눅스에서 NTFS 를 읽기 전용으로 붙일 수 있다. BitLocker 를 쓰면 복구 키를 미리 적어 둔다.
Secure Boot 상태는 권한 문제로 못 읽었다 — 켜져 있으면 드라이버 설치 때 MOK 등록이 필요하다(3절).

## 1. 설치 순서 한눈에

1. Ubuntu 22.04.5 LTS 설치(USB, "다른 것" 으로 빈 공간에 ext4 `/` + EFI 는 기존 것 공유). 부팅 순서는 GRUB.
2. `bash tools/setup/linux_setup.sh driver` → 재부팅(MOK 등록) → `nvidia-smi` 확인.
3. `bash tools/setup/linux_setup.sh base` → 새 셸 → `gh auth login`.
4. `bash tools/setup/linux_setup.sh repos` (우리 저장소 + BEHAVIOR-1K `bd049de` = v3.9.3-post1).
5. `sudo bash tools/setup/linux_import_from_windows.sh mount` → `… datasets` → `… data` (Windows 쪽 데이터 복사, 4 절).
6. `bash tools/setup/linux_setup.sh behavior` (공식 setup.sh, 데이터셋은 5 에서 옮겼으므로 `--dataset` 빼고).
7. `bash tools/setup/linux_setup.sh check` → `bash tools/setup/linux_first_check.sh 3` (5 절).
(1~4 는 이 저장소가 아직 리눅스에 없으므로, 처음엔 USB 나 Windows 파티션(`/mnt/win/behavior-2026/tools/setup/`)에서 스크립트를 부른다.)

## 2. NVIDIA 드라이버 (RTX 5070 Ti = Blackwell)

- Blackwell(RTX 50) 은 리눅스에서 **open 커널 모듈** 드라이버만 지원하고 R570 이상이 필요하다(추정 — NVIDIA 공지 기준). 스크립트는 `nvidia-driver-580-open`.
- 근거: 2025 상위팀 포크의 하드웨어 기록(리눅스) "The RTX 5090 runs the benchmark on driver 580.173.02. It gives a segmentation fault on driver 595.84." — **580 계열, 595 피하기**.
  같은 기록: "The RTX 5090 still needs a torch build with sm_120 kernels. Use torch 2.7.0+cu128." (공식 setup.sh 가 cu128 torch 를 깐다 — `check` 단계에서 확인.)
- Isaac Sim 5.1 문서의 리눅스 권장 드라이버 번호는 설치 전에 확인 못 함(추정: 580 계열). 설치 뒤 `nvidia-smi` 의 Driver Version 을 [설치기록.md](설치기록.md) 에 적는다.
- Secure Boot 켜짐이면 설치 중 정한 비밀번호로 재부팅 때 "Enroll MOK".

## 3. 공식 평가기 설치

- 공식 명령(평가 문서): `./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval` → conda 환경 `behavior`, Isaac Sim 5.1 pip, OmniGibson[eval], bddl3, joylo.
- 스크립트는 약관 동의 인자(`--accept-conda-tos --accept-nvidia-eula --accept-dataset-tos`)를 붙이고, 데이터셋은 Windows 에서 복사하므로 `--dataset` 을 뺀다.
  데이터셋을 새로 받고 싶으면 `--dataset` 을 붙이면 된다(약 35 GB 내려받기).
- 이어서 `pip install av "numpy<2"`(검은 프레임 검사 도구가 씀). 우리 도구(`tools/eval_instrumented.py`, `exp_run.ps1` 의 리눅스판은 아직 없음)는 파이썬이라 그대로 돈다.

## 4. Windows·WSL 에서 가져올 것

| 무엇 | Windows 위치 | 리눅스 위치 | 방법 |
|---|---|---|---|
| 우리 저장소 | GitHub `juyoung020/behavior-2026`(비공개) | `~/behavior-2026` | `gh repo clone` (repos 단계) |
| BEHAVIOR-1K | `C:\behavior-2026\BEHAVIOR-1K` (bd049de) | `~/behavior-2026/BEHAVIOR-1K` | GitHub 에서 같은 커밋 clone |
| datasets + 복호화 키 `omnigibson.key` | `C:\behavior-2026\BEHAVIOR-1K\datasets` | 같은 상대 경로 | `linux_import_from_windows.sh datasets` (NTFS 읽기 전용 → rsync) |
| π0.5 네이티브 가중치·데모 메타 | `C:\behavior-2026\data\pi05_native`, `data\2026-challenge-demos\meta` | `~/behavior-2026/data/` | `… data` |
| 재생 기준 판(nf_a) | `C:\behavior-2026\outputs\eval_turning_on_radio_20260929_195500_nf_a` | `~/behavior-2026/outputs/` | `… data` |
| π0.5 체크포인트·openpi | WSL `~/checkpoints`, `~/openpi` (ext4.vhdx 안) | `~/checkpoints` | `… wsl`(vhdx 를 qemu-nbd 로 읽기 전용) → `… ckpts`. openpi 는 새로 clone·`uv sync` 가 깔끔 |
| 키·토큰 | WSL `~/.config/behavior-2026/kau.env` 등 | 같은 경로 | 손으로 복사(저장소에 넣지 않는다) |

- Windows 파티션은 **항상 읽기 전용**으로 붙인다(스크립트가 `-o ro`). 복호화 키는 datasets 안에 있다(키 파일은 커밋 금지).
- git 신원: `juyoung020 <151780134+juyoung020@users.noreply.github.com>` (repos 단계가 설정).

## 5. 첫 확인

`bash tools/setup/linux_first_check.sh 3`
1. radio 인스턴스 0, **0 행동 150 스텝 × 3 판**(영상 저장), 판마다 검은 프레임 수(`black_frames.json`) + `tools/black_frame_check.py --max-ratio 0`.
   판정: 3 판 모두 0 이면 리눅스에서는 (B) 가 없다고 본다(Windows 에서는 판의 약 60% 에서 났다). 더 확실히 하려면 5 판.
2. Windows 기록과 같은 행동열(nf_a, 500 스텝) 재생 → `tools/trace_compare.py` 로 물리·판정·JSON 차이를 적는다(OS·드라이버가 달라 비트 동일이 아닐 수 있다 — 기록만).
3. 그 다음: π0.5 로 radio 한 판(공식 제한시간) → 제출 패키지 도구 흐름 확인. 리눅스용 실행기(`exp_run` 은 PowerShell)는 필요하면 bash 판을 만든다.

## 6. 확인할 것(설치 뒤)

- [ ] `nvidia-smi` 드라이버 번호, `vulkaninfo --summary` 에 RTX 5070 Ti 가 잡히는지
- [ ] `python -c "import torch;print(torch.__version__, torch.cuda.get_arch_list())"` 에 sm_120
- [ ] 첫 확인 결과(검은 프레임 0 / 재생 비교)를 [평가기_가속설계.md](평가기_가속설계.md) 5.2.2 와 [설치기록.md](설치기록.md) 에 적기
