# behavior-2026 — BEHAVIOR Challenge 2026 작업 폴더

제출 마감: 10/16 AoE (한국시간 10/17 토 20:59). 대회 요약·규칙은 [docs/](docs/README.md). 구현 계획·결정 기록은 [plan.md](plan.md).

## Windows (`C:\behavior-2026`)

```
BEHAVIOR-1K\          대회 프레임워크 (StanfordVL, 태그 v3.9.3-post1) — 원본, 고치지 않는다
  datasets\           시뮬레이터 데이터
    behavior-1k-assets\              장면·물체 에셋 3.9.0 (34.7 GB, 암호화 USD)
    omnigibson-robot-assets\         로봇 에셋 3.8.2
    2026-challenge-task-instances\   대회 과제 인스턴스
    omnigibson.key                   에셋 복호화 키
data\
  2026-challenge-demos\   데모 (LeRobot 형식) — 메타 + 0번 과제 turning_on_radio 만 (전체 3.27 TB)
src\                  우리 구현 (정책 래퍼, meridian 연동 등)
  configs\r1pro_openpi.yaml   평가기 로봇 설정 (공식 r1pro.yaml 에서 로봇 이름만 openpi 에 맞춤)
tools\                개발 도구 (run_pi05_server.sh = 정책 서버, run_eval_radio.ps1 = 평가기, -Gui 로 뷰어)
  setup\              설치·다운로드에 한 번 쓴 스크립트 (기록용, 다시 돌려도 이어받기)
refs\                 참고용 외부 저장소 (2025 상위 팀 코드 등)
outputs\              평가 결과 (롤아웃 JSON · 영상)
docs\                 문서 (목차는 docs\README.md, 원문 보관본은 docs\raw\)
logs\                 설치·다운로드·실행 로그
```

- conda 환경 `behavior`: Python 3.11, torch 2.7.0 cu128, Isaac Sim 5.1, OmniGibson(eval), BDDL, JoyLo.
  평가기: `conda activate behavior` → `python -m omnigibson.eval.eval ...`

## WSL (`Ubuntu-22.04`, 사용자 `juyoung`)

```
/opt/ros/humble          ROS 2 Humble desktop (meridian 용)
~/openpi                 π0.5 정책 서버 (wensi-ai/openpi, behavior 브랜치, uv 환경)
~/Isaac-GR00T            GR00T N1.7 (wensi-ai/Isaac-GR00T)
~/checkpoints/
  pi05_turning_on_radio        기본 제공 π0.5 체크포인트 (17 GB)
  groot_n17_turning_on_radio   기본 제공 GR00T 체크포인트 (6.5 GB)
  behavior_submission          2025 1위 제출 체크포인트 4개 (50.6 GB, 과제 0~49 한 모델)
  openpi_comet/pi05-b1kpt50-cs32   2025 2위 공개 가중치 (12.4 GB)  ← tools/setup/download_top_ckpts_wsl.sh
~/.cache/huggingface     토큰(juyoung02) · Cosmos-Reason2-2B (GR00T 백본, 게이트 동의 완료)
~/meridian_ws/src/       연구실 meridian 인식 모듈 (neoul-ro, 시뮬레이터에는 이것만 필요)
  meridian_msgs            메시지 정의
  meridian_frontend        Frontend: RGB-D → SAM → CLIP → 추적 → tracklet (seg·clip·geotracker 통합본, 정리 중)
  meridian_associator      DA 노드
  meridian_engine          DA 라이브러리 (graphcore 가 링크)
  meridian_graphcore       물체 그래프 저장소 (Spark-DSG)
```

meridian 의 sensor·slam 은 쓰지 않는다 — 시뮬레이터가 RGB-D 를 주고, 위치는 학습 데이터에선 정답 pose,
평가에선 바퀴 속도 적분(라이다·IMU 없음)으로 만든다. Windows 쪽 사본은 2026-09-29 지웠다(원격과 동일, 변경 없음 확인).

정책 서버는 WSL 에서, 평가기(시뮬레이터)는 Windows 에서 돌리고 localhost:8000 으로 잇는다.
