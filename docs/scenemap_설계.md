# scenemap — 새 인지 스택 설계 (2D SLAM · 물체 지도 · 계획기 질의)

작성 2026-09-30. 상태: **설계(구현 전)**. 코드는 `src/scenemap/` 에 새로 둔다.

- 근거는 사용자 팀 저장소 `juyoung020/robot-programming-team` 의 [docs/plan.md], [docs/model_selection.md] 다(09-30 판).
  - 이 문서는 그 선택을 **BEHAVIOR 2026 대회 환경에 맞춘 것**이다. 다른 점만 1절 표에 적는다.
- 앞서 검증한 규칙은 2절에 모은다. 근거는 [통합_실시간.md](통합_실시간.md) 2절이다.
- 추정은 "(추정)"으로 적는다.

## 0. 한눈에

| 부품 | 팀 문서의 선택 | scenemap 에서 | 상태 |
|---|---|---|---|
| 지도·위치 | Cartographer 식 2D SLAM. 지도 위 위치만 추정하고, 물체 위치(xyz)는 로봇 위치 + 카메라 장착 위치 + depth | 같음. 라이다 대신 **깊이 → 가상 스캔**. 2D 점유 격자(로그 오즈) + 평면 3자유도 스캔 매칭 | 설계 |
| 물체 인식 | YOLO-seg nano, 80종 밖이면 YOLOE / YOLO-World | **YOLOE-seg**(열린 어휘). 프롬프트 = 과제 BDDL 물체 이름. C++ frontend 에이전트가 만든다. 출력 형식은 이 문서 5절이 정한다 | 형식 제안 |
| 같은 물체 판단(DA) | 직접. 같은 이름끼리 위치로 비교, 부족하면 색 분포 | 같음 | 설계 |
| 지도 갱신 | 직접. 바뀐 부분만(DovSG·Khronos 참고) | 같음 + 들고 있는 물체 처리(2절 규칙) | 설계 |
| 저장·보기 | Spark-DSG + 뷰어, 계획기에는 JSON | 같음. 살아 있는 Hydra 식 층 그래프(3.5) + Spark-DSG JSON 직접 쓰기, 뷰어 sgviz(층 쌓기) | 동작 중 |
| 계획 | Qwen3.5-9B API | 같음(`src/agent`, KAU API) | 동작 중 |
| 행동 | π0.5 | 같음(네이티브 C++/CUDA 엔진, 평가기 프로세스 안) | 동작 중 |

## 1. 대회에 맞춰 다른 점

| 항목 | 팀 문서(리모) | 대회(BEHAVIOR 2026, R1Pro) | 그래서 scenemap 은 |
|---|---|---|---|
| 거리 센서 | 2D 라이다 EAI X2L | **라이다 없음.** 머리 zed RGB-D 720², 손목 RealSense 480² | 머리 깊이에서 높이 띠를 잘라 **가상 레이저 스캔**을 만든다(3.1) |
| 계산기 | Jetson Nano 4 GB 에 전부 | Jetson 제약 없음. 대신 **평가기 GPU(이 PC 24 GB, 대회 서버 24 GB 추정)를 시뮬레이터·π0.5 와 나눠 씀** | GPU 는 신경망(YOLOE TensorRT)만 필수. SLAM·물체 지도는 CPU 로 충분한지 먼저 재고(목표 keyframe 당 수 ms), 필요한 곳만 손 CUDA |
| 실행 틀 | ROS 2 노드 | **ROS 없음.** C++/CUDA 라이브러리(C ABI) + Rust 조율 | simlink(Rust)가 같은 프로세스에서 부른다. 평가기 프로세스 안에 올릴 수도 있게 C ABI 로만 드러낸다 |
| 파이썬 | 학습·외부 도구만 | 같음 | 실행 경로에 파이썬 없음. 채점 도구만 파이썬 |
| 카메라 외부 자세 | 캘리브레이션 | 평가 규칙상 시뮬레이터 카메라 자세 금지 | **proprio 관절값 + R1Pro URDF 순기구학**(오차 0.003 mm) |
| 위치 초기값 | 저장한 지도 위 위치 추정 | 판마다 새 장면, 지도 없음. 전역 자세 금지 | 판 시작 = 원점. 매 판 지도를 새로 만든다(SLAM). 저장 지도 위 위치 추정은 안 쓴다 |
| 정답 | 없음 | 학습 데이터에 원본 HDF5(정답 자세·물체 궤적) | 채점 도구가 정답으로 잰다(6절). 실행 경로에는 안 쓴다 |

## 2. 검증한 규칙

| 규칙 | 근거 | 내용 |
|---|---|---|
| 영상 k = 장면 k-1 | 통합_실시간 2.7 | 영상·깊이 stamp = 장면 시각(k-1). 짝지을 proprio(robot2cam·base_qvel·팔 끝·그리퍼)는 k-1 값. 짝짓기는 stamp 로 |
| 카메라 자세 = 순기구학 | 통합_실시간 2.6 | 머리: base_link → torso_joint1..4 → zed_link → 고정 변환. 손목도 같은 방식 |
| 시간 규칙 = stamp | — | 끊김·만료·갱신 주기는 전부 시뮬 시각으로. 경계에서 시뮬이 멈추면 시간도 멈춘 것으로 본다 |
| 손에 든 관측 버리기 | — | 점의 절반 이상이 팔 끝(proprio) 0.4 m 안인 검출은 물체 지도에 넣지 않는다 |
| 잡기·놓기로 옮기기 | — | 그리퍼(손가락 합 < 0.09)가 닫힐 때 손 가까운 물체를 '든 물체'로, 열릴 때 손 자세 변화만큼 옮긴다 |
| 깊이 보정 오도메트리의 문턱·사전항 | — | base_qvel 적분 예측 + 사전항(σ_xy 2 cm + 10%, σ_yaw 0.5° + 15%), 크게 뛴 보정 버리기(8 cm / 3° / 4σ) |
| 자기 몸 빼기 | — | 베이스 반경 0.55 m, 팔 끝 0.35 m, 어깨–팔 끝 선분 0.2 m 안 점은 스캔에서 뺀다 |
| 작은 물체 | — | 열린 어휘 검출기에서도 작은 마스크 하한·신뢰도를 인자로 두고 잰다 |

## 3. 구성

```
평가기          ── TCP(요약 + keyframe 영상) ──▶ simlink(Rust)
                                                     │  (같은 프로세스, C ABI)
                                                     ▼
            ┌──────────────── libscenemap (C++/CUDA) ────────────────┐
  영상 k ─▶ │ YOLOE-seg(TensorRT, C++ frontend 에이전트) ─ 검출(5절) ─┐│
  proprio ─▶│ ① slam2d: 가상 스캔 → 스캔 매칭 → 점유 격자 → 자세 ────┼┼─▶ 자세(계획기·π0.5 접착부)
            │ ② objmap: 검출 + 깊이 + 자세 → xyz → DA → 갱신 ◀──┘│
            │ ③ query: 2D 지도 + 물체 표(JSON / C 구조체) ───────────┼─▶ 계획기(src/agent, 같은 프로세스)
            │    save: Spark-DSG 파일(판 끝·요청 때)                  │
            └─────────────────────────────────────────────────────────┘
```

- **언어·틀**: 핵심은 C++20 라이브러리 `libscenemap`(CUDA 는 필요한 곳만). 드러내는 것은 C ABI 헤더 하나(`scenemap.h`)다.
  - Rust 쪽은 `scenemap-sys`(FFI) + 얇은 안전 래퍼 크레이트다. simlink 가 부르고, 계획기(`src/agent`)는 같은 프로세스에서 질의한다.
  - 평가기 프로세스 안에 올려야 하면 같은 C ABI 를 그대로 쓴다.
- **스레드**: 호출자 스레드 하나(simlink 의 관측 스레드)에서 `push_frame` → `update` 순서로 돈다. 질의는 스냅숏(읽기 전용 사본, 원자 교체)이라 계획기 스레드가 막지 않는다.
- **GPU 메모리 목표**: YOLOE 엔진 + 버퍼 합 1 GB 안(추정, 재서 정함). SLAM·물체 지도는 CPU 메모리(수십 MB).

### 3.1 ① slam2d — 깊이 가상 스캔 + 2D 점유 격자 + 스캔 매칭

코드: `src/scenemap/`(C++20, 외부 의존 없음) — `scan`(깊이 → 스캔), `grid`(점유 격자), `slam2d`(예측·맞추기·문턱), `fk`(R1Pro 순기구학). 채점: `src/scenemap/eval/`, 재생 도구 `tools/slam2d_eval`.

- **스캔**(keyframe 마다, 머리 깊이 4 px 간격, 0.3~8 m, k-1 짝의 순기구학 카메라 자세)
  - 로봇 몸 빼기: 순기구학 팔 뼈대(팔 받침–관절 1..7–그리퍼–손끝) 캡슐 r 0.09~0.10 m, 몸통 캡슐 0.18 m, 베이스 반경 0.55 m, 팔 끝 0.35 m 구.
  - 붙은 것 빼기: 베이스 기준 3D 칸(5 cm, 로봇 1.3 m 안)이 로봇이 10 cm·6° 움직일 때마다 두 번 연속 같은 자리(±1 칸)에 있으면 '들고 있는 것'으로 보고 뺀다.
  - 레이저 한 줄: 높이 띠 [0.10, 1.80] m 에서 방위 0.5° 칸마다 가장 가까운 점. 띠 아래(바닥) 점은 그 방위가 거기까지 비었다는 증거.
  - 맞추기 점: 수직면(|n_z| < 0.7) 점을 [0.10, 3.0] m 에서 모아 2.5 cm 칸마다 하나(평균 위치 + 깊이 법선의 수평 성분, 카메라 쪽).
    레이저 한 줄만으로 맞추면 머리가 탁자를 볼 때 탁자 가장자리만 남아 ep0 에서 53 cm 까지 틀렸다(아래 표 `B 레이저`).
- **지도**: 한 판 전체 하나의 2D 점유 격자(5 cm, 로그 오즈 +0.85/−0.4, 한계 ±4, 필요할 때 넓어짐).
  - 레이저 한 줄은 Bresenham 광선(빈칸 −, 끝 칸 +), 맞추기 점은 맞음만(광선 없음 — 가려진 벽 뒤를 지우지 않게). 한 스캔에서 칸마다 한 번, 맞음 우선.
  - 칸마다 맞은 점 평균과 법선 합을 쌓는다. 칸이 빈칸 쪽(L ≤ 0)으로 가면 비운다(옮겨진 물체 흔적이 남지 않게).
  - 움직임 거르기(Cartographer MotionFilter): 지난번 넣은 뒤 5 cm·2° 움직였거나 50 keyframe 지났을 때만 넣는다.
- **예측·문턱**
  - 예측 = 직전 보정 자세 ⊕ base_qvel 적분. 제자리 잡음(|v| < 1 cm/s, |w| < 0.01 rad/s)은 적분하지 않는다(서 있는 동안 yaw 가 1~2° 도는 것을 막음).
  - 제자리 규칙: 지난 keyframe 뒤 계속 제자리면 맞추지 않는다(팔로 물건을 옮기는 동안 스캔이 움직이는 물체에 끌려가 ep3000 에서 1 m 틀림).
  - 사전항 σ_xy = 2 cm + 이동의 10%, σ_yaw = 0.5° + 회전의 15%. 거절: 인라이어 < 50, 또는 8 cm·6°·4σ 넘게 뜀.
    yaw 6°: base_qvel wz 가 가끔 실제보다 0.17 rad/s 크다(ep3000 83 s, ep11600 시작) — 3° 로는 맞는 보정을 버리고 5° 가 굳었다.
- **맞추기 후보** — 같은 데이터(8판 + 긴 판 2개)로 비교

  | 후보 | 방식 |
  |---|---|
  | A. Cartographer 식 | 점유 확률 격자(본 적 없는 칸 0.1, 0.1~0.9)를 Catmull-Rom 쌍삼차 보간, 맞추기 점의 (1−P)² 최소화(LM) + 사전항 |
  | B. 2D point-to-line | 가장 가까운 칸 평균(거리 문턱 0.2×3 → 0.1×3 → 0.05×2 m)에 칸 법선으로 점-선 거리, Huber 3 cm, 가우스–뉴턴 8회 + 사전항 |
  | C. 지금 3D 점-평면(기준선) | 파이썬 시제품(5 cm voxel, 0.3~4 m, 붙은 것·제자리 규칙 없음) |

- **루프 닫기**: 넣지 않는다(09-30 판단). 긴 판 끝 오차가 B 로 4.8 cm(43 m)·5.3 cm(83 m)라 10 cm 기준 안이다.
  - 판 하나 전체를 한 격자로 쓰니 다시 온 곳에서는 옛 지도에 붙어 스스로 되돌아온다(ep11600 300 s: 29 → 10 cm).
  - 중간 최대(ep11600 25 cm)는 처음 가는 넓은 곳을 지날 때 쌓인 것이다. 물체 위치가 이 때문에 틀리는지 objmap 채점에서 다시 본다.
- **출력**: `base_pose(stamp)`(map), 카메라 광학 자세(순기구학 합성), 점유 격자(해상도·원점·폭·높이·i8 값 −1/0~100).

#### 3.1.1 측정(09-30, 학습 데모 재생, 영상 k ↔ proprio k-1, 정답 = 원본 HDF5)

- 원래 최대 = 판 시작 베이스 기준 위치 오차의 최대. 맞춘 뒤 최대 = 궤적 전체를 평면 강체 하나로 맞춘 뒤 최대(첫 스캔 전 적분 yaw 오차처럼 지도 전체를 돌리기만 하는 오차를 뺀 것).
- 시간 = keyframe 하나(스캔 + 맞추기 + 넣기), CPU 한 코어, 입력 160×120(640×480 의 4 px 간격).

  | 판 | 이동 | C 원래/맞춘 뒤 최대 cm | A 원래/맞춘 뒤 | **B 원래/맞춘 뒤** | B 레이저 한 줄 | 적분만 |
  |---|---|---|---|---|---|---|
  | 0 | 1.3 m | 1.5 / 1.6 | 2.2 / 2.2 | **1.7 / 1.9** | 14.0 | 2.4 |
  | 200 | 12.9 m | 2.1 / 1.8 | 2.2 / 2.0 | **2.6 / 1.6** | 9.8 | 179.7 |
  | 201 | 5.9 m | 4.4 / 3.7 | 3.9 / 2.8 | **3.1 / 1.8** | 43.4 | 22.3 |
  | 202 | 7.2 m | 1.5 / 1.1 | 4.9 / 3.3 | **1.2 / 1.4** | 8.3 | 212.8 |
  | 203 | 10.1 m | 2.1 / 0.9 | 2.7 / 2.7 | **1.4 / 0.9** | 5.2 | 42.3 |
  | 204 | 13.4 m | 2.0 / 1.4 | 4.7 / 4.6 | **2.8 / 2.2** | 6.1 | 282.0 |
  | 205 | 10.1 m | 1.8 / 1.3 | 3.4 / 2.3 | **3.9 / 3.0** | 9.2 | 208.9 |
  | 206 | 12.9 m | 2.9 / 1.5 | 3.8 / 2.7 | **2.3 / 1.5** | 5.4 | 125.9 |
  | 3000 | 43.5 m | 160.2 / 108.4 | 28.3 / 18.7 | **7.8 / 7.2** | 43.2 | 551.7 |
  | 11600 | 83.4 m | 172.1 / 112.1 | 32.4 / 31.1 | **25.0 / 16.2** | 142.4 | 1615.2 |
  | keyframe p50 / p99 | | (numpy 5~10 ms) | 0.6 / 5~9 ms | **0.5~1.0 / 2~3 ms** | 0.5 / 1 ms | |

- **B 를 고른다.** 긴 판 둘에서 C 보다 20배 안팎 좋고(C 는 제자리에서 팔로 나무를 옮길 때·들고 있는 물체에 끌려 1 m 넘게 틀렸다), A 보다 모든 판에서 같거나 좋고 가볍다.
- **통과 조건(C 대비 +0.5 cm 안)을 못 넘은 판**: 원래 최대로 200(+0.5), 204(+0.8), 205(+2.1). 맞춘 뒤로는 204(+0.8), 205(+1.7).
  - 205 는 처음 가는 곳을 지나며 1 m 에 약 1 cm 씩 쌓이는 오차다(40~50 s, 1.4 → 3.2 cm). 칸 크기(3 cm)·맞추기 칸(5 cm)을 바꿔도 ±1 cm 안에서 오르내려 더 줄이지 못했다.
  - 대신 201·202·203·206 과 긴 판 둘은 C 보다 좋다. 3D 를 2D 로 줄인 값으로 본다.
- CUDA 는 필요 없다: keyframe p99 3 ms(CPU)라 병목이 아니다. 평가기 원 해상도(720×720, 4 px 간격 180×180)는 점 수가 약 1.7배라 p99 5 ms 안팎으로 본다(추정).
- 버린 것: Cartographer 가중(이동 10·회전 40, `--carto-prior`)은 ep0 43 cm·ep3000 250 cm 로 나빴다. 깊이 4 m 까지만 쓰면 ep11600 에서 스캔이 비는 구간이 생겨 나빴다(8 m 로 바꿈).

### 3.2 ② objmap — 검출 → xyz → 같은 물체 → 갱신

- **위치 xyz**(팀 문서 그대로. 검출기 마스크의 무게중심을 깊이로 xyz 로 바꿔 기록, 보기는 2D 지도)
  - 검출 마스크를 1 px 깎고, 깊이 유효 화소를 k-1 짝 카메라 자세로 map 에 올린다.
  - 위치 = 각 축 **중앙값**이다. 크기 = 10~90 백분위 폭, 점 수, 보인 각도도 적는다.
  - 색 분포: 마스크 안 HSV 8×4 막대그래프(정규화)다. 같은 이름이 여럿일 때만 쓴다.
- **들어가기 전 거르기**
  - 손에 든 관측(2절)을 버린다.
  - 구조물 이름(바닥·벽·천장)은 물체 지도에 넣지 않는다. 대신 2D 지도가 맡는다.
  - 점 20개 미만은 버린다.
- **같은 물체 판단(DA)**: 같은 이름끼리만 비교한다.
  - 거리 문턱 = max(0.3 m, 0.5 × 큰 쪽 크기).
  - 한 프레임의 같은 이름 검출들과 기존 물체는 헝가리안 배정(거리 + 색 거리 가중)으로 1:1 로 맞춘다.
  - 안 맞은 검출은 '후보'가 되고, 2번 이상(다른 keyframe) 보이면 물체로 확정한다(최소 관측 2).
- **갱신(바뀐 부분만)**
  - 맞은 물체: 위치·크기를 관측 수 가중 평균으로 고친다. 크게 달라지면(> 문턱) '옮겨짐 의심' 이다.
  - **부재 확인**(DovSG 식 국소 갱신)
    - keyframe 마다 기존 물체 중 카메라 시야 안·가림 없음(깊이가 물체보다 10 cm 이상 멀다)인데 검출에 안 맞은 것은 부재 증거 +1 이다.
    - 3번 연속이면 '사라짐' 이다. 같은 이름이 다른 곳에 새로 확정되면 그쪽으로 '옮겨짐' 으로 잇는다(Khronos 식 이력: 처음 자리·옮긴 시각·자리 목록을 남김).
  - **들고 있는 물체**(2절)
    - 그리퍼가 닫히면 손 0.25 m 안 물체를 '든 물체'로 한다. 드는 동안 위치 = 손 + 잡은 순간의 상대 자세다.
    - 열리면 그 자리에 둔다. 다음 관측이 맞으면 그 값으로 고친다.
- **방(room)**: 1판에서는 2D 격자에서 문(YOLOE "door" 검출 + 좁은 통로) 기준으로 영역을 나누는 것을 나중 단계로 둔다(필요하나 특권 정보 금지).
  - 그 전에는 물체마다 `room: null` 이다.
- **저장**: Spark-DSG `DynamicSceneGraph` 로 저장한다(C++, ROS 없음). 층은 OBJECTS(물체 노드: 이름·위치·크기·이력), PLACES 는 나중(방), 2D 지도는 파일 옆에 PGM + YAML 로 둔다. 판 끝·요청 때 저장한다. 뷰어는 `spark-dsg visualize`(오프라인)다.

#### 3.2.1 진행(09-30, 멈춘 지점)

코드: `src/scenemap/include/scenemap/objmap.hpp`, `src/objmap.cpp`(규칙 요약은 헤더 첫머리). 입력은 4.2 약속 `sm_detections` 그대로.
**C ABI(`capi.cpp`)에는 아직 안 붙였다** — `sm_snap_objects/find/near` 는 0 개를 돌려준다.

- 위에 적은 설계에서 바뀐 것(재 보고 고침)
  - 같은 물체: 중심 거리만 쓰면 부분만 보이는 큰 가구(소파 3.3 m·계단·조리대)가 시점마다 따로 등록됐다(ep200 중복 23). map 축 상자 사이 틈 < 10 cm 도 같은 것으로 보고, 한 변 > 0.5 m 인 것은 상자를 합집합으로 키운다 → 중복 2.
  - 헝가리안 대신 틈·거리 순 탐욕 1:1(한 keyframe 에 같은 이름이 몇 개뿐).
  - 사라짐: 작은 물체(한 변 ≤ 0.5 m)만, 팔 끝 0.5 m 안은 판단 안 함(잡으러 다가갈 때 손 거르기와 겹쳐 캔이 '사라짐'이 됐다), 받침에 붙은 것은 안 봄.
  - 옮겨짐 잇기: 같은 이름이 '사라짐'인 것만 새 자리로 잇는다(다른 캔과 헷갈림 방지).
  - 들기: 팔 끝 → 물체 거리를 베이스 축으로 저장(로봇이 돌면 같이 돔). 놓을 때 놓은 점 아래 xy 가 겹치는 물체 중 윗면이 가장 높은 것에 붙인다 — 들고 다니는 쓰레기통에 넣은 캔 셋이 통을 따라 끝 자리로 감.
- 채점(`eval/export_gtdet.py` → `tools/objmap_eval` → `eval/score_objmap.py`): 정답 상자로 깊이를 라벨링한 '완벽한 검출'(320×240, 9 프레임마다), 자세는 slam2d.
  짝 = 같은 범주, 정답 상자까지 수평 거리 0.3 m 안(높이 차 0.5 m 까지 봐줌 — 통 안에 떨어뜨린 높이는 깊이로 못 봄).

  | 판 | 본 물체 | 찾음 | 중복 | 헛것 | 합쳐짐 | 옮겨짐 찾음 |
  |---|---|---|---|---|---|---|
  | 0 (라디오) | 12 | 12 (100%) | 0 | 0 | 0 | 1/1 |
  | 200 (쓰레기) | 43 | 40 (93%) | 3 | 0 | 0 | 4/4 |
  | 201 | 23 | 22 (96%) | 0 | 0 | 0 | 4/4 |
  | 204 | 43 | 38 (88%) | 2 | 0 | 0 | 4/4 |
  | 3000 (나무 들이기) | 56 | 30 (54%) — 정답 자세로도 31 | 11 | 0 | 2 | 0/0 |

  - 못 찾는 것: 천장 조명(2.3 m, 얇음), 전자레인지·오븐 일부(합쳐짐·가림). ep3000 은 정답 자세로도 54% 라 slam 이 아니라 objmap 쪽 문제다(아직 안 봄). 나무 토막은 과제 인스턴스 상자에 없어 '옮겨짐 0/0'(정답 쪽 한계, 따로 봐야 함).

#### 3.2.2 실시간 물체 기억: 종류·상자·사라짐·best view·저장(10-03)

코드: `src/scene_graph/scenemap`(objmap·capi·bestview·png·dsg_save), `src/scene_graph/runtime`(libsgrt, `src/crop.cu`). 시험: `scenemap/tests/test_objmem.cpp`, `runtime/tests/test_crop.cpp`.

- **이름 종류**(팀 벤치마크 dynamic-object-mapping-benchmark 의 정답 정의를 따름: 배경 구조물은 instance 0, 채점 안 함)
  - 구조물 — 물체 노드가 안 되고 격자만: wall, floor, ceiling, door, doorway, door frame, window, pillar, column, partition, staircase, stairs, stair, railing, baseboard.
  - 고정(가구·가전·붙박이) — 노드지만 `movable=false`, 사라짐 판정 없음, 상자는 한도 있는 합집합: table, desk, counter, sofa, shelf, cabinet, bed, refrigerator, oven, sink, lamp, plant, picture frame, rug, curtain, radiator, light switch, electric outlet …(`capi.cpp` `kStaticNames`).
  - 나머지는 옮길 수 있는 물체. 비교: 정규화(".n.NN" 버림, '_'→' ', 소문자) 뒤 머리 명사("glass door" → door, "floor lamp" → lamp). 표는 `sm_set_kind_names(ctx, SM_KIND_STRUCTURE|SM_KIND_STATIC, names, n)`(NULL = 기본)로 바꾼다. 확정(서로 다른 keyframe 2 번)된 것만 노드가 된다.
- **상자**: 마스크 안 카메라 깊이가 중앙값 ± max(3·1.4826·MAD, 0.10 m) 밖인 점은 버린다(뒤 벽이 비침). 큰 가구 합집합은 keyframe 마다 면마다 0.25 m 까지, 한 변 4 m 까지. 검출 하나에서 훑는 화소는 6000 개 안팎으로 간격을 넓힌다(720² 에서 keyframe 4.6 → 1.5 ms).
- **사라짐**: 작은 물체만, 3 번 연속 놓치고 첫 놓침에서 시뮬 2 s 넘게 지나야 한다. 고정 종류·큰 것은 사라지지 않는다.
- **best view**: 검출이 물체에 붙으면(`sm_last_assoc`, objmap `lastAssoc()`) 품질 = 유효 마스크 넓이(깊이 화소) × 점수. 지금 것 이상이면(같으면 최근) 바꾼다. 옮겨짐·놓기 사건이 나면 품질을 0 으로 내려 다음 관측이 바로 바꾼다.
  - 자르기: 상자 + 변마다 10 %, 긴 변 최대 256 px(넓이 평균). RGB 는 `sm_push_image_ex` 의 자르기 함수가 한다 — sgrt 는 장치 메모리에서 커널 한 번(32 상자씩) + 자른 것만 고정 메모리로 내려받는다(온 영상 복사 없음). 깊이는 호스트 깊이에서 같은 상자·같은 크기, uint16 mm.
  - 질의: `sm_snap_view(snap, id, &v)`.
- **모양(점 구름, 10-03)**: 물체에 붙은 관측마다 위치·크기에 쓴 마스크 안 깊이 점(MAD 띠 안)에서 팔 끝 0.10 m·베이스 수평 0.30 m 안 점을 빼고, 관측 안에서 복셀(0.02 m)마다 하나를 물체 구름에 넣는다(같은 칸은 새것으로 바꿈). 물체마다 4000 점 한도 — 넘으면 오래 안 고쳐진 점부터 버려 3600 으로(`sm_set_cloud_params`).
  - 색: 남긴 화소만 모은다 — sgrt 는 장치에서 화소 좌표를 올려 커널로 모으고 그 색만 내림(3000 점 0.016 ms). 영상이 없으면 회색 128.
  - 상태: 들기·받침 따라가기 = 구름 평행 이동(원점만 옮김, 회전 없음), 사라짐 = 마지막 구름 유지, 사라졌다 다른 자리에서 다시 찾음(옮겨짐 잇기) = 비우고 새로 쌓음(옛 자세에서의 모양을 옮겨 붙이면 회전·부분 관측이 섞여서), `sm_reset` = 비움.
  - 점은 원점 기준 float 로 두고 바뀔 때만 새 배열(쓸 때 복사) — 스냅숏은 포인터만 복사, `sm_snap_points` 로 읽음.
  - best view 마스크: 검출 마스크를 자른 상자·크기로(255 안).
- **저장**(`sm_save_dsg` / `sm_save_dsg_ex`): PNG(`objects/O<id>_rgb.png` 8 비트 RGB, `O<id>_depth.png` 16 비트 회색 mm, 자체 쓰기 + zlib)는 모습이 바뀐 것·파일이 없는 것만 쓴다. 새 판·새 디렉터리면 옛 `O*_*.png` 를 지운다. 순서 PNG → scene.json → view.json.
  - 파일: `objects/O<id>_rgb.png`(8 비트 RGB) · `_depth.png`(16 비트 회색 mm) · `_mask.png`(8 비트 회색, 255 = 마스크 안) · `_points.ply`(binary_little_endian, `float x,y,z`(map m) + `uchar red,green,blue`, 15 바이트/점, 주석 없음). 구름은 version 이 바뀐 것만 다시 쓴다.
  - 노드 `metadata.points = {path: "objects/O<id>_points.ply", n, voxel, stamp(구름이 마지막으로 바뀐 시뮬 s)}`, `metadata.rgbd.mask = "objects/O<id>_mask.png"`. view.json `objects[]` 에 `points{path,n,voxel,stamp}`, `rgbd.mask`.
  - scene.json 노드 메타데이터(기존 그대로 + 추가): `state, n_obs, score, first_pos, structural, handled, movable`, 모습이 있으면
    `rgbd = {rgb, depth(상대 경로), stamp, box_px[4](자른 영역, 원 영상 화소, 여유 포함), det_box_px[4], mask_area, depth_m(마스크 깊이 중앙값), score, cam_T[12](map ← 카메라 광학, 행 우선 3×4)}`. view.json `objects[]` 에 `movable`, `rgbd{rgb, depth}`.
- **벤치마크 형식**: `tools/map_timeline <ep.bin> <det.bin> <out>/<seq>` — C ABI 로만 재생해 `map_timeline.csv`(frame,obj_id,x,y,z,label,moving; 지도가 바뀐 프레임만, 사라짐 뺌, moving = 들고 있음)를 쓴다. 벤치마크 `read_timeline` 으로 읽힘을 합성 판으로 확인(실제 판 det.bin 은 원본 HDF5 가 없어 아직 못 만듦). 같은 폴더에 `map_points.npz`(키 `<obj_id>@<frame>`, N×3 float32, 그 프레임 중심 기준 map 좌표 — 모양이 바뀐 프레임만: 점 수 10 % 넘게 바뀜 또는 300 프레임)를 numpy 없이 직접 씀(zip 저장 방식 + NPY 1.0). 벤치마크 `MethodShapes.load` 로 경고 없이 읽힘(합성 판).
- **잰 시간(구름 붙인 뒤)**: keyframe 720² 검출 6(매번 best view 자르기 + 구름) 2.0 ms, 검출 4 평균 1.3 ms. 저장(물체 3): 다 바뀜(PNG 9·PLY 3) 0.74 ms(PLY 0.05 ms), 안 바뀜 0.30 ms, 구름만 0.40 ms.
- **잰 시간**(jy-desktop, RTX 5070 Ti, Release): keyframe 720² — slam2d 만 0.57 ms, + objmap·best view(검출 6, 매번 자르기, 호스트 RGBA) 1.48 ms. 장치 자르기 6 상자(185 kB) 커널 + 내려받기 0.017 ms(온 영상 2 MB 내려받기 0.12 ms). 저장(물체 3, 격자 포함) PNG 6 장 0.66 ms, PNG 없음 0.32 ms.

### 3.3 ③ query — 계획기 질의

- 같은 프로세스 Rust API 가 기본이다. 디버그·도구용 TCP JSON 줄은 선택이다.

| 질의 | 돌려주는 것 |
|---|---|
| `objects(filter)` | 물체 표: id, 이름, 위치(map·로봇 기준), 크기, 관측 수, 마지막 본 시각, 상태(보임·사라짐·옮겨짐·들고 있음), 처음 자리, 방 |
| `object(id)` | 한 물체 + 이력(자리 목록) |
| `near(position, r)` / `find(name)` | 가까운 순 / 이름으로 |
| `map(region?)` | 2D 격자(해상도·원점·크기·값), 로봇 자세. 계획기 LLM 에는 줄인 표현(방·통로·알려진 영역 경계, 물체 좌표 표)을 JSON 으로 |
| `reachable(from, to)` | 격자 위 최단 경로 길이(A*, 로봇 반경 부풀림) — 이동 판단용 |
| `pose()` | 지금 로봇 자세 |

- 이름 매칭: 계획기는 BDDL 이름(`radio_receiver.n.01`)으로 묻고, scenemap 은 YOLOE 프롬프트 id 로 저장한다. 표는 판 시작 때 같이 만든다(5절).

### 3.4 ④ rooms — 방 나누기(10-03)

코드: `scenemap/include/scenemap/rooms.hpp`, `src/rooms.cpp`. 시험: `tests/test_rooms.cpp`. 오프라인: `tools/rooms_pgm <memory 디렉터리> [출력] [되풀이 수]`(map.pgm·map.yaml·view.json 을 읽어 방·문·물체 배정을 찍고 rooms.pgm·rooms_color.ppm 을 씀, 되풀이 수를 주면 나누기 + id 잇기 시간 중앙값).

Hydra(RSS 2022) room finder 를 2D 격자로 옮겼다(Hydra 와 다른 점·견준 것은 아래 3.4.1). place 대신 격자 빈칸을 그대로 쓰고(칸마다 방 id 가 있어야 rooms.pgm·`sm_snap_room_at`·물체 배정이 됨), 문턱은 전역 하나가 아니라 성분마다 수명으로 고른다.

1. **분류**: 값 0..49 빈칸(광선이 한 번 지나간 칸 = 40 %), ≥ 65 점유, 그 사이·모름 = 막힘. 안쪽 모름 구멍 < 0.25 m²(빈칸이 둘레의 반 이상) → 빈칸, 빈칸에 둘러싸인 점유 점 ≤ 0.01 m²(의자 다리) → 빈칸(나누기에만). 빈칸 상자 + 2 칸만 잘라 계산.
2. **거리 변환**: 빈칸 → 막힌 칸(격자 밖 포함) 정확한 유클리드 거리(세로 두 번 쓸기 + 가로 Felzenszwalb 봉투). 여유 c = 거리 − res/2(벽 면까지). 위쪽 쓸기가 끝난 줄은 바로 가로 봉투를 돌리고, 봉투는 줄의 빈칸 토막마다 따로(양 끝 막힌 칸이 닻이라 토막 밖 포물선은 못 이김).
3. **한 번 쓸기 거름 + 붙이기**(ToMATo 꼴 지속성 나누기, 거름과 넘치기를 한 union-find 쓸기로): 문턱 층 = 0.025 m 통(0.60 → 0.30 m 가 13 층, 그 아래도 같은 폭).
   - c > 0.60 칸(맨 위 층, 방 안쪽 대부분)은 한 층이라 그 안의 만남은 언제나 수명 0 → 정렬 없이 줄 토막 연결 성분으로 바로 묶는다(캐시 친화).
   - 나머지 빈칸은 통으로 한 번 세기 정렬해 여유 큰 것부터 넣는다. 칸은 이미 들어온 이웃 중 여유가 가장 큰 칸의 씨앗을 물려받는다(가파른 오르막 = 넘치기). 태어남 번호 하나가 성분(union-find)이자 씨앗(union-find).
   - 두 성분이 만날 때 둘 다 0.5 m² 이상이고 어린 쪽 수명(태어난 문턱 − 만난 문턱) ≥ 0.10 m 면 둘 다 씨앗으로 얼리고(Hydra barcode/lifetime), 아니면 짧게 산 쪽 씨앗을 **만난 자리 건너편 씨앗**에 합친다(전에는 그 칸을 버리고 넘치기에 맡김).
   - 0.30 m 아래(늦은 단계)에서는 새 씨앗이 없다: 남은 홑성분 중 0.5 m² 이상은 씨앗, 작은 것은 약한 씨앗 — 만나는 대로 건너편에 붙고, 끝까지 씨앗 없는 덩이는 2 m² 이상이면 방 하나, 아니면 방 없음.
   - 결과적으로 폭 ≈ 2·(0.60 − 0.10) = 1.0 m 보다 좁은 통로로만 이어진 곳이 다른 방이 된다(보통 문 0.8–0.9 m). 0.6 m 보다 좁은 틈은 0.30 m 위에서 이어지지도 않는다.
4. **모음**: 방마다 넓이·무게중심·상자·최대 여유와 이음매(맞닿은 칸 쌍, 위치 합·최대 여유)를 같은 방 줄 토막 단위 한 번 쓸기로 모은다. 합친 뒤에는 모음끼리 더하고(다시 안 쓸음), 번호가 바뀌었을 때만 칸 id 를 고쳐 쓴다.
5. **합치기**: 이음매(맞닿은 칸 쌍)가 1.6 m 이상이면 문이 아님 → 합침. 2 m² 보다 작은 방은 이음매가 가장 긴 이웃에(이웃 없으면 버림).
6. **문(방–방 변)**: 남은 이음매(≥ 2 쌍)마다 가운데 위치, 폭 = 2·이음매 최대 여유 + res.
7. **id 유지**: 이전 나눔과 세계 칸 좌표로 겹침을 세어(격자가 넓어지거나 원점이 옮겨져도) 큰 겹침부터 욕심쟁이 짝(겹침 ≥ 0.3 × 작은 쪽 넓이). 짝 없는 방만 새 id. 나뉘면 큰 쪽이 옛 id, 합쳐지면 겹침 큰 옛 id.
8. **다시 나누기**: `sm_snapshot` 이 시뮬 3 s 마다 빈칸 분류가 0.25 m² 이상 바뀌었는지 보고 바뀌었을 때만 다시 나눈다(scenemap 잠금 밖, 방 추적기 자기 잠금, 한 번에 하나). 물체 배정·이름은 스냅숏마다 새로(수 µs).
9. **물체 배정**: 바닥 자리(xy 상자 반폭 max(크기/2, 0.1) + 0.15 m)의 방 칸 다수결(가구는 바닥 자리가 점유 칸이라 둘레 빈칸이 표를 줌), 없으면 1 m 안 가장 가까운 방 칸.
10. **이름(규칙)**: 머리 명사 점수 — kitchen(refrigerator/fridge/oven/stove/cooktop/dishwasher 3, microwave/toaster 2, sink 1), bathroom(toilet/bathtub/shower 3, sink 1), bedroom(bed 3, nightstand 2, wardrobe/dresser 1), living room(sofa/couch 3, tv/television/coffee table 2, armchair/fireplace 1), office(desk/monitor/office chair 2, computer/keyboard/printer 1). 확률 = 점수 / (합 + 1.5), "unknown" = 1.5 / (합 + 1.5). 가장 큰 점수 ≥ 2 면 그 종류(같은 종류 둘째는 "kitchen 2"), 아니면 "room <id>". 근거(물체 id·이름·종류·무게)를 남긴다. **외부 이름 고리**: `sm_set_room_name(ctx, room_id, name, conf)` — LLM·BDDL 방 이름이 규칙 이름을 덮음(id 가 이어지니 다시 나눠도 유지, `sm_reset` 에서 지움).

C ABI: `sm_get/set_room_params`, `sm_update_rooms(ctx, force)`, `sm_set_room_name`, `sm_snap_rooms`(id·이름·종류·확신도·무게중심·상자·넓이·물체 id), `sm_snap_room_doors`, `sm_snap_room_grid`(칸 방 id), `sm_snap_room_at(xy)`, `sm_snap_object_room(obj_id)`.

저장(방이 있을 때만, 기존 키·파일은 그대로):
- scene.json ROOMS 층: 노드 `NodeSymbol('R', id)` RoomNodeAttributes — name, position = 무게중심(z 0), bounding_box = 방 칸 xy 상자(z 0), semantic_class_probabilities, metadata `{area_m2, name_confidence, evidence[{object,name,type,weight}], type, external_name, max_clear_m, grid_value}`. 방→물체 변(층 사이), 방–방 변(weight = 문 폭, metadata `{relation:"door", pos[2], width}`).
- `rooms.pgm`: map.pgm 과 같은 크기·방향(위가 +y, map.yaml 그대로 씀), 8 비트, 0 = 방 없음, k = view.json `rooms[]` 의 `value`(id 순 자리 + 1, 255 에서 멈춤).
- view.json: `rooms_grid:"rooms.pgm"`, `rooms:[{id, value, name, type, conf, centroid[2], area, bbox[x0,y0,x1,y1], color[r,g,b](id 마다 고정), objects[ids]}]`, `room_doors:[{a, b, pos[2], width}]`, `objects[].room`(0 = 없음).

잰 것(jy-desktop Ryzen 9 9950X, Release -O2, 시뮬 돌리는 중이라 부하 4–10): 합성 두 방 + 0.9 m 문 → 방 2·문 1(폭 0.95), ㄱ자 방(팔 2 m) → 1, 복도 1.4 m + 방 셋 → 4·문 3(복도–방만), 잡음(모름 5 %·점유 점 0.5 %·40 % 칸 20 %·모름 덩이 0.3²) → 2·1, 크기 다른 방(큰 방 둘 + 폭 1.3 m 욕실, 문 0.7·0.9 m) → 3·2, 자람 3 단계 + 원점 −2 m 이동에서 id 그대로. 실제 판 `outputs/mem_seg_20261003_042852/memory`(301×285, 빈칸 14 m², 대부분 모름): 방 2 — 소파·스탠드 둘이 있는 10.4 m² 방(living room, 확신도 0.67)과 문(폭 0.58, 틈새 너머) 너머 보인 5.5 m² 띠(복도로 보임, 이름 없음). `mem_yolo26s_20261003_044921`(같은 판, 빈칸 14.9 m², 물체 22 개)도 방 2·문 1.

시간(중앙값, 같은 때 이전 판과 번갈아 잼; 집·실제 판은 id 잇기 포함, 작은 합성 판은 나누기만, 한 코어 고정):

| 경우 | 이전 | 지금 |
|---|---|---|
| 600×600 집(9 방·문 12·가구), `test_rooms` | 8.2–8.4 ms | 4.8–4.9 ms |
| 600×600 전부 빈칸 | 7.3 ms | 3.1 ms |
| 두 방 / ㄱ자 / 복도+방 셋 / 잡음 | 0.31 / 0.26 / 0.72 / 0.34 ms | 0.18 / 0.17 / 0.38 / 0.21 ms |
| 크기 다른 방 / 1.4 m 틈 두 방 | 0.63 / 0.31 ms | 0.37 / 0.18 ms |
| 실제 판 mem_seg / mem_yolo26s(`rooms_pgm … 200`) | 0.64 / 0.67 ms | 0.52 / 0.54 ms |

600×600 집 단계별(지금, 한 코어 고정): 분류 0.2 · 정리 0.4 · 거리 변환 1.7 · 정렬 0.2 · 맨 위 층 연결 0.2 · 거름 쓸기 0.8 · 칸 id 0.2 · 모음 0.4 · 감싸기·id 잇기 0.3 ms. 이전 판과 칸 단위로 99.8–100 % 같은 방(다른 칸은 문 경계 몇 칸 — 넘치기 대신 가파른 오르막이라). 실제 판 두 곳도 99.8 %.

남은 것: 2D 라 가구가 만든 좁은 틈(< 1 m)도 문으로 볼 수 있음 · 넓게 트인 거실–부엌은 한 방 · 모름 경계(탐험 끝)도 좁으면 문처럼 갈림(지도가 자라면 다시 나눠 고쳐짐) · LLM 이름은 고리만.

#### 3.4.1 Hydra 방 나누기와 견줌(10-03, `refs/code/Hydra` `src/rooms/*`, `src/backend/update_rooms_functor.cpp`, 논문 III-B)

Hydra 가 하는 것:
- **입력**: places 층(GVD 노드, 노드마다 가장 가까운 장애물 거리, 변 무게 = 변을 따라 지나는 복셀 거리의 최소 = 병목). 3D 빈 공간이라 가구 잡동사니에 덜 흔들림.
- **거름**(`getGraphFiltration`, include_nodes = false): 노드를 다 홑집합으로 넣고 변을 무게 큰 순(힙)으로 union-find, 변마다 크기 ≥ `min_component_size`(10–15 place) 성분 수를 적음 → 거리–성분 수 표. `BarcodeTracker` 가 성분 수명(barcode)도 적지만 LONGEST_LIFETIME 방식에서만 씀.
- **문턱 고르기**: 창 [min_dilation, max_dilation](기본 0.1–0.7 m, 데이터셋 설정 0.5–1.2 m) 안에서 하나만 고른다. 방식 REPEATED(같은 수가 가장 오래 이어진 구간) · LONGEST_LIFETIME · **PLATEAU**(설정들이 씀, ratio 0.15–0.25: 길이 ≥ ratio × 가장 긴 구간인 구간 중 성분 수가 가장 많은 것, 그 구간의 가장 낮은 거리 δ*) · PLATEAU_THRESHOLD. 논문은 "10 개 문턱의 중앙값 성분 수를 갖는 가장 큰 δ".
- **씨앗**: 거리 > δ*, 변 무게 > δ* 인 연결 성분 중 크기 ≥ `min_component_size`.
- **붙이기**: NEIGHBORS(설정 기본) = 씨앗 경계에서 변 무게 큰 순 우선순위 큐 넘치기(우리 넘치기와 같은 생각). 논문의 MODULARITY(씨앗 고정, 붙지 않은 노드만 욕심쟁이 modularity 이득, 최대 5 번, γ 1)도 있음.
- **방**: place ≥ `min_room_size` 인 무리. id 는 무리의 가장 오래된 place 시각 순으로 R0, R1… 새로 매김(부를 때마다 ROOMS 층을 다 지우고 다시 씀 — id 가 이어진다는 보장 없음). 위치 = place 평균(빈 공간 밖이면 가장 가까운 place 구 가장자리로). 방–방 변 = 다른 방 place 끼리 변이 있으면(문 자리·폭 없음).
- **다시 하기**: 증분 없음. 백엔드 갱신마다 places 층을 복제해 처음부터(논문: "느린 고수준 과정"). 값은 place 수(수백–수천)에 비례 — 복셀·칸 수가 아님.
- **평가**: 정답 방과 place 겹침으로 정밀도·재현율(`eval/src/room_metrics.cpp`).

우리가 따온 것: 장애물 거리 거름, 큰 거리부터 union-find 한 번 쓸기, 성분 크기 문턱, barcode 수명, 씨앗에서 병목 큰 순으로 넘치기(NEIGHBORS), 방–방 변 = 맞닿음.

다른 점:
- **칸 vs place**: 칸마다 방 id 를 내야 해서(rooms.pgm·`sm_snap_room_at`·물체 바닥 자리 다수결·문 이음매) 격자에서 바로 한다.
- **문턱**: 전역 δ* 하나 대신 성분마다 수명(지속성 나누기). 잰 것: 크기 다른 방(큰 방 둘 + 폭 1.3 m 욕실) 에서 PLATEAU(ratio 0.25, 같은 격자·같은 창 0.30–0.60)는 δ* = 0.325 m 를 골라 큰 방 둘이 이어져 **방 2**, 우리는 **3**. 나머지 합성 경우(두 방·ㄱ자·복도+방 셋·잡음·집·1.4 m 틈)는 같음(δ* = 0.425 또는 0.30). 큰 방 사이 문(반폭 0.425)과 작은 방 자체(가장 넓은 곳 0.65)의 크기가 겹치면 한 문턱으로는 못 가른다.
- **문 모양**: 이음매에서 문 위치·폭을 내고, 긴 이음매(≥ 1.6 m)·작은 방(< 2 m²)을 합친다(Hydra 에는 없음).
- **id**: 이전 나눔과 칸 겹침으로 이어 줌(Hydra 는 매번 새로 매김). 외부 이름이 다시 나눠도 붙어 있음.
- **다시 하기**: 시뮬 3 s 마다, 빈칸 분류가 0.25 m² 이상 바뀌었을 때만.

Hydra 가 더 나은·싼 것과 우리 판단(재서 정함):
- **성긴 그래프(place 수에 비례)**: 600×600 단계별로 재 보니 어떤 방법이든 칸마다 해야 하는 일(분류·정리·거리 변환·칸 id 쓰기·모음)이 4.8 ms 중 약 3.3 ms, 그래프 몫(정렬·맨 위 층 연결·거름 쓸기)은 약 1.2 ms. 뼈대(GVD) place 그래프를 만들고 모든 칸을 다시 place 에 붙이는 값이 이 1.2 ms 보다 싸지 않아 **안 함**. 대신 Hydra 처럼 "필요한 것만 그래프로" 를 칸에서 흉내: c > dil_max(방 안쪽 대부분)는 한 층이라 정렬·수명 계산 없이 줄 토막 연결 성분으로 끝냄.
- **증분 거리 변환·바뀐 곳만**: 다시 나누기가 3 s 에 한 번 ≤ 5 ms(코어의 0.2 %) — 복잡도·틀릴 위험에 비해 얻는 게 없어 **안 함**(바뀌었는지 보는 판정은 이미 있음).
- **한 번 쓸기 거름**: 이전 판도 이미 한 번 정렬 + 내림차순 union-find 였다. 이번에 거름과 넘치기·씨앗 없는 덩이 찾기·씨앗 칸 표시를 한 쓸기로 합침(ToMATo).
- **PLATEAU 문턱·MODULARITY**: 위 측정대로 문턱은 우리 쪽이 낫고, 2D 칸 격자에서 modularity 는 넘치기보다 비싸고 문 자리에서 경계를 세울 근거가 없어 안 씀.
- 그 밖에 빨라진 것: 줄마다 빈칸 토막별 봉투·바로 가로 봉투, `std::ceil` 대신 정수 올림(기본 x86-64 에선 함수 부름), 맨 위 통은 세지 않음(같은 통 잇단 ++ 의 저장–읽기 지연), 덧댄 분류 격자에 바로 분류, 줄 토막 모음, 스레드마다 일감 버퍼 재사용(600×600 이면 약 8 MB 를 쥐고 있음, 부를 때마다 쪽 잘못 없음), 이상한 설정 막기(간격 ≥ 1 mm, 층 0 아래 통 늘 하나).

### 3.5 ⑤ 살아 있는 장면 그래프 — Hydra 식 층(10-03)

코드: `scenemap/include/scenemap/sgraph.hpp`, `src/sgraph.cpp`(갱신), `src/capi.cpp`(keyframe 뒤 갱신·C ABI), `src/dsg_save.cpp`(scene.json 직접 쓰기). 시험: `tests/test_posemap.cpp` testGraph, 뷰어 `viewer/sglayers.py`·`test_sglayers.py`.

물체를 저장할 때만 노드로 만들던 것을, 판 내내 살아 있는 그래프로 바꿨다. keyframe 마다 바뀐 곳만 고치고, 스냅숏은 그때의 바뀌지 않는 사본(GraphView)을 포인터로 나눠 쓴다.

| 층(Spark-DSG 번호) | 노드 | 속성 | 층 안 변 | 갱신 |
|---|---|---|---|---|
| 2 OBJECTS | 'O'<물체 id>, 확정 물체 | 위치·상자·상태·movable·이름 | on(바닥이 아래 물체 윗면 ±0.1 m·xy 겹침) / in(중심이 더 큰 물체 상자 안) / near(0.6 m), 쌍마다 하나 | 검출이 있는 keyframe, 바뀐 것만(1 mm·상태·이름) |
| 2 AGENTS(partition 'a') | 'a'<k>, 로봇 keyframe 자세 | 위치·yaw·시각 | 앞뒤 | 0.5 m·30° 움직이거나 10 s 마다 |
| 3 PLACES | 'p'<k>, 빈 공간 뼈대 | 위치(z 0)·여유(distance)·frontier | 직선 시야가 빈칸인 이웃(2 m 안, 가까운 6 개), 무게 = 변을 따라 가장 작은 여유(병목) | 격자 보이는 값이 바뀐 상자 + 2 m 창만, 0.5 s 마다 |
| 4 ROOMS | 'R'<방 id>(3.4) | 이름·무게중심·상자·넓이 | 문(무게 = 폭, 자리) | 방 나눔이 바뀔 때 |
| 5 BUILDINGS | 'B'0 | 방 넓이 가중 중심 | — | 방과 함께 |

- 층 사이 변(부모 → 자식): 건물 → 방, 방 → place(방 칸), 방 → 물체(3.4 배정), place → 물체(가장 가까운 place, 3 m 안), place → agent(2 m 안).
- **PLACES 만들기**(Hydra 는 3D ESDF 의 GVD, 우리는 2D 격자):
  1. 창 = 지난 계산 뒤 보이는 값이 바뀐 칸 상자(격자의 두 번째 dirty 소비자 — 탐색 쪽 `sm_take_dirty` 와 따로) + 2 m, 그 안 빈칸 상자로 줄임.
  2. 거리 변환(Felzenszwalb, 창 + 1 m 여백): 막힘 = 점유(≥ 50)·모름·격자 밖. 여유 = 거리 − res/2, 상한 1 m.
  3. 능선 후보: 빈칸, 여유 ≥ 0.20 m, 네 방향(가로·세로·대각 둘) 중 하나에서 양 옆보다 작지 않음(GVD 흉내 — 넓은 방 안은 상한이 평평해 고른 격자처럼 깔림).
  4. 성글게: 여유 큰 순으로, 간격 clamp(1.5·여유, 0.5, 1.5) m 안에 다른 place(창 밖 포함)가 없을 때만. 창 안 옛 place 는 지우고, 0.3 m 안 옛 id 는 다시 씀(id 가 덜 흔들림).
  5. frontier: 여유 + 0.15 m 원 위 16 점에 모름(또는 격자 밖)이 있으면 — 가장 가까운 막힘이 모름 쪽. 탐색 쪽이 바로 씀.
  6. 변: 창 + 2 m 안 place 의 place 변만 다시.
- 자세 보정 고리(SLAM 모드): 루프 닫기가 없어 격자가 휘지 않으므로 그래프는 map 좌표 그대로다. 보정이 생기면 agent 노드는 keyframe 자세를 가지고 있어 다시 붙일 수 있고, place 는 격자에서 다시 만든다(창을 전부로) — 지금은 부르지 않음.
- **C ABI**(추가만): `sm_snap_graph_nodes(snap, SM_GL_OBJECTS|AGENTS|PLACES|ROOMS|BUILDINGS|ALL, &nodes)`, `sm_snap_graph_edges`, `sm_snap_graph_node(id)`, `sm_snap_graph_neighbors(id, edge_idx, cap)`(CSR), `sm_snap_place_path(from, to, min_clear, ids, cap, &len)`(place 그래프 다익스트라, 변 여유 ≥ min_clear 만), `sm_set_object_meta(ctx, id, json_members)`(SigLIP 2 emb·이름 같은 것을 scene.json 물체 metadata 에 덧붙임). sgrt: `sgrt_map_snapshot(s)` = 마지막 `sgrt_map` 스냅숏(그래프·물체·방을 같은 순간 그대로).
- **저장(scene.json)**: Spark-DSG 라이브러리로 그래프를 저장 때마다 새로 만들고 nlohmann 으로 쓰던 것을, 같은 Spark-DSG 1.1.3 JSON 형식을 직접 문자열로 쓰는 것으로 바꿨다(`to_chars`, place·agent·방·건물 노드 조각은 노드 ver 로 캐시). 한 쌍에 변 하나(Spark-DSG 규칙). 기본 `spark_dsg` 파이썬(`DynamicSceneGraph.load`, 뷰어 venv)과 C++(test_objmem) 둘 다 읽는다. 라이브러리 판은 `SM_DSG_SAVE=spark` 로 비교용으로 남김. Spark-DSG 자체는 고치지 않았다(포맷을 직접 써서 라이브러리 쪽 비용이 없어짐 — 벤더링 필요 없음). view.json 에 `graph{nodes(agent·place·건물), edges[[a, b, rel, w]]}` 를 덧붙여 뷰어가 spark_dsg 없이 층을 그린다.
- 주기 저장은 sgrt 저장 스레드에서(스텝을 안 막음, 앞 저장이 덜 끝났으면 그 주기는 건너뜀, `SGRT_SAVE_SYNC=1` 이면 예전처럼).

#### 3.5.1 Hydra 와 견줌

| | Hydra(RSS 2022, `refs/code/Hydra`) | 우리(2D) |
|---|---|---|
| 지도 | TSDF/ESDF 3D 복셀, 메시 | 2D 점유 격자(5 cm, 깊이 가상 스캔) |
| places | ESDF 의 GVD(복셀 둘 이상의 가장 가까운 장애물), 희소화 후 노드·변(무게 = 변 위 최소 거리) | 2D 거리 변환 능선 + 여유 비례 간격 희소화, 변 = 직선 시야, 무게 = 병목 여유(같은 생각) |
| 갱신 | 활성 창(로봇 둘레)만 앞단에서, 뒷단이 루프 닫기 때 변형(deformation graph) | 바뀐 칸 상자 + 2 m 창만(0.5 s), 루프 닫기 없음 — 정답 자세 모드에선 변형 필요 없음 |
| objects | 메시 의미 분할 → 물체 노드 | 열린/닫힌 어휘 검출 + 깊이 → objmap(3.2) |
| agents | 자세 그래프 노드 | keyframe 자세 노드(0.5 m·30°·10 s) |
| rooms | places 거름 + 씨앗 넘치기(3.4.1) | 같은 생각을 격자 칸에서(3.4) |
| frontier | places 의 active/anti frontier(3D) | place 둘레 원 위 모름 |
| 층 사이 | place → room, object → place | 같음 + 방 → 물체(바닥 자리 다수결), 물체 on/in/near |
| 질의 | 오프라인·ROS | 같은 프로세스 C ABI(스냅숏), place 길 찾기 |

### 3.6 자세 원천·격자 넣기 정책·단계별 시간(10-03)

- **자세 원천**(`sm_set_pose_mode`, sgrt `SGRT_POSE=slam|odom|gt`)
  - `slam`(기본, 실제 로봇·대회 제출): base_qvel 적분 + 스캔 맞추기(3.1). map = 판 시작 베이스 프레임.
  - `odom`: 적분만(비교용).
  - `gt`: 시뮬 정답 베이스 자세(`sm_push_pose`, 접착부가 `robot.get_position_orientation()` 을 스텝마다 넣음 — 한 번 수십 µs). map = 시뮬 world. 맞추기 없이 그 자세로 넣는다. **대회 규칙상 제출에는 못 씀**(진단·시각화·정답 비교용).
  - 카메라 외부 자세는 어느 모드든 proprio 순기구학(베이스 ← 카메라). 확인: 기록한 판 455 keyframe 에서 정답 베이스 ∘ 순기구학 cam_rel 과 시뮬 정답 머리 카메라 자세 차 = 위치 평균 0.15 mm·최대 0.24 mm, 회전 최대 0.074°(같은 스텝 짝).
  - 다른 모드에서도 외부 자세를 넣으면 떠밀림 진단(`sm_get_pose_diag`: 첫 keyframe 에서 두 프레임을 맞추고 그 뒤 keyframe 마다 차).
  - 사용자가 본 "지도·로봇 자세가 어긋남"의 원인: (1) slam 모드 map 은 판 시작 프레임이라 world(정답·GT 물체 자리)와 회전·평행 이동만큼 다름 — gt 모드에선 world 그대로. (2) 접착부가 영상 stamp 를 그 스텝 시각으로 넣어 영상(장면 k-1)을 proprio k 와 짝지었다 — sgrt 가 영상 stamp 를 직전 스텝 시각으로 넣게 고침(`SGRT_IMAGE_LAG=1` 기본, docs/통합_실시간.md 2.7).
- **SLAM 떠밀림**(bringing_water public_test 0, comet π0.5 3000 스텝 기록 재생, `sm_bench`): 로봇이 거의 제자리(약 30 cm·몸통·머리 움직임)
  | 모드 | 최대 위치 / yaw | rms 위치 / yaw |
  |---|---|---|
  | slam(영상 늦춤 1, 정책 1) | 3.7 cm / 0.95° | 2.3 cm / 0.52° |
  | slam(영상 늦춤 0) | 4.4 cm / 0.90° | 2.3 cm / 0.40° |
  | slam(옛 움직임 거르기 정책 0) | 4.9 cm / 1.30° | 3.0 cm / 0.71° |
  | odom(적분만) | 15.7 cm / 4.20° | 9.4 cm / 2.53° |
  베이스를 움직이는 판(move_robot 대본, 3.6.1)은 아래에.
- **넣기 정책**(`sm_set_map_update`, `SGRT_MAP_POLICY`): 1(기본, 사건 기반) — 움직였거나, 가상 스캔 방위 칸 서명(720 칸, 칸 거리 5 cm 단위)이 지난번 넣은 것과 2 칸 넘게 다르거나, 지난 넣기가 아직 로그 오즈를 바꾸고 있으면(한계 전) 매 keyframe 넣는다(광선 빈칸 지우기 포함). 아무것도 안 바뀌면 건너뜀, 50 keyframe 마다 한 번은 넣음. 0 = 옛 판(5 cm·2° 또는 50 keyframe ≈ 10 s). 시험(test_posemap): 서 있을 때 상자가 생기면 6 keyframe(1.2 s) 뒤 점유, 없어지면 11 keyframe 뒤 빈칸, 안 바뀐 20 keyframe 동안 넣기 0 번 — 옛 판은 30 keyframe 안에 못 봄.
- **바뀐 영역**: 격자 보이는 값(i8)이 실제로 바뀐 칸만 감싼다(전에는 넣은 스캔 전체 상자). 소비자 둘(탐색 `sm_take_dirty`, 장면 그래프).
- **단계별 시간**(`sm_get_timing` / `sgrt_get_stage_timing`): 단계마다 µs 막대그래프(2^(1/4) 칸, 할당 없음) — push_proprio, integrate, image_total, pair_pose, fk, scan, attach, match, insert, objmap, view_prep, gather, crop, cloud_add, snapshot, snap_grid, rooms, save, graph_obj, graph_places, graph_publish(+ sgrt det, step, map, record). 접착부는 끝에 표와 파이썬 쪽(정답 자세 읽기·준비·호출) 평균을 찍는다.
- **기록·재생**: `SGRT_RECORD=<파일>` 이면 sgrt 가 받은 입력(proprio·외부 자세·keyframe 깊이·RGB·검출)을 그대로 쓰고, `tools/sm_bench`(C ABI, 모드·늦춤·정책·저장 주기 바꿔 재생, 떠밀림·단계 표), `tools/stage_bench`(내부 API — 옛 판 소스로도 빌드되어 같은 기록으로 전후 비교)가 다시 재생한다.

#### 3.6.1 단계별 µs 전후(같은 기록 500 keyframe, 720² 깊이·검출 평균 2.1, jy-desktop Ryzen 9 9950X 한 코어 고정, Release -O2)

`stage_bench` — 전 = 624b54b(이 작업 앞), 후 = 지금. 넣기는 전 판이 서 있어 40 번만 넣었고(움직임 거르기), 후 판은 매 keyframe 넣는다.

| 단계 | 전 평균 / p50 / p99 µs | 후 평균 / p50 / p99 µs | 바꾼 것 |
|---|---|---|---|
| 순기구학 + 몸 캡슐 | 2.4 / 2.4 / 3.8 | 2.6 / 2.4 / 4.1 | — |
| 깊이 → 가상 스캔(+붙은 것) | 334 / 297 / 595 | 188 / 154 / 440 | 화소 방위 칸·수평 배율 캐시(머리 회전 2e-4 안이면 다시 씀, 아니면 빠른 atan2), 거리² 비교, 캡슐별 상자, 작업 버퍼 재사용·평평한 해시(맞추기 칸) |
| 스캔 맞추기(움직일 때 314 번) | 99 / 87 / 673 | 98 / 84 / 703 | — |
| 격자 넣기 | 65 / 59 / 193(40 번) | 24 / 23 / 58(1000 번) | 뜨거운 8 B·차가운 20 B 칸 배치, int16 로그 오즈, 보이는 값 LUT 캐시, 점 변환 한 번, 광선 칸 번호 더하기만 |
| objmap + 구름 | 243 / 235 / 845 | 170 / 163 / 600 | 열마다 마스크 칸 미리(안쪽 고리 나눗셈 없음), 10/50/90 백분위 한 번에 + 2048 표본, 거리², VoxelIndex 세대 비우기, 작업 버퍼 재사용 |
| 격자 사본(스냅숏) | 53 / 49 / 81(매번 exp) | 23 / 17 / 47(바뀔 때만 복사) | 보이는 값 캐시 복사 |
| **keyframe 합** | **613 / 578 / 1416** | **416 / 388 / 1242** | |

C ABI 전체(`sm_bench`, 같은 기록, 스냅숏 6 스텝마다, 저장 1 s 마다, 호스트 RGB 자르기): push_proprio 0.08 µs·integrate 0.07 µs/스텝, 스냅숏 평균 52 µs(방 다시 나눌 때 p99 0.66 ms), graph_obj 7 µs, graph_publish 10 µs, graph_places 470 µs(0.5 s 마다 — 거리 변환 170·후보 70·고르기 60·변 40 µs), 저장 평균 1.6 ms·p99 14 ms(PNG·PLY 가 대부분 — sgrt 에선 저장 스레드라 스텝 밖). 스텝당 평균(모든 호출) 176 µs, 그중 keyframe 이 6 스텝에 한 번. sgrt 실제 판에선 best view 자르기가 장치 커널(약 17 µs)이다.

- **Jetson Nano 추정**(Cortex-A57 1.43 GHz, 이 PC 한 코어보다 약 5–7 배 느리다고 봄 — 재 보지 않음): keyframe 2–3 ms, 스텝 평균 0.4–0.6 ms(5 Hz keyframe), places 2.5–3.5 ms(0.5 s 마다), 저장은 별도 스레드. 스캔·objmap 이 가장 크다 — 남은 손: 스캔 화소 간격(지금 가로 160 점)을 Nano 에선 넓히기, objmap 표본 수(6000)·백분위 표본(2048) 줄이기, NEON.

#### 3.6.2 시뮬 확인(10-03, bringing_water public_test 0 = instance 301, 3000 스텝, headless, YOLO26s-seg)

- 판 둘: (A) comet π0.5(로봇이 거의 제자리, `outputs/mem_pose_slam_20261003_065114`, slam 모드로 돌리며 기록), (B) move_robot 대본(제자리 360°, 1 m·0.8 m 앞뒤, 90°·180° 돌기, 그 뒤 서 있음 — `outputs/mem_pose_gt_move2_20261003_073442`, gt 모드). 둘 다 `SGRT_RECORD` 로 기록해 `sm_bench` 로 모드를 바꿔 다시 돌림(재생 = 실시간 판과 칸까지 같음 확인).
- 채점 `eval/score_map_gt.py`: 정답 바닥 지도(`src/sim/explore/gt_trav.py`, world) — 우리 점유 칸이 정답 막힘(벽·가구) 10 cm 안인 비율, 우리 빈칸이 정답 바닥인 비율. 물체 = 중심에서 가장 가까운 정답 물체 상자까지(어휘가 달라 범주 없이).

  | 판·모드 | 점유 10 cm 안 | 빈칸 정밀 | 물체 10 cm 안(중앙값) | 떠밀림 최대 / rms |
  |---|---|---|---|---|
  | A slam(판 시작 프레임 그대로 = 뷰어에 보이던 것) | 36 % | 68 % | — | 3.7 cm·0.94° / 2.3 cm·0.52° |
  | A slam(정답 시작 자세로 world 에 맞춤) | 99.8 % | 81 % | — | 〃 |
  | A gt | 100 % | 81 % | — | 0 |
  | B gt(실시간) | 94.8 % | 86 % | 95 % (0 m), 물체 40 | 0 |
  | B slam(재생, world 에 맞춤) | 98.3 % | 87 % | 95 % (0 m) | 3.5 cm·2.38° / 0.7 cm·0.51° |
  | B odom(재생) | — | — | — | 104 cm·98° (base_qvel wz 가 정답 회전 속도보다 최대 약 40 % 큼) |

  - 사용자가 본 어긋남은 대부분 프레임이었다(slam map = 판 시작 베이스 프레임 — world 와 회전·평행 이동만큼 다름). gt 모드는 world 그대로라 정답 물체·바닥과 바로 겹친다.
  - **영상 늦춤**(B, 돌 때 잘 보임): gt 모드 점유 10 cm 안 = 늦춤 0: 90.2 %, 1: 94.9 %, 2: 97.7 %, 3: 96.1 %. slam yaw 오차가 회전 속도에 비례(−0.027 s × ω, 상관 −0.74), 늦춤 2 에서 최대 yaw 2.53 → 1.22°. 반면 A(머리·몸통만 움직임)는 늦춤 1 이 가장 좋음(100 % vs 97.8 %). proprio 와 시뮬 정답은 같은 스텝(정답 베이스 ∘ 순기구학 = 정답 카메라 0.22 mm, base_qvel ↔ 정답 회전 속도 짝 0 스텝)이라, 베이스가 돌 때만 영상이 한 스텝 더 늦게 그려지는 것으로 보인다(추정 — 렌더 쪽 베이스 변환 갱신). 기본은 1 로 두고(`SGRT_IMAGE_LAG`, 0..7) 따로 본다.
- 실시간 단계 시간(B, sgrt 안, Isaac Sim·다른 에이전트와 CPU 를 나눔 — 재생 표보다 2–3 배): keyframe 당 scan 389·objmap 393·insert 78 µs, image_total 1.74 ms(장치 자르기·구름 색 포함), graph_places 0.98 ms(0.5 s 마다), 검출(YOLO26s) 7.8 ms, 저장은 저장 스레드(평균 11.7 ms, 스텝 밖). 파이썬 쪽: 시뮬 정답 자세 읽기 약 0.6 ms/번(그래서 영상 짝 스텝에만 읽게 바꿈), 준비 21 µs, sgrt_step 호출 평균 1.8 ms(검출 포함).
- 뷰어 8080: B 의 memory(층 쌓기 — 물체·궤적·places·방 4·건물, 물체 그림).

### 3.7 ⑥ 물체 영상 임베딩·이름 캐시 — SigLIP 2 B/32-256(10-03)

물체마다 **영상 벡터 1개(768-d)** 가 원본이고, 이름은 그 벡터와 라벨 표에서 뽑은 캐시다(상위 저장소 `docs/clip_candidates.md` 3.5·8절).
코드는 `src/scene_graph/clip/`(sgclip, [README](../src/scene_graph/clip/README.md)), sgrt 연결은 `runtime/src/sgrt_clip.*`.

- **켜기**: `SGRT_CLIP=1`(기본 엔진 `~/ovdet_models/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan`) 또는 plan 경로. 없으면 꺼짐.
  라벨 표 `SGRT_LABELS`(기본 `~/embed_work/labels/objects-v1`, training/embed 형식), 투영 표본 `SGC_IMG_SAMPLE`.
- **신호**: scenemap 새 ABI `sm_last_views(ctx, updated[], quality[], cap)` — 마지막 영상의 검출 k 가 그 물체 best view 를 바꿨는가와
  모습 품질(유효 마스크 넓이 × 점수). `sm_last_assoc` 과 짝(검출 → 물체 id). capi.cpp 의 best view 확정 고리에서만 적는다.
- **고르기(keyframe 마다)**: ① 임베딩이 없는 물체(품질 큰 것 먼저) ② best view 가 바뀌고 품질이 임베딩 때의 1.2 배 이상(좋아진 비율 순).
  최대 8 개. 같은 물체 검출 둘이면 품질 큰 것.
- **실행**: 원본 RGB(장치) + 검출 상자 + 검출기 마스크 비트 → CUDA 커널 하나(정사각 + 10 % 둘레, 256², 8 × 8 마스크 비율) →
  TensorRT(자기 스트림, 같은 CUDA 문맥). `sgc_submit` 은 커널이 원본을 다 읽을 때까지만 기다린다(커널 실행 3–12 µs, nsys).
  결과는 다음 스텝들의 `poll`(이벤트 질의)로 받는다 — 프레임 고리를 막지 않는다.
- **저장(저장 스레드, `sm_save_dsg_ex` 앞)**:
  - `objects/O<id>_emb.f16`: 768 × FP16, L2 정규화(원본). 바뀐 물체만, 임시 → rename.
  - `cache/names.json`: `{"table": {name, version, sha}, "objects": {"O12": {emb_sha, en: [[이름, 점수] ×5], ko, level, level_ko,
    general, general_ko, score, prob, margin, rolled, structural}}}`. 표 sha 가 바뀌었거나 emb_sha(FNV-1a 64) 가 바뀐 물체만 다시 뽑는다.
  - `cache/index/labels_<표 sha>_<투영>.idx`: 라벨 찾기 색인(처음 1 s 안팎에 만들고 다음부터 읽음).
  - scene.json 노드 metadata: `sm_set_object_meta` 로 `"emb": {path, sha, dim, dtype, model, stamp}`,
    `"names": {en, ko, score, prob, general, general_ko, rolled, structural, table, top}`. dsg_save 는 고치지 않았다.
- **sgrt C ABI(추가)**: `sgrt_clip_enabled`, `sgrt_object_embedding(id, out768)`, `sgrt_query_embedding(q768, k, ids, scores)`
  (살아 있는 물체 전부 — scenemap `structural` 은 "큰·고정 가구" 라 거르지 않음), `sgrt_query_label(text, …)`(라벨 표 영어·한국어
  이름과 같으면 미리 계산한 글 임베딩으로, 없으면 -2), `sgrt_object_names(id, …)`, `sgrt_get_clip_stats`.
- **글 쪽**: 로봇 밖 `clip/tools/text_query.py`(SigLIP 2 글 탑, `--serve` HTTP `/encode`·`/search`). 로봇 안 한국어 학생은
  training/embed 가 같은 약속(글 → 768-d L2)으로 만든다.
- **뷰어**: sgviz 물체 판에 이름(영·한·점수·상위 5·표 sha·emb), FastSAM(`object`)이면 노드 이름 대신 임베딩 이름, 글 찾기 칸
  (`SGVIZ_QUERY_URL`, 기본 `http://127.0.0.1:8091`).

#### 3.7.1 시뮬 확인(10-03, bringing_water public_test 0, 3000 스텝, headless, FastSAM-s 416, SGRT_POSE=gt)

- `outputs/clip_fastsam_20261003_075313/`: 물체 255 개 모두 임베딩·이름(표 objects-v1 `8e57b350d0f88b87`). 부엌: drop in sink / 싱크대,
  oven / 오븐, range hood, countertop, bar stool, ceiling light, couch / 소파, chair / 의자 …
- 글 찾기(`text_query.py`, 1위): "white chair"·"흰 의자" → chair #9(0.130 / 0.127), "sofa" → couch #170, "kitchen sink" → sink #19,
  "오븐" → oven 계열. 이 장면에는 라디오가 없다(라디오는 turning_on_radio — 상위 문서 8절).
- 같은 기억 폴더를 `sgclip_names` 로 다시 돌리면 0 개 다시 뽑음(표·emb_sha 같음).
- turning_on_radio public_test 0(`outputs/clip_fastsam_radio_20261003_081437`): 269 물체, CLIP 773 번. 라디오 O39 이름은 "extinguisher" 로
  틀리지만 "red radio"·"빨간 라디오" 1위, "라디오" 3위, "radio" 5위로 찾는다. `sgc_submit` 은 시뮬과 GPU 를 나눠 2.6–3.9 ms 기다린다
  (커널 자체는 µs — 다음 일: 원본을 다음 스텝까지 붙잡아 기다림을 미루는 선택).

## 4. 약속(인터페이스)

### 4.1 simlink → scenemap (C ABI, 같은 프로세스)

```c
typedef struct { double stamp; const float* proprio; int n_proprio; } sm_proprio;           // 매 스텝, 61 f32 (그 스텝 상태의 stamp)
typedef struct { double stamp; int cam; int w, h; const uint8_t* rgba; const float* depth_m;  // 영상 k: stamp = 장면 시각(k-1)
                 double fx, fy, cx, cy; } sm_image;
int  sm_push_proprio(sm_ctx*, const sm_proprio*);
int  sm_push_image(sm_ctx*, const sm_image*, const sm_detections* dets /* NULL 이면 scenemap 이 YOLOE 를 부름 */);
int  sm_snapshot(sm_ctx*, sm_snapshot_t** out);   // 읽기 전용 스냅숏(참조 카운트), 질의는 여기서
```

- simlink 가 이미 받는 것(평가기 원 텐서: RGBA u8, 깊이 f32 m, proprio 61)을 그대로 넘긴다. 자르기·축소는 scenemap 안에서 한다(ROS 계약의 640×480 은 더 이상 필요 없음).
- 카메라 외부 자세는 scenemap 이 proprio 로 직접 계산한다(순기구학 코드는 `src/agent/src/fk.rs` 와 같은 상수, C++ 로 새로).

- **09-30 구현**: `src/scenemap/include/scenemap.h` 에 통합 담당 제안(`src/integ/scenemap_stub/sm_api.h`, 00d745b)의 이름·형을 그대로 옮기고 `src/scenemap/src/capi.cpp` 로 구현했다(두 헤더는 같은 가드 `SM_API_H`).
  - 지금 되는 것: 자세(slam2d, 스냅숏 때 아직 영상 짝이 안 된 proprio 까지 적분해 최신 stamp 로), 상태, 격자(`sm_snap_map`), `sm_snap_reachable`(8방향 A*, 점유 ≥ 65 % 를 0.30 m 부풀림, 모르는 칸 1.5배, 목표 0.6 m 안 도착, 지도 밖이면 직선 거리).
  - 아직: 물체(`sm_snap_objects/find/near` 는 0 개) — objmap 이 붙으면 채운다. `sm_create` 의 설정 JSON 은 아직 읽지 않는다(기본값).
  - 짝짓기: 영상 stamp 까지 쌓인 proprio 를 적분하고, stamp 가 같은(없으면 그 앞 가장 가까운) proprio 의 순기구학으로 카메라 자세를 만든다. 깊이 표본 간격 = 가로 160 점 안팎(720 → 4 px).
  - 검증: `tools/capi_replay`(ep200 을 C ABI 로 넣음) — slam2d_eval 과 keyframe 자세 차 5e-16 m, 스냅숏 평균 81 µs, A* 0.4 ms. simlink 를 `SCENEMAP_LIB_DIR=<빌드 폴더>` 로 libscenemap.a 에 링크해 `cargo test` 통과(따로 둔 target 폴더).

### 4.2 검출기(YOLOE) 출력 — 약속(확정 09-30, `src/ovdet/include/ovdet.h` 와 `src/scenemap/include/scenemap.h` 가 같은 정의를 `SM_DETECTIONS_DEFINED` 가드로 가짐)

```c
typedef struct {
  double stamp;          // 입력 영상의 stamp 그대로(장면 k-1)
  int cam;               // 0 머리, 1 왼손목, 2 오른손목
  int img_w, img_h;      // 입력 영상 크기(원 텐서)
  int n;                 // 검출 수
  const int32_t*  cls;       // n: 프롬프트 표의 번호(판 시작 때 넘긴 이름 목록의 순서)
  const float*    score;     // n: 신뢰도
  const float*    box;       // n×4: x0,y0,x1,y1 원 영상 화소
  int mask_w, mask_h;        // 마스크 격자(YOLOE 입력 1024 기준 256×256)
  float mask_sx, mask_sy, mask_ox, mask_oy;   // 원 영상 화소 = 마스크 칸 × s + o (레터박스 되돌림)
  const uint32_t* mask_bits; // n × ceil(mask_w×mask_h/32): 행 우선 비트
} sm_detections;
```

- 프롬프트 표: 판 시작 때 계획기가 BDDL 물체 이름 목록을 넘긴다. 검출기와 scenemap 이 같은 표(순서)를 쓴다. 구조물 이름(바닥·벽·문)은 표 뒤쪽에 따로 둔다(문은 방 나누기용).
- 마스크는 검출기 해상도 그대로 비트로 넘긴다(복사 없음, 같은 프로세스). scenemap 이 깊이 화소로 샘플링한다.
- 비트 순서: 칸 k = j·mask_w + i 는 word k>>5 의 bit (k&31)(LSB 먼저). 칸 (i, j) 는 x ∈ [i·sx+ox, (i+1)·sx+ox), y ∈ [j·sy+oy, (j+1)·sy+oy) 를 덮고, 칸 중심은 (i+0.5)·s+o.
- 검출기 호출: `const sm_detections* ovd_detect(h, img, timing)`. 반환값은 핸들이 소유하고 다음 호출 전까지 유효하다. 입력은 RGBA u8 원 텐서(호스트·GPU). cls 는 `ovd_set_prompt` 이름 목록 번호다(BDDL 이름을 정규화한 것, 어휘 밖 이름도 번호는 유지하되 검출은 안 됨). 평가 전용 이름 점수·면적은 별도 getter.
- 한 번 부르기의 수명: 다음 `sm_push_image` 전까지 유효하다. scenemap 은 필요한 것만 복사한다.

### 4.3 계획기 ↔ scenemap

- Rust 트레이트 `SceneQuery`(3.3 표) — `src/agent` 의 그래프 질의 자리(`SceneGraph`)에 들어간다. 계획기 쪽 필드 이름은 지금 JSON(`objects`, `position`, `position_robot`, `moved`, `handled`, `room`)을 유지해 계획기 수정을 줄인다.

## 5. 정확도 평가(채점 도구를 새 스택에 맞게 옮김)

| 무엇 | 도구 | 잴 것 · 통과 조건 |
|---|---|---|
| 자세 | 원본 HDF5 정답 자세(`gt_traj.py` 방식) | 이동 거리별 위치·yaw 오차. 8판 + 긴 판 2개. C(지금 깊이 ICP)보다 나쁘지 않을 것 |
| 2D 지도 | 정답 장면(과제 인스턴스의 벽·가구 상자를 바닥 단면으로) | 점유 칸 정밀도·재현율(5 cm 여유), 벽 두께 |
| 물체 위치 | 정답 물체 궤적 | 보인 물체마다: 찾음, 위치 오차(중앙·최대), 두 번 등록된 수, 둘이 하나로 합쳐진 수(팀 문서 '잴 것') |
| 지도 갱신 | ep200(쓰레기 줍기) 등 옮겨짐이 있는 판 | 옮겨짐·사라짐·생김 판정 P/R/F1(팀 문서), 판정까지 걸린 시간 |
| 검출기 없이 물체 지도만 | 정답 상자를 깊이에 투영한 '완벽한 검출'(`gt_frontend.py` 방식) | DA·갱신 논리만 따로 검증 |
| 지연 | 같은 입력 재생 | keyframe 당 slam2d·objmap p50/p99(CPU·GPU 판 각각), 질의 p50/p99 |
| CPU = CUDA | CUDA 로 옮기는 부분마다 | 같은 입력에서 비트 같음, 또는 이유 있는 허용 오차(합 순서) |

## 6. 순서

1. ~~slam2d CPU 기준판: 가상 스캔 + 격자 + 매칭 후보 A·B, 기준선 C 와 10판 비교 → 고름 → 루프 닫기 판단.~~ 09-30 끝: B, 루프 닫기 없음(3.1.1).
2. objmap CPU: '완벽한 검출'로 DA·갱신·들고 있는 물체 검증 — **진행 중(09-30 멈춤)**: ep0·200·201·204 는 찾음 88~100%, ep3000 54%(3.2.1).
3. C ABI — **slam2d 부분 끝(09-30, 4.1)**: sm_api.h 이름·형을 scenemap.h 로 옮김, simlink 가 SCENEMAP_LIB_DIR 로 링크해 cargo test 통과. 남은 것: objmap 을 capi 에 붙이기(sm_push_image 의 dets → ObjectMap, 스냅숏 물체 표·find·near, set_labels 이름 표, mark_handled), 설정 JSON 읽기.
4. YOLOE 출력 연결(C++ frontend 에이전트, ovdet.h 가 4.2 에 맞춤) → 실제 검출로 같은 채점.
5. 느린 곳만 CUDA(slam2d 는 필요 없음, 3.1.1), Spark-DSG 저장·뷰어, 방 나누기.

### 6.1 다음에 이어서 할 것(09-30 마무리 시점)

1. objmap 을 C ABI 에 붙이기: `capi.cpp` 의 `sm_push_image` 에서 dets(NULL 이면 나중에 YOLOE) → `ObjectMap::update`(T_mc = slam2d 자세 ∘ 순기구학 머리 카메라, 팔 끝 = proprio 17:20·42:45 를 map 으로, 그리퍼 = 24+25·49+50, base_yaw), 매 proprio 에 `updateHands`. 스냅숏에 `sm_object` 표(이름 = set_labels 표, state·handled·first_pos), find(이름 부분 일치·점수 순)·near.
2. ep3000 objmap 54% 원인 보기(정답 자세로도 같음 → DA·거르기 쪽). 나무 토막 정답(과제 인스턴스 상자 밖) 넣기.
3. objmap 채점을 `capi_replay` 처럼 C ABI 로 한 번 더(자세·물체가 slam2d_eval·objmap_eval 과 같은지).
4. YOLOE(ovdet) 실제 검출로 같은 채점 — 이름 틀림·작은 물체.
5. slam2d 남은 것: 짧은 판 204·205 가 C 보다 1~2 cm 나쁨(3.1.1). 루프 닫기는 안 넣음(끝 5 cm) — objmap 채점에서 긴 판 물체 위치가 틀리면 다시 본다.
6. Spark-DSG 저장(submodule 을 src/scenemap/third_party 로 새로 둘지 — 저장·뷰어가 필요해질 때 판단), 방 나누기.

재현: `~/scenemap_eval/`(ep_*.bin·ep_*_det.bin 입력은 남겨 둠, `export_episode.py`·`export_gtdet.py` 로 다시 만들 수 있음), 빌드 `cmake src/scenemap` → `slam2d_eval`·`objmap_eval`·`capi_replay`·`test_fk`.

## 7. 아직 모르는 것

| 무엇 | 왜 | 어떻게 |
|---|---|---|
| 머리만으로 가상 스캔이 충분한가 | 머리 카메라 시야는 약 100° 이고 로봇이 돌 때만 옆을 본다 | 09-30: 레이저 한 줄로는 모자람(3.1.1 `B 레이저`), 수직면 점을 다 쓰면 충분. 손목 깊이는 안 씀 |
| ~~긴 판에서 누적 오차~~ | 09-30 쟀다: B 로 43 m 판 최대 7.8 cm, 83 m 판 25 cm(끝 5 cm) | 3.1.1 |
| 같은 이름이 여러 개 | 팀 문서의 약점(컵 두 개) | 색 분포 + 헝가리안. ep200 캔 3개로 잰다 |
| YOLOE 이름 품질 | BDDL 이름이 영상 말과 다를 수 있다(`radio_receiver`) | 프롬프트 표에 사람 말 이름을 같이 두고(검출기 쪽), 이름 틀림 비율을 잰다 |
