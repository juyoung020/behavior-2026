# realbag — 공개 실제 로봇 ROS bag 으로 지각 파이프라인 돌리기

ROS 를 설치하지 않고 실제 로봇 bag 을 우리 파이프라인(ovdet 검출 + scenemap SLAM·물체 기억·장면 그래프)에 넣고, sgview 로 실시간처럼 다시 본다.
잰 값과 실패 사례는 robot-agent `docs/map_vla/MAP_STATE_PLAN.md` "실제 데이터" 절.

| 파일 | 하는 일 |
|---|---|
| `bag2stream.py` | (오프라인 변환만, 파이썬 `rosbags`) bag → 스트림 폴더: `rgb/`·`depth/`(uint16 mm, 컬러에 맞춘 깊이) PNG, `frames.csv`(영상 시각·채점용 정답), `odom.csv`(바퀴 오도메트리 전부), `meta.json`(내부 파라미터, base ← 카메라 `T_bc`). 종류 `openloris`(TF 그대로) · `tum_pioneer`(TF 가 이름뿐이라 깊이 바닥 평면으로 높이·기울기). `--refit-only` 는 `T_bc` 만 다시 |
| `realbag_run.cpp` | 스트림 → 검출(`--det fastsam` = FastSAM-s + SigLIP 2 이름, `yoloe`, `none`) → scenemap(`--robot limo_omx` 기본, `--pose slam\|odom\|gt`). 카메라 외부 자세는 `sm_set_cam_extrinsic`, 오도메트리는 LIMO proprio 0–5(영상 시각에 보간 하나 더). 검출 캐시 `--dump/--load`, 정답 비교(ATE: SE(2) 맞춤·첫 프레임 맞춤, 오도메트리만 대비), `--ref-map`(정답 자세로 만든 지도와 점유 칸 비교), 여러 판을 한 지도에(`a,b,…` + `--pose gt`), 재생 판 `--sg <run>`(stream.sgs·memory·cam·meta.json, 학습 뷰어 group `real_bags`), 실시간 `--live host:port` |
| `sgs_play.cpp` | 기록한 `stream.sgs` 를 벽시계에 맞춰 sgview(`--ingest`)로: 자세 60 Hz 보간, 지도·요약은 기록 그대로. `--rate`·`--loop` |
| `record_live.mjs` · `make_video.sh` | 머리 없는 Chrome 으로 sgview 화면을 찍고(재생과 함께) bag RGB 를 작은 창으로 붙여 MP4·GIF |
| `rb_util.hpp` | JSON·폴더·JPEG·스트림 받기(Capture) |

```bash
python3 -m venv ~/realbag_venv && ~/realbag_venv/bin/pip install rosbags numpy opencv-python-headless
~/realbag_venv/bin/python src/scene_graph/tools/realbag/bag2stream.py openloris office1-1.bag ~/streams/ol_office1-1
cmake -S src/scene_graph/tools/realbag -B ~/realbag_build && cmake --build ~/realbag_build -j 4
~/realbag_build/realbag_run ~/streams/ol_office1-1 out/ol11 --dump dets/ol11.gz --sg ~/trainview_work/real_bags/ol_office1-1
sgview <run>/replays/<ep>.sg/memory --port 8080 --ingest 127.0.0.1:9001 &
~/realbag_build/sgs_play <run>/replays/<ep>.sg/stream.sgs 127.0.0.1:9001 --rate 1 --loop
```

데이터: OpenLORIS-Scene(CC BY-ND 4.0 — 원본 영상은 고치지 않고 출처 표시, 파생 데이터셋 배포 금지), TUM RGB-D fr2 pioneer(CC BY 4.0). bag·변환 결과는 커밋하지 않는다.
