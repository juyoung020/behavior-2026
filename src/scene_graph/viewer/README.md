# sgviz — 물체 기억 실시간 뷰어 (Spark-DSG + viser)

메모리 런타임이 ~1 s 마다 다시 쓰는 디렉터리(`scene.json`, `map.pgm`/`map.yaml`,
`objects/O<id>_{rgb,depth}.png`)를 브라우저에서 실시간으로 본다.
`spark_dsg.viser.ViserRenderer` 위에 얇게 얹은 것(그래프 읽기·상자 모서리는 spark_dsg API).

## 실행

```bash
~/sdsg_venv/bin/python src/scene_graph/viewer/sgviz.py <memory_dir> [--port 8080]
# 예: outputs/mem_comet12_water_gui_*/memory
```

그다음 브라우저에서 http://localhost:8080 . (원격이면 `ssh -L 8080:localhost:8080`.)

환경: `~/sdsg_venv`(python 3.11, spark_dsg) 에 `viser`, `pillow` 추가
(`VIRTUAL_ENV=~/sdsg_venv uv pip install viser pillow`).

## 보이는 것

- 바닥: 2D 점유 지도(회색 단계 그대로, 205 = 미지 = 청회색).
- 로봇: 검은 원판 + 빨간 화살표(robot_pose x, y, yaw).
- 물체: 상태별 색 구(seen 초록, moved 주황, held 파랑, gone 회색 반투명), `name#id` 이름표,
  경계 상자(한 변이 "Max box side" 보다 크면 그 길이로 잘라 그림).
- 지지 관계: 물체→받침 선 + `on`/`in` 이름표(edge metadata `relation`).
- moved 물체: first_pos → pos 궤적.
- 오른쪽 패널 "Object": 노드를 누르거나 드롭다운에서 고르면 name/id/state/pos/first_pos/n_obs/
  score/마지막 관측/관계, `metadata.rgbd` 가 있으면 RGB·깊이(색입힘) 조각과 box_px/depth_m.
  데이터가 바뀌면 패널도 갱신. 그림은 고를 때만 읽는다(경로+mtime 캐시).
- "Display" 폴더: 이름표/상자/관계/궤적/구조물/gone/지도 켜고 끄기, 노드 크기.

scene.json 의 mtime 을 0.25 s 마다 보고, 바뀐 노드·간선만 다시 그린다(카메라 유지).

## 시험

```bash
~/sdsg_venv/bin/python src/scene_graph/viewer/test_sgviz.py
```

합성 디렉터리(물체 3, rgbd 1, 지지 간선 1)를 spark_dsg 로 만들어 읽기·차이 계산·그림 자르기·
지도 텍스처를 확인하고, 실제 viser 서버를 띄워 선택 패널과 HTTP 200 을 확인한다(브라우저 없음).

## 참고

- `rgbd.rgb/depth` 그림이 전체 프레임이면 box_px 로 자르고(여백 8 px), 이미 조각이면 그대로 쓴다.
- spark_dsg 의 `ViserRenderer.draw()`(GraphHandle) 는 viser 1.x 에서 없어진 API
  (`server.add_folder` 등)를 써서 그대로는 안 돈다. 그래서 서버 소유만 물려받고 그리기는 여기서 한다.
- 예전 Rust 시제품(sgview)은 지웠다. 이미 빌드된 `target/` 은 git 무시 대상으로 남아 있다.
