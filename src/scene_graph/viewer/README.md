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
- 물체 형태: `metadata.points.path`(`objects/O<id>_points.ply`, binary_little_endian, float x,y,z 지도 좌표 m
  + uchar red,green,blue)의 세그먼트 점을 실제 색으로 그린다(점 크기 = voxel × 배율,
  "Point colour" 로 true colour / state colour 전환). PLY 는 numpy 로 직접 읽고,
  파일 mtime·크기가 바뀔 때만 다시 읽는다(scene.json 보다 늦게 써져도 다음 poll 에 잡힘).
- 점이 있는 물체는 경계 상자를 그리지 않는다. 상자는 점이 없는 물체에만, "Boxes" 를 켰을 때만
  (기본 꺼짐; 한 변이 "Max box side" 보다 크면 잘라 그림). bounding_box 자체는 scene.json 에 그대로 있다.
- `name#id` 이름표(점 위). 중심점은 선택 사항("Centre markers", 기본 꺼짐). 점이 없는 물체는 상태별 색 구
  (seen 초록, moved 주황, held 파랑, gone 회색 반투명).
- 지지 관계: 물체→받침 선 + `on`/`in` 이름표(edge metadata `relation`).
- moved 물체: first_pos → pos 궤적.
- 고르기: 물체의 점(복셀) 아무 데나 누르면 그 물체가 선택된다. viser 장면 클릭
  (`server.scene.on_click`)의 광선(origin, direction)으로 numpy 에서 고른다: 물체마다 점 배열과
  (반경만큼 키운) AABB 를 들고, 광선이 AABB 를 지나는 물체만 점-광선 수직 거리 ≤ max(2×voxel, 2 cm)
  인 점을 찾아 광선 방향으로 가장 가까운 물체를 고른다. 점이 없는 물체는 구(반경 1.5×)로 맞춘다.
  빈 곳을 누르면 선택은 그대로(드래그는 카메라 회전). 50 물체 × 4000 점 최악 ~1.7 ms.
  선택된 점 구름은 노랗게 물들이고 조금 키워 보인다(구 물체는 노란 철망 구).
- 오른쪽 패널 "Object": 물체를 누르거나 드롭다운에서 고르면 name/id/state/pos/first_pos/n_obs/
  score/마지막 관측/관계/점 수, `metadata.rgbd` 가 있으면 box_px/depth_m 과 조각 그림:
  - RGB 조각 + 세그먼트(`rgbd.mask` = `objects/O<id>_mask.png`, 8-bit 255 = 안쪽, 같은 조각 상자):
    바깥은 어둡게, 경계는 노란 선.
  - 깊이 조각은 회색조(가까울수록 밝게). 범위는 마스크 안 유효 깊이의 2–98 백분위, 마스크 밖은 어둡게,
    0(무효)은 검정. 아래에 마스크 안 깊이 최소/중앙/최대 [m] (마스크가 없으면 조각 전체).
  - 작은 조각은 정수배로 키워 보여 준다.
  데이터가 바뀌면 패널도 갱신. 그림은 고를 때만 읽는다(경로+mtime 캐시).
- "Display" 폴더: 이름표/세그먼트 점/점 색/점 크기/중심점/상자/관계/궤적/구조물/gone/지도, 노드 크기.

scene.json 의 mtime 을 0.25 s 마다 보고, 바뀐 노드·간선만 다시 그린다(카메라 유지).

## 시험

```bash
~/sdsg_venv/bin/python src/scene_graph/viewer/test_sgviz.py
```

합성 디렉터리(물체 3, 점 PLY·rgbd·마스크 1, 지지 간선 1)를 spark_dsg 로 만들어 읽기·PLY(이진/ascii)·
차이 계산·마스크 겹침·회색조 깊이와 통계·지도 텍스처·"점 있으면 상자 없음"·PLY 갱신 감지,
광선 고르기(맞음/앞의 것 우선/빗나감/시간)와 뷰어 안에서 클릭→패널을 확인하고, 실제 viser 서버를 띄워 선택 패널과 HTTP 200 을 확인한다(브라우저 없음).

## 참고

- `rgbd.rgb/depth/mask` 그림이 box_px 크기면 그대로, 전체 프레임이면 box_px 로 자른다(여백 8 px).
- spark_dsg 의 `ViserRenderer.draw()`(GraphHandle) 는 viser 1.x 에서 없어진 API
  (`server.add_folder` 등)를 써서 그대로는 안 돈다. 그래서 서버 소유만 물려받고 그리기는 여기서 한다.
- 예전 Rust 시제품(sgview)은 지웠다. 이미 빌드된 `target/` 은 git 무시 대상으로 남아 있다.
