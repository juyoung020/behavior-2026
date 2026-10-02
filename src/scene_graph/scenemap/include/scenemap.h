/* scenemap C ABI (docs/scenemap_설계.md 3.3·4절). simlink(Rust)·평가기 프로세스가 같은 프로세스에서 부른다.
 *
 * 함수 이름·형은 통합 담당 제안(src/sim/integ/scenemap_stub/sm_api.h, 00d745b)을 그대로 옮긴 것이다. 두 헤더는 같은 가드
 * (SM_API_H)를 써서 어느 쪽을 먼저 include 해도 한 번만 정의된다.
 *
 * 스레드: sm_push_* · sm_reset 은 한 스레드(simlink 관측 스레드)에서. sm_snapshot 과 sm_snap_* 는 아무 스레드에서
 * (읽기 전용 스냅숏, 참조 카운트). sm_set_labels·sm_mark_handled 는 계획기 스레드에서 온다(scenemap 이 잠금, 다음 스냅숏부터 보임).
 * 시각: stamp 는 전부 시뮬 시각 [s](판 시작 = 0). 영상 k 의 stamp = 장면 시각 k-1, proprio 는 그 스텝 상태의 stamp.
 * 짝짓기: 영상은 stamp 가 같은 proprio(없으면 그 앞 가장 가까운 것)의 순기구학 카메라 자세로 올린다. base_qvel 은
 * proprio i 가 (i-1 → i) 구간 속도다(학습 데모 정답에서 잰 짝, 3.1.1).
 */
#ifndef SCENEMAP_H
#define SCENEMAP_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 검출기(YOLOE, src/scene_graph/ovdet) 출력 — 4.2 절 약속. ovdet.h 와 똑같은 정의라 둘 다 include 해도 된다. */
#ifndef SM_DETECTIONS_DEFINED
#define SM_DETECTIONS_DEFINED
typedef struct {
  double stamp;               /* the input image's stamp, unchanged */
  int cam;                    /* 0 head, 1 left wrist, 2 right wrist (passed through) */
  int img_w, img_h;           /* input image size */
  int n;                      /* number of detections (score order) */
  const int32_t* cls;         /* n: index into the prompt table (order of the names given to ovd_set_prompt) */
  const float* score;         /* n: confidence */
  const float* box;           /* n x 4: x0, y0, x1, y1 in input pixels */
  int mask_w, mask_h;         /* mask grid */
  float mask_sx, mask_sy, mask_ox, mask_oy; /* input pixel = mask cell x s + o: cell (i, j) covers
                                               x in [i sx + ox, (i+1) sx + ox), y in [j sy + oy, (j+1) sy + oy) */
  const uint32_t* mask_bits;  /* n x ceil(mask_w x mask_h / 32) words: row-major bits, cell k = j mask_w + i is bit
                                 (k & 31) (LSB first) of word k >> 5 */
} sm_detections;
#endif

#ifndef SM_API_H
#define SM_API_H

typedef struct sm_ctx sm_ctx;
typedef struct sm_snapshot_t sm_snapshot_t;

typedef struct { double stamp; const float* proprio; int n_proprio; } sm_proprio;          /* 매 스텝 61 f32 */
typedef struct {
  double stamp; int cam; int w, h;       /* cam 0 머리, 1 왼손목, 2 오른손목. 원 텐서 크기 */
  const uint8_t* rgba;                   /* w×h×4 (NULL 가능) */
  const float* depth_m;                  /* w×h 미터 (NULL 가능) */
  double fx, fy, cx, cy;                 /* 그 해상도의 내부 파라미터(평가기 eval_utils.CAMERA_INTRINSICS) */
} sm_image;

typedef struct { double stamp; double x, y, yaw; } sm_pose2;

enum { SM_SEEN = 0, SM_GONE = 1, SM_MOVED = 2, SM_HELD = 3 };
typedef struct {
  uint32_t id;
  const char* name;           /* 프롬프트 표 이름(스냅숏 수명 동안 유효) */
  float score;
  double pos[3], extent[3], first_pos[3];   /* map */
  uint32_t n_obs;
  double last_seen;           /* 시뮬 시각 */
  int32_t state;              /* SM_SEEN.. */
  int32_t handled;            /* 계획기 표시 */
  int32_t structural;
} sm_object;

typedef struct { double resolution; double origin[2]; int32_t width, height; const int8_t* cells; } sm_grid;

typedef struct {
  double last_proprio_stamp, last_image_stamp;  /* 반영된 마지막 입력 시각 */
  int32_t n_objects, n_images, n_proprio;
} sm_status;

/* 만들기·판 */
sm_ctx* sm_create(const char* config_json);                  /* NULL = 기본값 */
void    sm_destroy(sm_ctx*);
int     sm_set_labels(sm_ctx*, const char* const* names, int n);   /* 프롬프트 표(검출기와 같은 순서) */
int     sm_reset(sm_ctx*);                                    /* 새 판: 지도·물체 비우고 원점 */
/* 입력 */
int     sm_push_proprio(sm_ctx*, const sm_proprio*);
int     sm_push_image(sm_ctx*, const sm_image*, const sm_detections* dets /* NULL: scenemap 이 검출기(YOLOE)를 부름 */);
int     sm_mark_handled(sm_ctx*, uint32_t id);
/* 질의(스냅숏) */
int     sm_snapshot(sm_ctx*, sm_snapshot_t** out);
void    sm_snapshot_release(sm_snapshot_t*);
sm_pose2  sm_snap_pose(const sm_snapshot_t*);
sm_status sm_snap_status(const sm_snapshot_t*);
int     sm_snap_objects(const sm_snapshot_t*, const sm_object** out);        /* 개수, 배열은 스냅숏 수명 동안 */
int     sm_snap_find(const sm_snapshot_t*, const char* name, uint32_t* ids, float* scores, int cap);   /* 점수 순 */
int     sm_snap_near(const sm_snapshot_t*, const double p[3], double r, uint32_t* ids, int cap);        /* 가까운 순 */
/* 격자: cells[y·width + x] 는 칸 (x, y), 칸 왼쪽 아래 모서리 = origin + (x, y)·resolution. −1 모름, 0..100 점유 % */
int     sm_snap_map(const sm_snapshot_t*, sm_grid* out);
double  sm_snap_reachable(const sm_snapshot_t*, const double from[2], const double to[2]);  /* 경로 길이 m, < 0 = 못 감 */

#endif /* SM_API_H */

/* ---- 이름 종류(추가 ABI) ----
 * 구조물(SM_KIND_STRUCTURE): 물체 노드가 안 되고 2D 격자만(기본: wall, floor, ceiling, door, doorway, door frame, window,
 *   pillar, column, partition, staircase, stairs, stair, railing, baseboard).
 * 고정(SM_KIND_STATIC): 가구·가전·붙박이 — 물체 노드지만 movable = false, 사라짐 판정 안 함, 상자는 한도 있는 합집합
 *   (기본: table, desk, counter, sofa, shelf, cabinet, bed, refrigerator, oven, sink, lamp, plant, picture frame, rug,
 *   curtain, radiator, light switch, electric outlet ... — capi.cpp kStaticNames).
 * 나머지는 옮길 수 있는 물체(SM_KIND_OBJECT). 이름 비교는 정규화(".n.NN" 버림, '_'→' ', 소문자) 뒤 머리 명사:
 *   이름 == 항목 이거나 " 항목" 으로 끝남("glass door" → door, "floor lamp" → lamp). 구조물 표가 먼저.
 * sm_set_kind_names 는 그 종류의 표를 통째로 바꾼다(names == NULL: 기본 표로). 지금 labels 에 바로 적용되고,
 * 이미 만들어진 물체는 그대로 둔다(sm_reset 뒤부터 깨끗). */
enum { SM_KIND_OBJECT = 0, SM_KIND_STRUCTURE = 1, SM_KIND_STATIC = 2 };
int sm_set_kind_names(sm_ctx*, int32_t kind, const char* const* names, int32_t n);
/* 스냅숏 물체 id 가 옮길 수 있는 것인가: 1 / 0(고정), -1 = 없음 */
int sm_snap_movable(const sm_snapshot_t*, uint32_t id);

/* ---- 물체별 RGB-D best view(추가 ABI, 위 함수·구조체는 그대로) ----
 * objmap 이 물체에 붙인 검출마다 품질 = 유효 마스크 넓이 × 점수 가 가장 큰(같으면 최근) 모습 하나를 물체마다 둔다.
 * RGB 자르기: 상자 + 변마다 10 % 여유, 긴 변 최대 256 px(넓이 평균으로 줄임). scenemap 은 자를 영역과 출력 버퍼만
 * 정하고, 실제 자르기는 호출자 함수가 한다(sgrt: 장치 메모리에서 CUDA 로 자르고 자른 것만 내려받음). 깊이는 호스트에서. */
typedef struct {
  int32_t x0, y0, x1, y1;      /* 원(검출 입력) 영상 화소, [x0, x1) × [y0, y1) */
  int32_t out_w, out_h;        /* 출력 크기(줄였으면 넓이 평균) */
  uint8_t* dst;                /* out_w × out_h × 3 RGB8, scenemap 이 준 호스트 버퍼 */
} sm_crop_req;
/* reqs 를 다 채우면 0. 실패하면 그 keyframe 의 새 모습은 버린다. scenemap 잠금 밖에서 불린다. */
typedef int (*sm_crop_fn)(void* user, const sm_crop_req* reqs, int32_t n);

/* sm_push_image 와 같고, best view 를 고칠 검출이 있으면 crop(user, ..) 으로 RGB 를 자른다.
 * crop == NULL 이면 im->rgba(호스트, w×h×4)에서 자르고, 그것도 없으면 best view 는 건너뛴다. */
int sm_push_image_ex(sm_ctx*, const sm_image*, const sm_detections*, sm_crop_fn crop, void* user);
/* 마지막 영상의 검출 k → 물체 id(0 = 안 붙음). 검출 수를 돌려주고 ids 에 min(n, cap) 개. */
int sm_last_assoc(sm_ctx*, uint32_t* ids, int cap);

typedef struct {
  uint32_t id, version;        /* version: 모습이 바뀔 때마다 +1 */
  double stamp;
  int32_t box_px[4];           /* 자른 영역 x0, y0, x1, y1(원 영상 화소, 여유 포함) */
  int32_t det_box_px[4];       /* 검출 상자 */
  float mask_area;             /* 유효 마스크 넓이(깊이 화소) */
  float depth_m;               /* 마스크 안 깊이 중앙값 */
  float score;
  double cam_T[12];            /* map ← 카메라 광학, 행 우선 3×4 */
  int32_t w, h;                /* 자른 그림 크기 */
  const uint8_t* rgb;          /* w×h×3(NULL = RGB 없음), 스냅숏 수명 동안 */
  const uint16_t* depth_mm;    /* w×h, 0 = 깊이 없음 */
} sm_view;
/* 스냅숏 안 물체 id 의 best view. 1 = 있음, 0 = 없음, < 0 = 오류. */
int sm_snap_view(const sm_snapshot_t*, uint32_t id, sm_view* out);

/* 저장(로봇 기억). dir 에 세 파일을 원자적으로(임시 파일 → rename) 바꿔 쓴다:
 *   scene.json — Spark-DSG DynamicSceneGraph(OBJECTS 층: 확정 물체 노드, 이름·위치 xyz·상자·상태 메타데이터)
 *                (물체 best view 가 있으면 노드 metadata.rgbd = 그림 경로·stamp·상자·넓이·깊이·카메라 자세 — sm_save_dsg_ex)
 *   view.json  — 계획기·뷰어용 요약(자세, 물체 표, 최근 사건)
 *   map.pgm    — 2D 점유 격자(+ map.yaml: 해상도·원점)
 * Spark-DSG 없이 빌드하면(SM_HAVE_SPARK_DSG 미정의) scene.json 은 건너뛴다. 0 = 성공. */
int sm_save_dsg(sm_ctx*, const char* dir);
/* sm_save_dsg + best view PNG(dir/objects/O<id>_rgb.png · O<id>_depth.png, 지난 저장 뒤 바뀐 것·없는 것만 씀)와 시간.
 * sm_save_dsg 도 같은 일을 한다(stats 만 없음). scene.json 노드 metadata.rgbd, view.json objects[].rgbd 가 경로를 가리킴. */
typedef struct {
  int32_t n_objects;           /* 저장한 물체 노드 수 */
  int32_t n_png;               /* 이번에 새로 쓴 PNG 수 */
  float png_ms, total_ms;
} sm_save_stats;
int sm_save_dsg_ex(sm_ctx*, const char* dir, sm_save_stats* stats /* NULL 가능 */);

#ifdef __cplusplus
}
#endif
#endif /* SCENEMAP_H */
