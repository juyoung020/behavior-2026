/* scenemap C ABI (docs/scenemap_설계.md 3.3·4절). simlink(Rust)·평가기 프로세스가 같은 프로세스에서 부른다.
 *
 * 함수 이름·형은 통합 담당 제안(src/integ/scenemap_stub/sm_api.h, 00d745b)을 그대로 옮긴 것이다. 두 헤더는 같은 가드
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

/* 검출기(YOLOE, src/ovdet) 출력 — 4.2 절 약속. ovdet.h 와 똑같은 정의라 둘 다 include 해도 된다. */
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

#ifdef __cplusplus
}
#endif
#endif /* SCENEMAP_H */
