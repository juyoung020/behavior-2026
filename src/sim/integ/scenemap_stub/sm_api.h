/* scenemap C ABI — simlink 가 부르는 함수들(제안, docs/scenemap_설계.md 3.3·4.1 을 그대로 C 로).
 *
 * src/scene_graph/scenemap/include/scenemap.h 에는 아직 sm_detections 만 있다. 이 파일은 simlink(통합)가 기대하는 나머지 함수의
 * 선언이다. scenemap 담당이 scenemap.h 로 옮기면(같은 이름·같은 형) simlink 는 이 파일 대신 그것을 쓰고, 가짜 구현
 * (sm_stub.cpp) 대신 libscenemap 을 링크한다(simlink build.rs, 환경변수 SCENEMAP_LIB_DIR).
 *
 * 스레드: sm_push_* · sm_reset 은 한 스레드(simlink 관측 스레드)에서. sm_snapshot 과 sm_snap_* 는 아무 스레드에서
 * (읽기 전용 스냅숏, 참조 카운트). sm_set_labels·sm_mark_handled 는 계획기 스레드에서 온다(scenemap 이 잠금, 다음 스냅숏부터 보임).
 * 시각: stamp 는 전부 시뮬 시각 [s](판 시작 = 0). 영상 k 의 stamp = 장면 시각 k-1, proprio 는 그 스텝 상태의 stamp.
 */
#ifndef SM_API_H
#define SM_API_H
#include <stdint.h>
#include "scenemap.h" /* sm_detections */

#ifdef __cplusplus
extern "C" {
#endif

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
int     sm_snap_map(const sm_snapshot_t*, sm_grid* out);
double  sm_snap_reachable(const sm_snapshot_t*, const double from[2], const double to[2]);  /* 경로 길이 m, < 0 = 못 감 */

#ifdef __cplusplus
}
#endif
#endif
