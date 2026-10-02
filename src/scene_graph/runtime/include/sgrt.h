/* sgrt — scene graph runtime: 평가기(또는 로봇) 프로세스 안에서 ① 물체 기억을 실시간으로 굴리는 C ABI 하나.
 *
 *   매 스텝   sgrt_step(proprio)                      → scenemap 자세 적분, 든 물체 따라가기
 *   keyframe  sgrt_step(proprio + 머리 RGB + 깊이)    → ovdet(YOLOE, TensorRT) 검출 → scenemap objmap 갱신
 *   주기 저장 시뮬 시각 save_s 마다 sm_save_dsg       → out_dir/scene.json(Spark-DSG) · view.json · map.pgm
 *
 * keyframe 인지는 sgrt_want_image() 가 알려 준다(호출자는 그 스텝에만 영상 포인터를 넘기면 된다 — 매 스텝 복사 없음).
 * RGB 는 호스트나 장치 메모리(ovdet 이 장치에서 바로 읽음), 깊이는 호스트 f32 미터(scenemap 은 CPU).
 * 스레드: 한 스레드에서 부른다. 파이썬 없음 — 평가기 쪽 접착부는 포인터만 넘긴다(src/scene_graph/runtime/glue).
 */
#ifndef SGRT_H
#define SGRT_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sgrt sgrt;

typedef struct {
  const char* engine;       /* YOLOE TensorRT plan (ovdet) */
  const char* names;        /* <engine>.names.txt */
  const char* out_dir;      /* 저장 디렉터리 */
  int32_t kf_every;         /* 몇 스텝마다 keyframe (6) */
  double save_s;            /* 저장 주기, 시뮬 초 (1.0) */
  float conf_th;            /* 검출 신뢰도 (0.25) */
} sgrt_config;

void   sgrt_default_config(sgrt_config* c);
sgrt*  sgrt_create(const sgrt_config* c, char* err, size_t err_len);
void   sgrt_destroy(sgrt*);
/* 새 판: 지도·물체 비우고, 프롬프트 표(이 판에서 찾을 물체 이름) 지정.
 * 엔진: sgrt_config.engine 이 YOLOE(큰 열린 어휘, 프롬프트로 켜고 끔) 또는 닫힌 어휘 YOLO11/YOLO26-seg(COCO-80) plan.
 * 환경 변수 SGRT_PROMPT = task | all | auto(기본). auto 는 엔진 어휘가 200 이하(닫힌 어휘)면 all — 엔진 이름 전부를 표로
 * 쓰고(과제 이름 중 어휘 밖 것은 err 에 적음), 아니면 task(prompt 그대로). prompt 가 NULL·0 개면 늘 all.
 * 이름 종류(구조물·고정·옮길 수 있음)는 scenemap 기본 표: COCO 의 dining table·couch·bed·refrigerator·oven·sink·tv·toilet·
 * microwave·potted plant·bench 는 고정, person 은 구조물(노드 안 됨), 나머지(cup·bottle·book·chair …)는 옮길 수 있음. */
int    sgrt_begin(sgrt*, const char* const* prompt, int32_t n, char* err, size_t err_len);
/* 이름 종류 표 바꾸기(scenemap sm_set_kind_names 그대로: kind 1 구조물 — 노드 안 됨, 2 고정 가구·가전 — movable=false;
 * names == NULL 이면 기본 표). sgrt_begin 앞뒤 아무 때나. */
int    sgrt_set_kind_names(sgrt*, int32_t kind, const char* const* names, int32_t n);
/* 다음 sgrt_step 에 영상을 넣어야 하는가 */
int    sgrt_want_image(const sgrt*);
/* 한 스텝. rgb/depth 는 keyframe 이 아니면 NULL. */
int    sgrt_step(sgrt*, double stamp, const float* proprio, int32_t n_proprio,
                 const uint8_t* rgb, int32_t rgb_on_device, int64_t row_stride, int32_t pix_stride, int32_t w, int32_t h,
                 const float* depth_m, double fx, double fy, double cx, double cy);
int    sgrt_save(sgrt*);    /* 지금 저장 */
/* 통계: keyframe 수, 마지막 검출 수, 확정 물체 수, 마지막 검출 ms, 마지막 저장 ms */
void   sgrt_stats(const sgrt*, int32_t* n_kf, int32_t* n_det, int32_t* n_obj, float* det_ms, float* save_ms);

/* 마지막 keyframe·저장 시간(ms). kf_ms = scenemap 갱신 전체(자르기 포함), crop_ms = best view RGB 자르기(장치 커널 +
 * 자른 것만 내려받기), n_crops = 그때 자른 물체 수. */
typedef struct {
  float det_ms, kf_ms, crop_ms, save_ms;
  int32_t n_crops;
  int32_t n_png;            /* 마지막 저장에서 새로 쓴 PNG 수 */
  int32_t n_ply;            /* 마지막 저장에서 새로 쓴 점 구름 PLY 수 */
  float gather_ms;          /* 마지막 keyframe: 구름 점 색 모으기(장치에서 남긴 화소만 → 호스트) */
  int32_t n_points;         /* 그때 색을 모은 점 수 */
} sgrt_timing;
void   sgrt_get_timing(const sgrt*, sgrt_timing* out);


/* 지도 보기(탐색·안전 정지용): 지금 스냅숏의 2D 점유 격자·자세·방 격자. 포인터는 다음 sgrt_map 이나 sgrt_destroy 까지 유효.
 * cells[y·w + x]: −1 모름, 0..100 점유 %(scenemap sm_snap_map 그대로), 칸 (x, y) 왼쪽 아래 = origin + (x, y)·res.
 * pose = map 기준 로봇 베이스 (x, y, yaw rad). room_ids 는 방 나누기가 없으면 NULL. 반환 0 = 성공. */
typedef struct {
  double stamp;
  double pose[3];
  double res;
  double origin[2];
  int32_t w, h;
  const int8_t* cells;
  double room_res;
  double room_origin[2];
  int32_t room_w, room_h;
  const uint32_t* room_ids;
  int32_t n_rooms;
  /* 마지막 keyframe 가상 스캔(scenemap sm_snap_scan): 베이스 기준 m, scan_pose = 그때 map 자세. n = 0 이면 없음 */
  double scan_pose[3];
  float scan_origin[2];
  int32_t n_hit;  const float* hit_x; const float* hit_y;
  int32_t n_free; const float* free_x; const float* free_y;
  /* 지난 sgrt_map 뒤 바뀐 칸 경계 상자(이 격자 칸 좌표, 끝 포함). dirty = 0 이면 안 바뀜. 격자 모양(원점·크기)이 바뀌었으면
   * 호출자가 전부 바뀐 것으로 본다. map_version = slam2d 격자 insert 횟수 */
  int32_t dirty;
  int32_t dirty_box[4];
  uint64_t map_version;
  /* 물체 기억에서 옮길 수 있는 물체(움직일 수 있음 → 비용 힌트): map x, y, 반지름(상자 반 대각, m) */
  int32_t n_movable;
  const float* movable_xyr;
} sgrt_map_view;
int    sgrt_map(sgrt*, sgrt_map_view* out);

#ifdef __cplusplus
}
#endif
#endif /* SGRT_H */
