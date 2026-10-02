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
/* 새 판: 지도·물체 비우고, 프롬프트 표(이 판에서 찾을 물체 이름) 지정 */
int    sgrt_begin(sgrt*, const char* const* prompt, int32_t n, char* err, size_t err_len);
/* 다음 sgrt_step 에 영상을 넣어야 하는가 */
int    sgrt_want_image(const sgrt*);
/* 한 스텝. rgb/depth 는 keyframe 이 아니면 NULL. */
int    sgrt_step(sgrt*, double stamp, const float* proprio, int32_t n_proprio,
                 const uint8_t* rgb, int32_t rgb_on_device, int64_t row_stride, int32_t pix_stride, int32_t w, int32_t h,
                 const float* depth_m, double fx, double fy, double cx, double cy);
int    sgrt_save(sgrt*);    /* 지금 저장 */
/* 통계: keyframe 수, 마지막 검출 수, 확정 물체 수, 마지막 검출 ms, 마지막 저장 ms */
void   sgrt_stats(const sgrt*, int32_t* n_kf, int32_t* n_det, int32_t* n_obj, float* det_ms, float* save_ms);

#ifdef __cplusplus
}
#endif
#endif /* SGRT_H */
