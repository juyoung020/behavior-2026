/* scenemap C ABI (docs/scenemap_설계.md 4절). simlink(Rust)·평가기 프로세스가 같은 프로세스에서 부른다.
 * 지금은 약속한 입력 형식만 있다 — 함수들은 slam2d·objmap 이 붙는 대로 채운다. */
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

#ifdef __cplusplus
}
#endif
#endif /* SCENEMAP_H */
