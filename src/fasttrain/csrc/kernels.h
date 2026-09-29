// CUDA 커널 실행 함수 — nvdec.cpp(드라이버 API, ffnvcodec 헤더)와 kernels.cu(런타임 API)를 잇는 C 인터페이스.
// 두 헤더 묶음은 CUcontext 등 같은 이름을 서로 다르게 정의하므로 한 파일에 같이 넣지 않는다.
#pragma once
#include <cstddef>
#include <cstdint>

namespace ft {

// PIL 크기 조정 계획의 GPU 사본 (pil_resize.h PilPlan). 계수·범위는 GPU 포인터.
struct PilDev {
    int in_w, in_h, out_w, out_h, pad_x, pad_y, target, ybox_first, rows;
    int kh, kv;      // 가로·세로 ksize
    const int* bh;   // 가로 범위 [out_w][2]
    const int* kkh;  // 가로 계수 [out_w][kh]
    const int* bv;   // 세로 범위 [out_h][2] (ybox_first 만큼 당겨짐)
    const int* kkv;  // 세로 계수 [out_h][kv]
};

// NV12(NVDEC 출력) → 색 변환 표 → RGB24 (원래 torchcodec 출력 + float 왕복과 같은 값)
void launch_nv12_lut(const uint8_t* y, const uint8_t* uv, int pitch, int W, int H, const uint8_t* lut,
                     uint8_t* rgb, void* stream);
// PIL BILINEAR 두 단계 (가로 → 8비트 중간 → 세로) + 0 채운 target² 에 붙이기. tmp 는 rows × out_w × 3 바이트.
void launch_resize_pil(const uint8_t* rgb, const PilDev& p, uint8_t* tmp, uint8_t* out, void* stream);

void* dev_alloc(size_t bytes);
void dev_free(void* p);
void dev_sync(void* stream);
void set_device(int device);
void h2d(void* dst, const void* src, size_t bytes);
const char* last_error();

}  // namespace ft
