// CUDA 커널 실행 함수 — nvdec.cpp(드라이버 API, ffnvcodec 헤더)와 kernels.cu(런타임 API)를 잇는 C 인터페이스.
// 두 헤더 묶음은 CUcontext 등 같은 이름을 서로 다르게 정의하므로 한 파일에 같이 넣지 않는다.
#pragma once
#include <cstdint>
#include <cstddef>

namespace ft {

// 크기 조정 탭 표 (한 입력 크기·한 방향당): 출력 224 칸마다 첫 입력 위치·탭 수·가중치(최대 8개)·분할 위치.
// split[i] = k 이면 탭 0..k-1 과 k.. 를 따로 순차 FMA 한 뒤 더한다 (cuBLAS split-K 와 같은 순서, 0 = 안 나눔).
struct TapsDev {
    const int* start;
    const int* count;
    const float* weight;  // [224][8]
    const int* split;     // [224]
};

// NV12(NVDEC 출력) → 색 변환 표 → RGB24 (원래 torchcodec 출력 + float 왕복과 같은 값)
void launch_nv12_lut(const uint8_t* y, const uint8_t* uv, int pitch, int W, int H, const uint8_t* lut,
                     uint8_t* rgb, void* stream);
// JAX GPU resize 와 같은 순서의 두 단계 크기 조정 → uint8 224x224x3 (th = 행 방향, tw = 열 방향)
void launch_resize(const uint8_t* rgb, int H, int W, TapsDev th, TapsDev tw, float* tmp, uint8_t* out,
                   void* stream);

void* dev_alloc(size_t bytes);
void dev_free(void* p);
void dev_sync(void* stream);
void set_device(int device);
void h2d(void* dst, const void* src, size_t bytes);
const char* last_error();

}  // namespace ft
