// 색 변환 + 크기 조정 CUDA 커널.
//
// 크기 조정은 원래 openpi 가 쓰는 openpi_client.image_tools.resize_with_pad (PIL BILINEAR) 를 Pillow 11.2.1
// Resample.c 와 같은 정수 연산으로 옮긴 것이다 (계수·범위 계산은 호스트 pil_resize.h):
//   가로: tmp[r][xx][c] = clip8(2^21 + Σ_x rgb[ybox_first + r][xmin + x][c] × kh[xx][x])     (ImagingResampleHorizontal_8bpc)
//   세로: out[yy][xx][c] = clip8(2^21 + Σ_y tmp[ymin + y][xx][c] × kv[yy][y])                (ImagingResampleVertical_8bpc)
//   clip8(s) = s >> 22 을 [0, 255] 로 자름.  그 뒤 0 으로 채운 target² 의 (pad_x, pad_y) 에 붙인다 (image_tools.py:53-56).
// int32 합 순서도 원래와 같지만, 정수 덧셈이라 순서가 달라도 결과가 같다.
#include "kernels.h"

#include <cuda_runtime.h>

namespace ft {

namespace {
constexpr int PRECISION_BITS = 32 - 8 - 2;

__device__ __forceinline__ uint8_t clip8(int s) {
    int v = s >> PRECISION_BITS;  // 산술 시프트 (Pillow clip8_lookups[in >> PRECISION_BITS])
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

__global__ void k_nv12_lut(const uint8_t* __restrict__ y, const uint8_t* __restrict__ uv, int pitch, int W, int H,
                           const uint8_t* __restrict__ lut, uint8_t* __restrict__ rgb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || r >= H) return;
    uint32_t Y = y[(size_t)r * pitch + x];
    const uint8_t* c = uv + (size_t)(r >> 1) * pitch + (x & ~1);
    uint32_t key = (Y << 16) | ((uint32_t)c[0] << 8) | (uint32_t)c[1];
    const uint8_t* e = lut + (size_t)key * 3;
    uint8_t* o = rgb + ((size_t)r * W + x) * 3;
    o[0] = e[0];
    o[1] = e[1];
    o[2] = e[2];
}

// 가로 단계: 입력 행 ybox_first + r, 출력 열 xx, 채널 c
__global__ void k_pil_h(const uint8_t* __restrict__ rgb, PilDev p, uint8_t* __restrict__ tmp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p.rows * p.out_w * 3;
    if (idx >= total) return;
    int c = idx % 3;
    int xx = (idx / 3) % p.out_w;
    int r = idx / (3 * p.out_w);
    const uint8_t* row = rgb + (size_t)(p.ybox_first + r) * p.in_w * 3;
    int xmin = p.bh[2 * xx], n = p.bh[2 * xx + 1];
    const int* k = p.kkh + (size_t)xx * p.kh;
    int ss = 1 << (PRECISION_BITS - 1);
    for (int x = 0; x < n; ++x) ss += (int)row[(xmin + x) * 3 + c] * k[x];
    tmp[idx] = clip8(ss);
}

// 세로 단계 + 붙이기: 출력 target² 전체 (붙인 영역 밖은 0)
__global__ void k_pil_v(const uint8_t* __restrict__ tmp, PilDev p, uint8_t* __restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p.target * p.target * 3;
    if (idx >= total) return;
    int c = idx % 3;
    int X = (idx / 3) % p.target;
    int Y = idx / (3 * p.target);
    int xx = X - p.pad_x, yy = Y - p.pad_y;
    if (xx < 0 || xx >= p.out_w || yy < 0 || yy >= p.out_h) {
        out[idx] = 0;
        return;
    }
    int ymin = p.bv[2 * yy], n = p.bv[2 * yy + 1];
    const int* k = p.kkv + (size_t)yy * p.kv;
    int ss = 1 << (PRECISION_BITS - 1);
    for (int y = 0; y < n; ++y) ss += (int)tmp[((size_t)(ymin + y) * p.out_w + xx) * 3 + c] * k[y];
    out[idx] = clip8(ss);
}

thread_local cudaError_t g_err = cudaSuccess;
}  // namespace

void launch_nv12_lut(const uint8_t* y, const uint8_t* uv, int pitch, int W, int H, const uint8_t* lut, uint8_t* rgb,
                     void* stream) {
    dim3 b(32, 8), g((W + 31) / 32, (H + 7) / 8);
    k_nv12_lut<<<g, b, 0, (cudaStream_t)stream>>>(y, uv, pitch, W, H, lut, rgb);
    g_err = cudaGetLastError();
}

void launch_resize_pil(const uint8_t* rgb, const PilDev& p, uint8_t* tmp, uint8_t* out, void* stream) {
    int n1 = p.rows * p.out_w * 3;
    k_pil_h<<<(n1 + 255) / 256, 256, 0, (cudaStream_t)stream>>>(rgb, p, tmp);
    int n2 = p.target * p.target * 3;
    k_pil_v<<<(n2 + 255) / 256, 256, 0, (cudaStream_t)stream>>>(tmp, p, out);
    g_err = cudaGetLastError();
}

void* dev_alloc(size_t bytes) {
    void* p = nullptr;
    g_err = cudaMalloc(&p, bytes);
    return g_err == cudaSuccess ? p : nullptr;
}
void dev_free(void* p) { cudaFree(p); }
void dev_sync(void* stream) { g_err = cudaStreamSynchronize((cudaStream_t)stream); }
void set_device(int device) { g_err = cudaSetDevice(device); }
void h2d(void* dst, const void* src, size_t bytes) { g_err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice); }
const char* last_error() { return g_err == cudaSuccess ? nullptr : cudaGetErrorString(g_err); }

}  // namespace ft
