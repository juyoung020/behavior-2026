// 색 변환 + 크기 조정 CUDA 커널.
//
// 크기 조정은 원래 openpi image_tools.resize_with_pad(jax.image.resize LINEAR, antialias) 의 **GPU 컴파일 결과와 같은
// 계산 순서**로 한다. XLA 가 만든 프로그램(~/fasttrain_work/hlo_resize_720.txt, tools/ft_verify.py 가 다시 뽑는다):
//   1) 가중치 W[입력, 224] (jax 가 GPU 에서 계산한 값을 그대로 받아 쓴다)
//   2) cuBLAS SGEMM #1: T[i, (w,c)] = Σ_h W[h,i] · X[h,(w,c)]      (행 방향 먼저)
//   3) cuBLAS SGEMM #2: O[(i,c), j] = Σ_w T[i,(w,c)] · W[w,j]      (열 방향)
//   4) round-half-even → clamp[0,255] → uint8
// SGEMM 은 각 출력 원소를 k 오름차순 FMA 로 누적하되, cuBLAS 가 split-K 를 고르면 k 구간마다 따로 누적해 더한다.
// 어디서 나누는지는 GPU·cuBLAS 에 따라 달라서 calib.py 가 시작 때 원래 함수로 알아내 split 표로 넘긴다.
// 0 인 가중치는 fma(x,0,acc)=acc 라 건너뛰어도 비트가 같다 (docs/학습환경_가속.md 3절).
#include "kernels.h"

#include <cuda_runtime.h>

namespace ft {

namespace {
constexpr int OUT = 224;
constexpr int MAXTAP = 8;

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

// T[i][n] = Σ (h 오름차순, split 에서 두 구간) W[h][i] * X[h][n],  n = w*3 + c
__global__ void k_resize_rows(const uint8_t* __restrict__ rgb, int W, TapsDev t, float* __restrict__ T) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    int WC = W * 3;
    if (n >= WC) return;
    int s = t.start[i], cnt = t.count[i], sp = t.split[i];
    float p = 0.f, q = 0.f;
    for (int k = 0; k < cnt; ++k) {
        float x = (float)rgb[(size_t)(s + k) * WC + n];
        float w = t.weight[i * MAXTAP + k];
        if (sp && k >= sp)
            q = __fmaf_rn(w, x, q);
        else
            p = __fmaf_rn(w, x, p);
    }
    T[(size_t)i * WC + n] = sp ? __fadd_rn(p, q) : p;
}

// O[i][j][c] = round(Σ (w 오름차순, split 에서 두 구간) T[i][w*3+c] * W[w][j])
__global__ void k_resize_cols(const float* __restrict__ T, int W, TapsDev t, uint8_t* __restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= OUT * OUT * 3) return;
    int c = idx % 3;
    int j = (idx / 3) % OUT;
    int i = idx / (3 * OUT);
    const float* row = T + (size_t)i * W * 3;
    int s = t.start[j], cnt = t.count[j], sp = t.split[j];
    float p = 0.f, q = 0.f;
    for (int k = 0; k < cnt; ++k) {
        float x = row[(s + k) * 3 + c];
        float w = t.weight[j * MAXTAP + k];
        if (sp && k >= sp)
            q = __fmaf_rn(x, w, q);
        else
            p = __fmaf_rn(x, w, p);
    }
    float acc = sp ? __fadd_rn(p, q) : p;
    float r = rintf(acc);  // round-half-even (XLA round-nearest-even)
    r = fminf(fmaxf(r, 0.f), 255.f);
    out[idx] = (uint8_t)r;
}

thread_local cudaError_t g_err = cudaSuccess;
}  // namespace

void launch_nv12_lut(const uint8_t* y, const uint8_t* uv, int pitch, int W, int H, const uint8_t* lut, uint8_t* rgb,
                     void* stream) {
    dim3 b(32, 8), g((W + 31) / 32, (H + 7) / 8);
    k_nv12_lut<<<g, b, 0, (cudaStream_t)stream>>>(y, uv, pitch, W, H, lut, rgb);
    g_err = cudaGetLastError();
}

void launch_resize(const uint8_t* rgb, int H, int W, TapsDev th, TapsDev tw, float* tmp, uint8_t* out, void* stream) {
    (void)H;
    int WC = W * 3;
    dim3 b1(256), g1((WC + 255) / 256, OUT);
    k_resize_rows<<<g1, b1, 0, (cudaStream_t)stream>>>(rgb, W, th, tmp);
    int n = OUT * OUT * 3;
    k_resize_cols<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(tmp, W, tw, out);
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
