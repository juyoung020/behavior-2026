// C ABI — 파이썬(ctypes)·다른 언어에서 부르는 입구. torch 없음. 실패하면 0 이 아닌 값/NULL 을 돌려주고 ft_last_error() 에 이유.
#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "loader.h"

namespace {
thread_local std::string g_err;

template <class F>
int guard(F&& f) {
    try {
        f();
        return 0;
    } catch (const std::exception& e) {
        g_err = e.what();
        return 1;
    }
}

template <class F>
void* guard_ptr(F&& f) {
    try {
        return f();
    } catch (const std::exception& e) {
        g_err = e.what();
        return nullptr;
    }
}

void ckr(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

std::vector<std::string> strs(int n, const char** p) {
    std::vector<std::string> v;
    for (int i = 0; i < n; ++i) v.emplace_back(p[i]);
    return v;
}
}  // namespace

extern "C" {

const char* ft_last_error(void) { return g_err.c_str(); }

// ---------------------------------------------------------------- 표 (CPU 만)
void* ft_table_open(const char* dir) {
    return guard_ptr([&] { return (void*)new ft::Table(dir); });
}
void ft_table_close(void* t) { delete (ft::Table*)t; }
int64_t ft_table_len(void* t) { return ((ft::Table*)t)->n; }
// out[6] = S, A, M, H, T, cams
int ft_table_dims(void* tp, int32_t* out) {
    auto* t = (ft::Table*)tp;
    out[0] = t->S, out[1] = t->A, out[2] = t->M, out[3] = t->H, out[4] = t->T, out[5] = t->cams;
    return 0;
}
// 샘플 여러 개의 이미지 외 값 + 영상 요청 (검증용). 각 출력은 [n][...] 연속.
int ft_table_samples(void* tp, const int64_t* idx, int64_t n, float* act, float* st, int32_t* tok, uint8_t* tm,
                     int32_t* req) {
    return guard([&] {
        auto* t = (ft::Table*)tp;
        for (int64_t k = 0; k < n; ++k) {
            t->sample(idx[k], act + (size_t)k * t->H * t->M, st + (size_t)k * t->M, tok + (size_t)k * t->T,
                      tm + (size_t)k * t->T);
            if (req) std::memcpy(req + (size_t)k * t->cams * 2, t->req + (size_t)idx[k] * t->cams * 2, t->cams * 8);
        }
    });
}

// ---------------------------------------------------------------- 섞기 순서 (CPU 만)
int ft_sampler_order(int64_t n, int bs, uint64_t seed, int shuffle, int persistent, int64_t nbatches, int64_t* out) {
    return guard([&] {
        ft::Sampler s(n, bs, seed, shuffle != 0, persistent != 0);
        for (int64_t i = 0; i < nbatches; ++i) s.next(out + i * bs);
    });
}

// ---------------------------------------------------------------- 엔진 (검증용 직접 사용)
void* ft_engine_create(int nfiles, const char** videos, const char** idx, const uint8_t* lut, int nsizes,
                       const int32_t* sizes, const int32_t* start, const int32_t* count, const float* weight,
                       const int32_t* srow, const int32_t* scol, int threads, int device) {
    return guard_ptr([&] {
        return (void*)new ft::Engine(strs(nfiles, videos), strs(nfiles, idx), lut, std::vector<int>(sizes, sizes + nsizes),
                                     start, count, weight, srow, scol, threads, device);
    });
}
void ft_engine_destroy(void* e) { delete (ft::Engine*)e; }
int ft_engine_info(void* e, int file, int64_t* out4) {
    return guard([&] {
        auto v = ((ft::Engine*)e)->info(file);
        std::memcpy(out4, v.data(), 4 * 8);
    });
}
int ft_engine_stats(void* e, double* out4) {
    return guard([&] {
        auto v = ((ft::Engine*)e)->stats();
        std::memcpy(out4, v.data(), 4 * 8);
    });
}
// 디코딩 결과를 호스트로 (mode 0: 224² / mode 1: 원 해상도 RGB). out_bytes 는 확인용.
int ft_engine_decode_host(void* ep, const int64_t* req, int n, int mode, uint8_t* out, int64_t out_bytes) {
    return guard([&] {
        auto* e = (ft::Engine*)ep;
        ckr(cudaSetDevice(e->device()), "cudaSetDevice");
        size_t stride = 224 * 224 * 3;
        if (mode == 1) {
            auto inf = e->info((int)req[0]);
            stride = (size_t)inf[0] * inf[1] * 3;
        }
        if ((int64_t)(stride * n) != out_bytes) throw std::runtime_error("출력 크기가 다르다");
        uint8_t* d = nullptr;
        ckr(cudaMalloc((void**)&d, stride * n), "cudaMalloc");
        std::unique_ptr<uint8_t, void (*)(uint8_t*)> hold(d, [](uint8_t* p) { cudaFree(p); });
        e->run(req, n, d, mode);
        ckr(cudaMemcpy(out, d, stride * n, cudaMemcpyDeviceToHost), "cudaMemcpy");
    });
}
int ft_engine_resize_host(void* ep, const uint8_t* in, int n, int W, uint8_t* out) {
    return guard([&] {
        auto* e = (ft::Engine*)ep;
        ckr(cudaSetDevice(e->device()), "cudaSetDevice");
        uint8_t *di = nullptr, *dout = nullptr;
        ckr(cudaMalloc((void**)&di, (size_t)n * W * W * 3), "cudaMalloc");
        std::unique_ptr<uint8_t, void (*)(uint8_t*)> h1(di, [](uint8_t* p) { cudaFree(p); });
        ckr(cudaMalloc((void**)&dout, (size_t)n * 224 * 224 * 3), "cudaMalloc");
        std::unique_ptr<uint8_t, void (*)(uint8_t*)> h2(dout, [](uint8_t* p) { cudaFree(p); });
        ckr(cudaMemcpy(di, in, (size_t)n * W * W * 3, cudaMemcpyHostToDevice), "cudaMemcpy");
        e->resize(di, n, W, dout);
        ckr(cudaMemcpy(out, dout, (size_t)n * 224 * 224 * 3, cudaMemcpyDeviceToHost), "cudaMemcpy");
    });
}

// ---------------------------------------------------------------- 로더
void* ft_loader_create(const char* table_dir, const uint8_t* lut, int nsizes, const int32_t* sizes, const int32_t* start,
                       const int32_t* count, const float* weight, const int32_t* srow, const int32_t* scol, int threads,
                       int device, int batch, int shuffle, uint64_t seed, int persistent, int nslots) {
    return guard_ptr([&] {
        return (void*)new ft::Loader(table_dir, lut, std::vector<int>(sizes, sizes + nsizes), start, count, weight, srow,
                                     scol, threads, device, batch, shuffle != 0, seed, persistent != 0, nslots);
    });
}
void* ft_loader_engine(void* l) { return &((ft::Loader*)l)->engine(); }
// dl_out[10]: DLManagedTensor* (이미지 3, 이미지 마스크 3, 상태, 토큰, 토큰 마스크, 행동). idx_out[batch].
int ft_loader_next(void* lp, int64_t* idx_out, void** dl_out) {
    return guard([&] {
        ft::Slot* s = ((ft::Loader*)lp)->next(idx_out);
        for (int k = 0; k < 10; ++k) dl_out[k] = &s->dl[k];
    });
}
// out[4] = 배치 수, 소비 쪽이 기다린 초, 생산 쪽이 빈 슬롯을 기다린 초, 채우기 초
int ft_loader_stats(void* lp, double* out4) {
    auto s = ((ft::Loader*)lp)->stats();
    out4[0] = s.batches, out4[1] = s.wait_next_s, out4[2] = s.wait_slot_s, out4[3] = s.fill_s;
    return 0;
}
// JAX 가 아직 들고 있는 슬롯이 있으면 지우지 않고 1 을 돌려준다(프로세스 끝까지 남겨 둔다).
int ft_loader_destroy(void* lp) {
    auto* l = (ft::Loader*)lp;
    if (l->busy_slots()) return 1;
    delete l;
    return 0;
}
// 소비 쪽 오류 때 넘겨받지 못한 텐서를 돌려준다.
void ft_dl_release(void* t) {
    auto* d = (DLManagedTensor*)t;
    if (d && d->deleter) d->deleter(d);
}
}
