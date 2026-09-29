// NVDEC 직접 호출 디코딩 엔진.
//
// 요청 하나 = (영상 파일, 프레임 번호). 처리:
//   1. 색인(.ftidx)에서 그 프레임 직전 키프레임 k 를 찾는다 (GOP 8 → 최대 8 패킷)
//   2. pread 로 k..목표 패킷만 읽어 길이 접두 NAL → Annex-B 로 바꾼다 (매개변수 집합을 앞에 붙임)
//   3. cuvid 파서 → NVDEC 에 넣고 EOS 로 비운다. B 프레임이 없어(ftprep 가 확인) 마지막으로 표시된 그림이 목표다
//   4. 목표 그림만 매핑 → 색 변환 표 커널 → (mode 0) 크기 조정 커널 → 결과를 출력 텐서 칸에 바로 쓴다
// 스레드마다 자기 CUDA 스트림·NVDEC 디코더(해상도별)를 가져 여러 요청이 동시에 NVDEC 에 들어간다.
// 이 파일은 ffnvcodec(dynlink) 헤더만 쓰고 CUDA 툴킷 헤더는 쓰지 않는다 (타입 이름이 겹친다).
#include "engine.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <ffnvcodec/dynlink_loader.h>

#include "kernels.h"

namespace ft {
namespace {

CudaFunctions* cu = nullptr;
CuvidFunctions* cv = nullptr;
std::once_flag g_load;

void load_libs() {
    std::call_once(g_load, [] {
        if (cuda_load_functions(&cu, nullptr) < 0) throw std::runtime_error("libcuda.so.1 을 못 읽었다");
        if (cuvid_load_functions(&cv, nullptr) < 0) throw std::runtime_error("libnvcuvid.so.1 을 못 읽었다");
    });
}

void ck(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return;
    const char* name = nullptr;
    if (cu && cu->cuGetErrorName) cu->cuGetErrorName(r, &name);
    throw std::runtime_error(std::string(what) + " 실패: " + (name ? name : std::to_string((int)r)));
}
#define CK(x) ck((x), #x)

void ck_rt(const char* what) {
    if (const char* e = last_error()) throw std::runtime_error(std::string(what) + ": " + e);
}

struct Sample {
    uint64_t off;
    uint32_t size;
    uint32_t key;
    int64_t pts;
};

struct VideoFile {
    int fd = -1;
    uint32_t w = 0, h = 0, timescale = 0, nal_len = 4;
    std::vector<uint8_t> params;
    std::vector<Sample> s;
    std::vector<int32_t> key_before;  // 각 표본의 직전(자기 포함) 키프레임
    int cls = -1;                     // 해상도 종류 (sizes 에서의 위치)
};

template <class T>
T rd(const std::vector<char>& b, size_t& p) {
    T v;
    std::memcpy(&v, b.data() + p, sizeof(T));
    p += sizeof(T);
    return v;
}

VideoFile load_index(const std::string& video, const std::string& idx) {
    std::ifstream f(idx, std::ios::binary);
    if (!f) throw std::runtime_error("색인을 못 연다: " + idx);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (b.size() < 40 || std::memcmp(b.data(), "FTIDX1\0\0", 8) != 0) throw std::runtime_error("색인 형식이 아니다: " + idx);
    size_t p = 8;
    VideoFile v;
    v.w = rd<uint32_t>(b, p);
    v.h = rd<uint32_t>(b, p);
    v.timescale = rd<uint32_t>(b, p);
    v.nal_len = rd<uint32_t>(b, p);
    uint32_t plen = rd<uint32_t>(b, p);
    uint32_t monotonic = rd<uint32_t>(b, p);
    if (!monotonic) throw std::runtime_error("표시 순서 != 디코딩 순서 (B 프레임) 는 지원 안 함: " + idx);
    v.params.assign(b.begin() + p, b.begin() + p + plen);
    p += plen;
    uint64_t n = rd<uint64_t>(b, p);
    if (b.size() < p + n * 24) throw std::runtime_error("색인이 잘렸다: " + idx);
    v.s.resize(n);
    v.key_before.resize(n);
    int32_t last_key = -1;
    for (uint64_t i = 0; i < n; ++i) {
        v.s[i].off = rd<uint64_t>(b, p);
        v.s[i].size = rd<uint32_t>(b, p);
        v.s[i].key = rd<uint32_t>(b, p);
        v.s[i].pts = rd<int64_t>(b, p);
        if (v.s[i].key) last_key = (int32_t)i;
        v.key_before[i] = last_key;
    }
    v.fd = ::open(video.c_str(), O_RDONLY);
    if (v.fd < 0) throw std::runtime_error("영상을 못 연다: " + video);
    return v;
}

struct Worker;

struct Dec {  // 해상도 하나의 NVDEC 디코더 + 요청마다 새로 만드는 파서
    CUvideoparser parser = nullptr;
    CUvideodecoder dec = nullptr;
    unsigned w = 0, h = 0, nsurf = 0;
    std::vector<std::pair<int, long long>> shown;
    std::string err;
};

int CUDAAPI on_sequence(void* ud, CUVIDEOFORMAT* f) {
    Dec* d = (Dec*)ud;
    unsigned want = (f->min_num_decode_surfaces ? f->min_num_decode_surfaces : 8) + 4;
    unsigned dw = f->display_area.right - f->display_area.left;
    unsigned dh = f->display_area.bottom - f->display_area.top;
    if (d->dec && (d->w != dw || d->h != dh || d->nsurf < want)) {
        cv->cuvidDestroyDecoder(d->dec);
        d->dec = nullptr;
    }
    if (!d->dec) {
        CUVIDDECODECREATEINFO ci;
        std::memset(&ci, 0, sizeof(ci));
        ci.CodecType = f->codec;
        ci.ChromaFormat = f->chroma_format;
        ci.OutputFormat = cudaVideoSurfaceFormat_NV12;
        ci.bitDepthMinus8 = f->bit_depth_luma_minus8;
        ci.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
        ci.ulNumOutputSurfaces = 2;
        ci.ulCreationFlags = cudaVideoCreate_PreferCUVID;
        ci.ulNumDecodeSurfaces = want;
        ci.ulWidth = f->coded_width;
        ci.ulHeight = f->coded_height;
        ci.ulMaxWidth = f->coded_width;
        ci.ulMaxHeight = f->coded_height;
        ci.display_area.left = (short)f->display_area.left;
        ci.display_area.top = (short)f->display_area.top;
        ci.display_area.right = (short)f->display_area.right;
        ci.display_area.bottom = (short)f->display_area.bottom;
        ci.ulTargetWidth = dw;
        ci.ulTargetHeight = dh;
        CUresult r = cv->cuvidCreateDecoder(&d->dec, &ci);
        if (r != CUDA_SUCCESS) {
            d->err = "cuvidCreateDecoder 실패 " + std::to_string((int)r);
            return 0;
        }
        d->w = dw;
        d->h = dh;
        d->nsurf = want;
    }
    return (int)d->nsurf;
}

int CUDAAPI on_decode(void* ud, CUVIDPICPARAMS* pp) {
    Dec* d = (Dec*)ud;
    if (!d->dec) return 0;
    CUresult r = cv->cuvidDecodePicture(d->dec, pp);
    if (r != CUDA_SUCCESS) {
        d->err = "cuvidDecodePicture 실패 " + std::to_string((int)r);
        return 0;
    }
    return 1;
}

int CUDAAPI on_display(void* ud, CUVIDPARSERDISPINFO* di) {
    Dec* d = (Dec*)ud;
    if (di) d->shown.emplace_back(di->picture_index, (long long)di->timestamp);
    return 1;
}

struct Worker {
    std::thread th;
    CUstream stream = nullptr;
    std::vector<Dec> decs;
    uint8_t* rgb = nullptr;
    float* tmp = nullptr;
    std::vector<uint8_t> pkt, annexb;
    // 통계
    long long frames = 0, parsers = 0, requests = 0;
    double wait_s = 0;
};

}  // namespace

struct Engine::Impl {
    std::vector<VideoFile> files;
    std::vector<int> sizes;
    std::vector<TapsDev> trow, tcol;  // 해상도별 행·열 방향
    std::vector<void*> allocs;
    const uint8_t* lut = nullptr;
    CUcontext ctx = nullptr;
    int device = 0;
    std::vector<Worker> workers;

    // 작업 나눠 주기
    std::mutex m;
    std::condition_variable cv_job, cv_done;
    long long gen = 0;
    bool stop = false;
    int active = 0;
    std::atomic<int> next{0};
    const int64_t* req = nullptr;
    int n = 0;
    uint8_t* out = nullptr;
    int mode = 0;
    std::exception_ptr err;

    void process(Worker& w, int i);
    void loop(int wi);
};

void Engine::Impl::process(Worker& w, int i) {
    int64_t fi = req[2 * i], t = req[2 * i + 1];
    if (fi < 0 || fi >= (int64_t)files.size()) throw std::runtime_error("파일 번호 범위 밖: " + std::to_string(fi));
    VideoFile& v = files[fi];
    if (t < 0 || t >= (int64_t)v.s.size()) throw std::runtime_error("프레임 번호 범위 밖: " + std::to_string(t));
    int k = v.key_before[t];
    if (k < 0) throw std::runtime_error("앞에 키프레임이 없다");
    Dec& d = w.decs[v.cls];
    if (d.parser) cv->cuvidDestroyVideoParser(d.parser);
    d.parser = nullptr;
    CUVIDPARSERPARAMS pp;
    std::memset(&pp, 0, sizeof(pp));
    pp.CodecType = cudaVideoCodec_HEVC;
    pp.ulMaxNumDecodeSurfaces = d.nsurf ? d.nsurf : 1;
    pp.ulMaxDisplayDelay = 0;
    pp.pUserData = &d;
    pp.pfnSequenceCallback = on_sequence;
    pp.pfnDecodePicture = on_decode;
    pp.pfnDisplayPicture = on_display;
    CK(cv->cuvidCreateVideoParser(&d.parser, &pp));
    w.parsers++;
    d.shown.clear();
    d.err.clear();
    for (int s = k; s <= t; ++s) {
        const Sample& sm = v.s[s];
        w.pkt.resize(sm.size);
        size_t got = 0;
        while (got < sm.size) {
            ssize_t r = ::pread(v.fd, w.pkt.data() + got, sm.size - got, (off_t)(sm.off + got));
            if (r <= 0) throw std::runtime_error("pread 실패");
            got += (size_t)r;
        }
        w.annexb.clear();
        if (s == k) w.annexb.insert(w.annexb.end(), v.params.begin(), v.params.end());
        size_t p = 0;
        const unsigned nl = v.nal_len;
        while (p + nl <= sm.size) {
            uint32_t len = 0;
            for (unsigned b = 0; b < nl; ++b) len = (len << 8) | w.pkt[p + b];
            p += nl;
            if (p + len > sm.size) throw std::runtime_error("NAL 길이가 표본을 넘는다");
            static const uint8_t sc[4] = {0, 0, 0, 1};
            w.annexb.insert(w.annexb.end(), sc, sc + 4);
            w.annexb.insert(w.annexb.end(), w.pkt.begin() + p, w.pkt.begin() + p + len);
            p += len;
        }
        CUVIDSOURCEDATAPACKET pk;
        std::memset(&pk, 0, sizeof(pk));
        pk.payload = w.annexb.data();
        pk.payload_size = (unsigned long)w.annexb.size();
        pk.flags = CUVID_PKT_TIMESTAMP;
        pk.timestamp = s;
        CK(cv->cuvidParseVideoData(d.parser, &pk));
        if (!d.err.empty()) throw std::runtime_error(d.err);
    }
    CUVIDSOURCEDATAPACKET eos;
    std::memset(&eos, 0, sizeof(eos));
    eos.flags = CUVID_PKT_ENDOFSTREAM;
    CK(cv->cuvidParseVideoData(d.parser, &eos));
    if (!d.err.empty()) throw std::runtime_error(d.err);
    w.frames += t - k + 1;
    w.requests++;
    if ((int)d.shown.size() != t - k + 1)
        throw std::runtime_error("표시된 그림 수 " + std::to_string(d.shown.size()) + " != 넣은 패킷 수 " +
                                 std::to_string(t - k + 1));
    if (d.shown.back().second != t)
        throw std::runtime_error("마지막 그림의 시각표 " + std::to_string(d.shown.back().second) + " != 목표 " +
                                 std::to_string(t));
    int pic = d.shown.back().first;

    auto t0 = std::chrono::steady_clock::now();
    CUVIDPROCPARAMS vpp;
    std::memset(&vpp, 0, sizeof(vpp));
    vpp.progressive_frame = 1;
    vpp.output_stream = w.stream;
    CUdeviceptr dp = 0;
    unsigned int pitch = 0;
    CK(cv->cuvidMapVideoFrame(d.dec, pic, &dp, &pitch, &vpp));
    struct Unmap {  // 커널·동기화 중 예외가 나도 매핑을 풀어 둔다
        CUvideodecoder dec;
        CUdeviceptr p;
        ~Unmap() {
            if (p) cv->cuvidUnmapVideoFrame(dec, p);
        }
    } unmap{d.dec, dp};
    const uint8_t* y = (const uint8_t*)dp;
    const uint8_t* uv = y + (size_t)pitch * ((d.h + 1) & ~1u);
    const int W = (int)d.w, H = (int)d.h;
    if (mode == 1) {
        launch_nv12_lut(y, uv, (int)pitch, W, H, lut, out + (size_t)i * W * H * 3, w.stream);
        ck_rt("색 변환 커널");
    } else {
        launch_nv12_lut(y, uv, (int)pitch, W, H, lut, w.rgb, w.stream);
        ck_rt("색 변환 커널");
        launch_resize(w.rgb, H, W, trow[v.cls], tcol[v.cls], w.tmp, out + (size_t)i * 224 * 224 * 3, w.stream);
        ck_rt("크기 조정 커널");
    }
    CK(cu->cuStreamSynchronize(w.stream));
    unmap.p = 0;
    CK(cv->cuvidUnmapVideoFrame(d.dec, dp));
    w.wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void Engine::Impl::loop(int wi) {
    Worker& w = workers[wi];
    try {
        CK(cu->cuCtxPushCurrent(ctx));
        CK(cu->cuStreamCreate(&w.stream, CU_STREAM_NON_BLOCKING));
    } catch (...) {
        std::lock_guard<std::mutex> g(m);
        if (!err) err = std::current_exception();
    }
    long long seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> g(m);
            cv_job.wait(g, [&] { return stop || gen != seen; });
            if (stop) break;
            seen = gen;
        }
        for (;;) {
            int i = next.fetch_add(1);
            if (i >= n) break;
            try {
                process(w, i);
            } catch (...) {
                std::lock_guard<std::mutex> g(m);
                if (!err) err = std::current_exception();
                next.store(n);
            }
        }
        {
            std::lock_guard<std::mutex> g(m);
            if (--active == 0) cv_done.notify_all();
        }
    }
    for (Dec& d : w.decs) {
        if (d.parser) cv->cuvidDestroyVideoParser(d.parser);
        if (d.dec) cv->cuvidDestroyDecoder(d.dec);
    }
    if (w.stream) cu->cuStreamDestroy(w.stream);
    CUcontext dummy;
    cu->cuCtxPopCurrent(&dummy);
}

Engine::Engine(const std::vector<std::string>& videos, const std::vector<std::string>& indexes, uintptr_t lut_dev,
               const std::vector<int>& sizes, const int32_t* start, const int32_t* count, const float* weight,
               const int32_t* split_rows, const int32_t* split_cols, int threads, int device)
    : p_(new Impl) {
    load_libs();
    if (videos.size() != indexes.size()) throw std::runtime_error("videos 와 indexes 길이가 다르다");
    auto& P = *p_;
    P.lut = (const uint8_t*)lut_dev;
    P.sizes = sizes;
    P.device = device;
    CK(cu->cuInit(0));
    CUdevice dev;
    CK(cu->cuDeviceGet(&dev, device));
    CK(cu->cuDevicePrimaryCtxRetain(&P.ctx, dev));
    for (size_t i = 0; i < videos.size(); ++i) {
        P.files.push_back(load_index(videos[i], indexes[i]));
        VideoFile& v = P.files.back();
        if (v.w != v.h) throw std::runtime_error("정사각 영상만: " + videos[i]);
        for (size_t c = 0; c < sizes.size(); ++c)
            if ((int)v.w == sizes[c]) v.cls = (int)c;
        if (v.cls < 0) throw std::runtime_error("크기 조정 탭이 없는 해상도: " + std::to_string(v.w));
    }
    int maxw = 0;
    for (int s : sizes) maxw = std::max(maxw, s);
    auto up = [&](const void* src, size_t bytes) {
        void* d = dev_alloc(bytes);
        if (!d) throw std::runtime_error("GPU 메모리 할당 실패");
        h2d(d, src, bytes);
        ck_rt("탭 복사");
        P.allocs.push_back(d);
        return d;
    };
    for (size_t c = 0; c < sizes.size(); ++c) {
        TapsDev t;
        t.start = (const int*)up(start + c * 224, 224 * sizeof(int));
        t.count = (const int*)up(count + c * 224, 224 * sizeof(int));
        t.weight = (const float*)up(weight + c * 224 * 8, 224 * 8 * sizeof(float));
        t.split = (const int*)up(split_rows + c * 224, 224 * sizeof(int));
        P.trow.push_back(t);
        t.split = (const int*)up(split_cols + c * 224, 224 * sizeof(int));
        P.tcol.push_back(t);
    }
    P.workers.resize(threads);
    for (auto& w : P.workers) {
        w.decs.resize(sizes.size());
        w.rgb = (uint8_t*)dev_alloc((size_t)maxw * maxw * 3);
        w.tmp = (float*)dev_alloc((size_t)224 * maxw * 3 * sizeof(float));
        if (!w.rgb || !w.tmp) throw std::runtime_error("GPU 메모리 할당 실패");
        P.allocs.push_back(w.rgb);
        P.allocs.push_back(w.tmp);
    }
    for (int i = 0; i < threads; ++i) P.workers[i].th = std::thread([this, i] { p_->loop(i); });
}

Engine::~Engine() {
    auto& P = *p_;
    {
        std::lock_guard<std::mutex> g(P.m);
        P.stop = true;
    }
    P.cv_job.notify_all();
    for (auto& w : P.workers)
        if (w.th.joinable()) w.th.join();
    for (void* a : P.allocs) dev_free(a);
    for (auto& v : P.files)
        if (v.fd >= 0) ::close(v.fd);
    if (cu && P.ctx) {
        CUdevice dev;
        if (cu->cuDeviceGet(&dev, P.device) == CUDA_SUCCESS) cu->cuDevicePrimaryCtxRelease(dev);
    }
}

void Engine::run(const int64_t* req, int n, uintptr_t out, int mode) {
    auto& P = *p_;
    if (mode == 1) {
        for (int i = 1; i < n; ++i)
            if (P.files[req[2 * i]].w != P.files[req[0]].w) throw std::runtime_error("mode 1 은 같은 해상도끼리만");
    }
    std::unique_lock<std::mutex> g(P.m);
    if (P.err) {
        auto e = P.err;
        P.err = nullptr;
        std::rethrow_exception(e);
    }
    P.req = req;
    P.n = n;
    P.out = (uint8_t*)out;
    P.mode = mode;
    P.next.store(0);
    P.active = (int)P.workers.size();
    P.gen++;
    P.cv_job.notify_all();
    P.cv_done.wait(g, [&] { return P.active == 0; });
    if (P.err) {
        auto e = P.err;
        P.err = nullptr;
        std::rethrow_exception(e);
    }
}

void Engine::resize(uintptr_t in, int n, int W, uintptr_t out) {
    auto& P = *p_;
    int cls = -1;
    for (size_t c = 0; c < P.sizes.size(); ++c)
        if (P.sizes[c] == W) cls = (int)c;
    if (cls < 0) throw std::runtime_error("탭이 없는 해상도");
    float* tmp = (float*)dev_alloc((size_t)224 * W * 3 * sizeof(float));
    if (!tmp) throw std::runtime_error("GPU 메모리 할당 실패");
    for (int i = 0; i < n; ++i)
        launch_resize((const uint8_t*)in + (size_t)i * W * W * 3, W, W, P.trow[cls], P.tcol[cls], tmp,
                      (uint8_t*)out + (size_t)i * 224 * 224 * 3, nullptr);
    dev_sync(nullptr);
    const char* e = last_error();
    dev_free(tmp);
    if (e) throw std::runtime_error(std::string("크기 조정 커널: ") + e);
}

std::vector<int64_t> Engine::info(int f) const {
    const auto& v = p_->files.at(f);
    return {v.w, v.h, v.timescale, (int64_t)v.s.size()};
}

std::vector<int64_t> Engine::pts(int f) const {
    const auto& v = p_->files.at(f);
    std::vector<int64_t> o(v.s.size());
    for (size_t i = 0; i < v.s.size(); ++i) o[i] = v.s[i].pts;
    return o;
}

std::vector<double> Engine::stats() const {
    double r = 0, fr = 0, pa = 0, ws = 0;
    for (auto& w : p_->workers) {
        r += w.requests;
        fr += w.frames;
        pa += w.parsers;
        ws += w.wait_s;
    }
    return {r, fr, pa, ws};
}

}  // namespace ft
