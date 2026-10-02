// NVDEC 직접 호출 디코딩 엔진.
//
// 요청 하나 = (영상 파일, 프레임 번호). 처리:
//   1. 색인(.ftidx)에서 그 프레임 직전 키프레임 k 를 찾는다 (GOP 8 → 최대 8 패킷)
//   2. pread 로 k..목표 패킷만 읽어 길이 접두 NAL → Annex-B 로 바꾼다 (매개변수 집합을 앞에 붙임)
//   3. cuvid 파서 → NVDEC 에 넣고 EOS 로 비운다. B 프레임이 없어(ftprep 가 확인) 마지막으로 표시된 그림이 목표다
//   4. 목표 그림만 매핑 → 색 변환 표 커널 → (mode 0) 크기 조정 커널 → 결과를 출력 텐서 칸에 바로 쓴다
// 스레드마다 자기 CUDA 스트림·NVDEC 디코더(해상도별)를 가져 여러 요청이 동시에 NVDEC 에 들어간다.
// 요청은 작업 큐로 들어오고(submit), 배치 하나의 마지막 작업이 끝나면 Latch 로 알린다.
// 이 파일은 ffnvcodec(dynlink) 헤더만 쓰고 CUDA 툴킷 헤더는 쓰지 않는다 (타입 이름이 겹친다).
#include "engine.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <ffnvcodec/dynlink_loader.h>

#include "kernels.h"
#include "pil_resize.h"

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
    int cls = -1;                     // 해상도 종류 (Impl::plans 에서의 위치)
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
    uint8_t* tmp = nullptr;  // PIL 가로 단계의 8비트 중간 영상
    std::vector<uint8_t> pkt, annexb;
    // 통계
    std::atomic<long long> frames{0}, parsers{0}, requests{0};
    double wait_s = 0;
};

struct Job {
    int64_t file, frame;
    uint8_t* out;
    int mode;
    Latch* latch;
};

}  // namespace

struct Engine::Impl {
    std::vector<VideoFile> files;
    std::vector<std::pair<int, int>> sizes;  // 해상도 (W, H)
    std::vector<PilDev> plans;               // 해상도별 PIL 크기 조정 계획 (GPU)
    std::vector<void*> allocs;
    const uint8_t* lut = nullptr;  // GPU
    CUcontext ctx = nullptr;
    int device = 0;
    std::vector<Worker> workers;

    // 작업 큐: 스레드들이 배치 경계와 상관없이 계속 가져간다 (배치 끝에서 노는 스레드가 없게)
    std::mutex m;
    std::condition_variable cv_job;
    std::deque<Job> q;
    bool stop = false;

    void process(Worker& w, const Job& j);
    void loop(int wi);
};

void Engine::Impl::process(Worker& w, const Job& j) {
    const int64_t fi = j.file, t = j.frame;
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
    if (j.mode == 1) {
        launch_nv12_lut(y, uv, (int)pitch, W, H, lut, j.out, w.stream);
        ck_rt("색 변환 커널");
    } else {
        launch_nv12_lut(y, uv, (int)pitch, W, H, lut, w.rgb, w.stream);
        ck_rt("색 변환 커널");
        launch_resize_pil(w.rgb, plans[v.cls], w.tmp, j.out, w.stream);
        ck_rt("크기 조정 커널");
    }
    // 결과가 GPU 메모리에 다 써진 뒤에야 작업 완료로 센다 (배치 완료 = 모든 이미지 준비 끝)
    CK(cu->cuStreamSynchronize(w.stream));
    unmap.p = 0;
    CK(cv->cuvidUnmapVideoFrame(d.dec, dp));
    w.wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void Engine::Impl::loop(int wi) {
    Worker& w = workers[wi];
    std::string init_err;
    try {
        CK(cu->cuCtxPushCurrent(ctx));
        CK(cu->cuStreamCreate(&w.stream, CU_STREAM_NON_BLOCKING));
    } catch (const std::exception& e) {
        init_err = e.what();
    }
    for (;;) {
        Job j;
        {
            std::unique_lock<std::mutex> g(m);
            cv_job.wait(g, [&] { return stop || !q.empty(); });
            if (q.empty()) break;  // stop 이고 남은 일 없음
            j = q.front();
            q.pop_front();
        }
        try {
            if (!init_err.empty()) throw std::runtime_error(init_err);
            process(w, j);
        } catch (const std::exception& e) {
            j.latch->fail(e.what());
        }
        if (j.latch->left.fetch_sub(1) == 1) j.latch->finish();
    }
    for (Dec& d : w.decs) {
        if (d.parser) cv->cuvidDestroyVideoParser(d.parser);
        if (d.dec) cv->cuvidDestroyDecoder(d.dec);
    }
    if (w.stream) cu->cuStreamDestroy(w.stream);
    CUcontext dummy;
    cu->cuCtxPopCurrent(&dummy);
}

Engine::Engine(const std::vector<std::string>& videos, const std::vector<std::string>& indexes, const uint8_t* lut_host,
               int threads, int device)
    : p_(new Impl) {
    load_libs();
    if (videos.size() != indexes.size()) throw std::runtime_error("videos 와 indexes 길이가 다르다");
    if (threads < 1) throw std::runtime_error("threads >= 1");
    auto& P = *p_;
    P.device = device;
    CK(cu->cuInit(0));
    CUdevice dev;
    CK(cu->cuDeviceGet(&dev, device));
    CK(cu->cuDevicePrimaryCtxRetain(&P.ctx, dev));
    set_device(device);
    ck_rt("cudaSetDevice");
    for (size_t i = 0; i < videos.size(); ++i) {
        P.files.push_back(load_index(videos[i], indexes[i]));
        VideoFile& v = P.files.back();
        for (size_t c = 0; c < P.sizes.size(); ++c)
            if (P.sizes[c] == std::make_pair((int)v.w, (int)v.h)) v.cls = (int)c;
        if (v.cls < 0) {
            v.cls = (int)P.sizes.size();
            P.sizes.emplace_back((int)v.w, (int)v.h);
        }
    }
    size_t max_rgb = 0, max_tmp = 0;
    auto up = [&](const void* src, size_t bytes) {
        void* d = dev_alloc(bytes);
        if (!d) throw std::runtime_error("GPU 메모리 할당 실패");
        h2d(d, src, bytes);
        ck_rt("GPU 로 복사");
        P.allocs.push_back(d);
        return d;
    };
    P.lut = (const uint8_t*)up(lut_host, (size_t)(1u << 24) * 3);
    for (auto& s : P.sizes) {
        PilPlan pl = pil_plan(s.first, s.second, 224);
        PilDev d;
        d.in_w = pl.in_w, d.in_h = pl.in_h, d.out_w = pl.out_w, d.out_h = pl.out_h, d.pad_x = pl.pad_x;
        d.pad_y = pl.pad_y, d.target = pl.target, d.ybox_first = pl.ybox_first, d.rows = pl.rows;
        d.kh = pl.h.ksize, d.kv = pl.v.ksize;
        d.bh = (const int*)up(pl.h.bounds.data(), pl.h.bounds.size() * sizeof(int));
        d.kkh = (const int*)up(pl.h.kk.data(), pl.h.kk.size() * sizeof(int));
        d.bv = (const int*)up(pl.v.bounds.data(), pl.v.bounds.size() * sizeof(int));
        d.kkv = (const int*)up(pl.v.kk.data(), pl.v.kk.size() * sizeof(int));
        P.plans.push_back(d);
        max_rgb = std::max(max_rgb, (size_t)s.first * s.second * 3);
        max_tmp = std::max(max_tmp, (size_t)pl.rows * pl.out_w * 3);
    }
    P.workers = std::vector<Worker>(threads);
    for (auto& w : P.workers) {
        w.decs.resize(std::max<size_t>(P.sizes.size(), 1));
        w.rgb = (uint8_t*)dev_alloc(std::max<size_t>(max_rgb, 1));
        w.tmp = (uint8_t*)dev_alloc(std::max<size_t>(max_tmp, 1));
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

void Engine::submit(const int64_t* req, int n, uint8_t* out, int mode, Latch* latch) {
    auto& P = *p_;
    if (n <= 0) {
        latch->finish();
        return;
    }
    size_t stride = 224 * 224 * 3;
    if (mode == 1) {
        const auto& v0 = P.files.at(req[0]);
        for (int i = 1; i < n; ++i)
            if (P.files.at(req[2 * i]).w != v0.w) throw std::runtime_error("mode 1 은 같은 해상도끼리만");
        stride = (size_t)v0.w * v0.h * 3;
    }
    {
        std::lock_guard<std::mutex> g(P.m);
        for (int i = 0; i < n; ++i) P.q.push_back(Job{req[2 * i], req[2 * i + 1], out + (size_t)i * stride, mode, latch});
    }
    P.cv_job.notify_all();
}

void Engine::run(const int64_t* req, int n, uint8_t* out, int mode) {
    Latch l;
    l.reset(n);
    submit(req, n, out, mode, &l);
    if (n > 0) l.wait();
    if (!l.err.empty()) throw std::runtime_error(l.err);
}

void Engine::resize(const uint8_t* in, int n, int W, int H, uint8_t* out) {
    auto& P = *p_;
    set_device(P.device);
    // 검증용: 이 해상도의 계획을 새로 만든다 (엔진 파일 목록과 무관)
    PilPlan pl = pil_plan(W, H, 224);
    std::vector<void*> tmpa;
    auto up = [&](const void* src, size_t bytes) {
        void* d = dev_alloc(bytes);
        if (!d) throw std::runtime_error("GPU 메모리 할당 실패");
        h2d(d, src, bytes);
        tmpa.push_back(d);
        return d;
    };
    PilDev d;
    d.in_w = pl.in_w, d.in_h = pl.in_h, d.out_w = pl.out_w, d.out_h = pl.out_h, d.pad_x = pl.pad_x;
    d.pad_y = pl.pad_y, d.target = pl.target, d.ybox_first = pl.ybox_first, d.rows = pl.rows;
    d.kh = pl.h.ksize, d.kv = pl.v.ksize;
    d.bh = (const int*)up(pl.h.bounds.data(), pl.h.bounds.size() * sizeof(int));
    d.kkh = (const int*)up(pl.h.kk.data(), pl.h.kk.size() * sizeof(int));
    d.bv = (const int*)up(pl.v.bounds.data(), pl.v.bounds.size() * sizeof(int));
    d.kkv = (const int*)up(pl.v.kk.data(), pl.v.kk.size() * sizeof(int));
    uint8_t* tmp = (uint8_t*)dev_alloc((size_t)pl.rows * pl.out_w * 3);
    if (!tmp) throw std::runtime_error("GPU 메모리 할당 실패");
    tmpa.push_back(tmp);
    for (int i = 0; i < n; ++i)
        launch_resize_pil(in + (size_t)i * W * H * 3, d, tmp, out + (size_t)i * 224 * 224 * 3, nullptr);
    dev_sync(nullptr);
    const char* e = last_error();
    for (void* a : tmpa) dev_free(a);
    if (e) throw std::runtime_error(std::string("크기 조정 커널: ") + e);
}

int Engine::device() const { return p_->device; }

std::vector<int64_t> Engine::info(int f) const {
    const auto& v = p_->files.at(f);
    return {v.w, v.h, v.timescale, (int64_t)v.s.size()};
}

std::vector<double> Engine::stats() const {
    double r = 0, fr = 0, pa = 0, ws = 0;
    for (auto& w : p_->workers) {
        r += w.requests.load();
        fr += w.frames.load();
        pa += w.parsers.load();
        ws += w.wait_s;
    }
    return {r, fr, pa, ws};
}

}  // namespace ft
