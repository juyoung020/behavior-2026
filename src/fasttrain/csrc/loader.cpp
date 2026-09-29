// 네이티브 로더 구현. 설명은 loader.h.
#include "loader.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ft {

namespace {
void ckr(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
size_t align256(size_t x) { return (x + 255) & ~size_t(255); }

void dl_deleter(DLManagedTensor* t) {
    Slot* s = (Slot*)t->manager_ctx;
    s->owner->release(s);
}
}  // namespace

// ------------------------------------------------------------------ Table
const void* Table::map(const std::string& path, size_t expect) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("표 파일을 못 연다: " + path);
    struct stat st;
    fstat(fd, &st);
    if ((size_t)st.st_size != expect) {
        ::close(fd);
        throw std::runtime_error(path + ": 크기 " + std::to_string(st.st_size) + " != 기대 " + std::to_string(expect));
    }
    void* p = expect ? mmap(nullptr, expect, PROT_READ, MAP_SHARED, fd, 0) : nullptr;
    ::close(fd);
    if (expect && p == MAP_FAILED) throw std::runtime_error("mmap 실패: " + path);
    maps_.emplace_back(p, expect);
    return p;
}

Table::Table(const std::string& dir) {
    std::ifstream f(dir + "/table.txt");
    if (!f) throw std::runtime_error("표 머리를 못 연다: " + dir + "/table.txt");
    std::string line;
    int nfiles = -1;
    while (std::getline(f, line)) {
        if (line.rfind("file\t", 0) == 0) {
            auto a = line.find('\t'), b = line.find('\t', a + 1);
            files.push_back(line.substr(a + 1, b - a - 1));
            indexes.push_back(line.substr(b + 1));
            continue;
        }
        std::istringstream is(line);
        std::string k;
        is >> k;
        if (k == "ftable") {
            int v;
            is >> v;
            if (v != 1) throw std::runtime_error("표 판 번호가 다르다");
        } else if (k == "n") is >> n;
        else if (k == "state_dim") is >> S;
        else if (k == "action_dim") is >> A;
        else if (k == "model_dim") is >> M;
        else if (k == "horizon") is >> H;
        else if (k == "max_token_len") is >> T;
        else if (k == "cams") is >> cams;
        else if (k == "files") is >> nfiles;
        else if (k == "delta") {
            int nd;
            is >> nd;
            for (int i = 0; i < nd; ++i) {
                int a, s;
                is >> a >> s;
                delta.emplace_back(a, s);
            }
        }
    }
    if (n <= 0 || S <= 0 || A <= 0 || M < A || M < S || H <= 0 || T <= 0 || cams <= 0 || nfiles != (int)files.size())
        throw std::runtime_error("표 머리가 이상하다: " + dir);
    if (A > 64) throw std::runtime_error("행동 차원 64 초과는 지원 안 함");
    for (auto& d : delta)
        if (d.first < 0 || d.first >= A || d.second < 0 || d.second >= S) throw std::runtime_error("델타 대응 범위 밖");
    const size_t N = (size_t)n;
    abs = (const int64_t*)map(dir + "/abs_index.i64", N * 8);
    task = (const int32_t*)map(dir + "/task.i32", N * 4);
    ep_last = (const int32_t*)map(dir + "/ep_last.i32", N * 4);
    state_ex = (const float*)map(dir + "/state_ex.f32", N * S * 4);
    action = (const float*)map(dir + "/action.f32", N * A * 4);
    state_out = (const float*)map(dir + "/state_out.f32", N * M * 4);
    tok_off = (const uint64_t*)map(dir + "/tok_off.u64", (N + 1) * 8);
    tok_ids = (const int32_t*)map(dir + "/tok_ids.i32", tok_off[N] * 4);
    req = (const int32_t*)map(dir + "/req.i32", N * cams * 2 * 4);
    norm = (const double*)map(dir + "/norm_action.f64", (size_t)2 * A * 8);
    for (size_t i = 0; i < N; ++i)
        if (ep_last[i] < (int64_t)i || ep_last[i] >= n) throw std::runtime_error("ep_last 가 이상하다");
}

Table::~Table() {
    for (auto& m : maps_)
        if (m.first) munmap(m.first, m.second);
}

void Table::sample(int64_t i, float* actions, float* state, int32_t* tokens, uint8_t* tmask) const {
    if (i < 0 || i >= n) throw std::runtime_error("샘플 번호 범위 밖: " + std::to_string(i));
    const double* mean = norm;
    const double* std_ = norm + A;
    const int64_t last = ep_last[i];
    const float* s = state_ex + (size_t)i * S;
    float tmp[64];
    for (int k = 0; k < H; ++k) {
        // lerobot: 창 번호 = min(에피소드 끝 - 1, 현재 + k) (dataset_reader.py:195)
        const int64_t r = std::min<int64_t>(i + k, last);
        const float* a = action + (size_t)r * A;
        for (int d = 0; d < A; ++d) tmp[d] = a[d];
        // MappedDeltaActions: float32 행동 - float32 상태 (transforms.py:265)
        for (auto& p : delta) tmp[p.first] = a[p.first] - s[p.second];
        // Normalize: (x - mean) / (std + 1e-6), mean·std float64 → float64 결과 (transforms.py:137-139)
        // → 원래는 float64 로 넘어가 JAX(x64 꺼짐)가 float32 로 바꾼다 = 가장 가까운 float32
        float* o = actions + (size_t)k * M;
        for (int d = 0; d < A; ++d) o[d] = (float)(((double)tmp[d] - mean[d]) / (std_[d] + 1e-6));
        for (int d = A; d < M; ++d) o[d] = 0.f;  // PadStatesAndActions
    }
    std::memcpy(state, state_out + (size_t)i * M, (size_t)M * 4);
    const uint64_t off = tok_off[i];
    const int len = (int)(tok_off[i + 1] - off);
    for (int j = 0; j < T; ++j) {
        tokens[j] = j < len ? tok_ids[off + j] : 0;  // 원래 채움 값 False → 0
        tmask[j] = j < len ? 1 : 0;
    }
}

// ------------------------------------------------------------------ Sampler
Sampler::Sampler(int64_t n, int bs, uint64_t seed, bool shuffle, bool persistent)
    : mt_((uint32_t)(seed & 0xffffffffu)), n_(n), bs_(bs), shuffle_(shuffle), persistent_(persistent) {
    if (bs <= 0 || n < bs) throw std::runtime_error("배치가 데이터보다 크다");
}

void Sampler::randperm(std::vector<int64_t>& r) {
    r.resize(n_);
    std::iota(r.begin(), r.end(), 0);
    const bool small = n_ < (int64_t)(std::numeric_limits<uint32_t>::max() / 20);
    for (int64_t i = 0; i < n_ - 1; ++i) {
        uint64_t x;
        if (small) {
            x = mt_();
        } else {
            uint64_t hi = mt_(), lo = mt_();
            x = (hi << 32) | lo;
        }
        const int64_t z = (int64_t)(x % (uint64_t)(n_ - i));
        std::swap(r[i], r[z + i]);
    }
}

void Sampler::start_epoch() {
    if (first_ || !persistent_) {
        mt_();  // _base_seed: int64 random_() = 32비트 두 번
        mt_();
    }
    first_ = false;
    if (shuffle_) {
        randperm(perm_);
    } else {
        perm_.resize(n_);
        std::iota(perm_.begin(), perm_.end(), 0);
    }
    pos_ = 0;
    epoch_++;
}

void Sampler::next(int64_t* out) {
    if (epoch_ < 0) start_epoch();
    if (pos_ + bs_ > n_) {                // drop_last: 남은 것은 버리고
        if (shuffle_) randperm(scratch_);  // 에포크 끝에 sampler 가 한 번 더 부르는 randperm (sampler.py:68)
        start_epoch();
    }
    std::memcpy(out, perm_.data() + pos_, (size_t)bs_ * 8);
    pos_ += bs_;
}

// ------------------------------------------------------------------ Loader
Loader::Loader(const std::string& table_dir, const uint8_t* lut_host, const std::vector<int>& sizes,
               const int32_t* start, const int32_t* count, const float* weight, const int32_t* split_rows,
               const int32_t* split_cols, int threads, int device, int batch, bool shuffle, uint64_t seed,
               bool persistent, int nslots)
    : tab_(new Table(table_dir)),
      eng_(new Engine(tab_->files, tab_->indexes, lut_host, sizes, start, count, weight, split_rows, split_cols,
                      threads, device)),
      smp_(tab_->n, batch, seed, shuffle, persistent),
      B_(batch),
      device_(device) {
    if (nslots < 2) throw std::runtime_error("슬롯은 2개 이상");
    if (tab_->cams != 3) throw std::runtime_error("카메라 3개만 지원 (openpi B1KInputs)");
    ckr(cudaSetDevice(device), "cudaSetDevice");
    cudaStream_t cs;
    ckr(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "cudaStreamCreate");
    copy_stream_ = cs;
    const Table& t = *tab_;
    const size_t B = (size_t)batch;
    // 이미지 외 블록 배치 (호스트·GPU 같은 모양, 256 바이트 정렬)
    size_t off = 0;
    size_t o_imask[3];
    for (int c = 0; c < 3; ++c) {
        o_imask[c] = off;
        off = align256(off + B);
    }
    const size_t o_state = off;
    off = align256(off + B * t.M * 4);
    const size_t o_tok = off;
    off = align256(off + B * t.T * 4);
    const size_t o_tm = off;
    off = align256(off + B * t.T);
    const size_t o_act = off;
    off = align256(off + B * t.H * t.M * 4);
    const size_t rest = off;
    const size_t img1 = B * 224 * 224 * 3;
    const size_t img = align256(3 * img1);
    for (int i = 0; i < nslots; ++i) {
        auto s = std::make_unique<Slot>();
        s->owner = this;
        s->id = i;
        s->img_bytes = img;
        s->rest_bytes = rest;
        ckr(cudaMalloc((void**)&s->dev, img + rest), "cudaMalloc 슬롯");
        ckr(cudaHostAlloc((void**)&s->host, rest, cudaHostAllocDefault), "cudaHostAlloc 슬롯");
        std::memset(s->host, 0, rest);
        for (int c = 0; c < 3; ++c) std::memset(s->host + o_imask[c], 1, B);  // 이미지 마스크: 카메라 3개 모두 True
        s->req.resize(3 * B * 2);
        s->idx.resize(B);
        s->latch.on_done = [this, sp = s.get()](Latch*) {
            std::lock_guard<std::mutex> g(m_);
            ready_[sp->seq] = sp;
            cv_.notify_all();
        };
        auto set = [&](int k, uint8_t* data, std::vector<int64_t> shape, uint8_t code, uint8_t bits) {
            DLManagedTensor& d = s->dl[k];
            std::memset(&d, 0, sizeof(d));
            for (size_t j = 0; j < shape.size(); ++j) s->shape[k][j] = shape[j];
            d.dl_tensor.data = data;
            d.dl_tensor.device = {kDLCUDA, device};
            d.dl_tensor.ndim = (int32_t)shape.size();
            d.dl_tensor.dtype = {code, bits, 1};
            d.dl_tensor.shape = s->shape[k];
            d.dl_tensor.strides = nullptr;
            d.dl_tensor.byte_offset = 0;
            d.manager_ctx = s.get();
            d.deleter = dl_deleter;
        };
        const int64_t b = batch;
        for (int c = 0; c < 3; ++c) set(c, s->dev + c * img1, {b, 224, 224, 3}, kDLUInt, 8);
        uint8_t* r = s->dev + img;
        for (int c = 0; c < 3; ++c) set(3 + c, r + o_imask[c], {b}, kDLBool, 8);
        set(6, r + o_state, {b, t.M}, kDLFloat, 32);
        set(7, r + o_tok, {b, t.T}, kDLInt, 32);
        set(8, r + o_tm, {b, t.T}, kDLBool, 8);
        set(9, r + o_act, {b, t.H, t.M}, kDLFloat, 32);
        free_.push_back(i);
        slots_.push_back(std::move(s));
    }
    o_state_ = o_state;
    o_tok_ = o_tok;
    o_tm_ = o_tm;
    o_act_ = o_act;
    producer_ = std::thread([this] { produce(); });
}

void Loader::fill(Slot& s) {
    const Table& t = *tab_;
    smp_.next(s.idx.data());
    uint8_t* h = s.host;
    for (int b = 0; b < B_; ++b)
        t.sample(s.idx[b], (float*)(h + o_act_) + (size_t)b * t.H * t.M, (float*)(h + o_state_) + (size_t)b * t.M,
                 (int32_t*)(h + o_tok_) + (size_t)b * t.T, h + o_tm_ + (size_t)b * t.T);
    for (int c = 0; c < 3; ++c)
        for (int b = 0; b < B_; ++b) {
            const int32_t* q = t.req + ((size_t)s.idx[b] * t.cams + c) * 2;
            s.req[((size_t)c * B_ + b) * 2] = q[0];
            s.req[((size_t)c * B_ + b) * 2 + 1] = q[1];
        }
    cudaStream_t cs = (cudaStream_t)copy_stream_;
    ckr(cudaMemcpyAsync(s.dev + s.img_bytes, s.host, s.rest_bytes, cudaMemcpyHostToDevice, cs), "복사");
    ckr(cudaStreamSynchronize(cs), "복사 동기화");
    s.latch.reset(3 * B_);
    eng_->submit(s.req.data(), 3 * B_, s.dev, 0, &s.latch);
}

void Loader::produce() {
    try {
        ckr(cudaSetDevice(device_), "cudaSetDevice");
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> g(m_);
        err_ = e.what();
        cv_.notify_all();
        return;
    }
    for (;;) {
        Slot* s;
        {
            std::unique_lock<std::mutex> g(m_);
            double t0 = now_s();
            cv_.wait(g, [&] { return stop_ || !free_.empty(); });
            st_.wait_slot_s += now_s() - t0;
            if (stop_) return;
            s = slots_[free_.front()].get();
            free_.pop_front();
            s->seq = next_in_++;
        }
        try {
            double t0 = now_s();
            fill(*s);
            std::lock_guard<std::mutex> g(m_);
            st_.fill_s += now_s() - t0;
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> g(m_);
            err_ = e.what();
            cv_.notify_all();
            return;
        }
    }
}

Slot* Loader::next(int64_t* idx_out) {
    std::unique_lock<std::mutex> g(m_);
    double t0 = now_s();
    cv_.wait(g, [&] { return stop_ || !err_.empty() || ready_.count(next_out_); });
    st_.wait_next_s += now_s() - t0;
    if (!err_.empty()) throw std::runtime_error(err_);
    if (stop_) throw std::runtime_error("로더가 멈췄다");
    Slot* s = ready_[next_out_];
    ready_.erase(next_out_);
    next_out_++;
    if (!s->latch.err.empty()) {
        err_ = s->latch.err;
        throw std::runtime_error(err_);
    }
    s->refs.store(10);
    st_.batches += 1;
    if (idx_out) std::memcpy(idx_out, s->idx.data(), (size_t)B_ * 8);
    return s;
}

void Loader::release(Slot* s) {
    if (s->refs.fetch_sub(1) == 1) {
        std::lock_guard<std::mutex> g(m_);
        free_.push_back(s->id);
        cv_.notify_all();
    }
}

LoaderStats Loader::stats() {
    std::lock_guard<std::mutex> g(m_);
    return st_;
}

bool Loader::busy_slots() {
    for (auto& s : slots_)
        if (s->refs.load() > 0) return true;
    return false;
}

Loader::~Loader() {
    {
        std::lock_guard<std::mutex> g(m_);
        stop_ = true;
    }
    cv_.notify_all();
    if (producer_.joinable()) producer_.join();
    eng_.reset();  // 남은 디코딩 작업을 다 끝낸 뒤 스레드를 멈춘다
    for (auto& s : slots_) {
        cudaFree(s->dev);
        cudaFreeHost(s->host);
    }
    if (copy_stream_) cudaStreamDestroy((cudaStream_t)copy_stream_);
}

}  // namespace ft
