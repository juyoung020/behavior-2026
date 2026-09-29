// 네이티브 학습 데이터 로더 — torch DataLoader·워커 프로세스 없이 C++ 스레드와 CUDA 로 배치를 만든다.
//
//   Table   : Rust `ftprep table` 이 만든 프레임별 표 (mmap). 샘플 하나의 이미지 외 값(행동 창 32개의 델타·정규화, 상태,
//             토큰)을 원래 openpi 와 같은 산술로 만든다.
//   Sampler : torch DataLoader(shuffle, generator seed, drop_last) 가 내는 인덱스 순서를 그대로 (MT19937 + randperm).
//   Loader  : GPU 슬롯 링. 생산 스레드가 슬롯에 이미지 외 값을 채워 복사하고 엔진에 디코딩 작업을 넣는다. 엔진 스레드가
//             마지막 이미지를 쓰면 슬롯이 준비됨. next() 가 순서대로 내주고, 10개 DLPack 텐서가 모두 풀리면 슬롯을 돌려받는다.
#pragma once
#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "dlpack_min.h"
#include "engine.h"

namespace ft {

struct Table {
    int64_t n = 0;
    int S = 0, A = 0, M = 0, H = 0, T = 0, cams = 0;
    std::vector<std::pair<int, int>> delta;  // (행동 차원, 상태 차원)
    std::vector<std::string> files, indexes;
    const int64_t* abs = nullptr;
    const int32_t* task = nullptr;
    const int32_t* ep_last = nullptr;
    const float* state_ex = nullptr;
    const float* action = nullptr;
    const float* state_out = nullptr;
    const uint64_t* tok_off = nullptr;
    const int32_t* tok_ids = nullptr;
    const int32_t* req = nullptr;
    const double* norm = nullptr;  // action mean[A], std[A]

    explicit Table(const std::string& dir);
    ~Table();
    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;

    // 샘플 i 의 이미지 외 값. actions [H][M] f32, state [M] f32, tokens [T] i32, tmask [T] u8 (0/1)
    void sample(int64_t i, float* actions, float* state, int32_t* tokens, uint8_t* tmask) const;

private:
    std::vector<std::pair<void*, size_t>> maps_;
    const void* map(const std::string& path, size_t expect_bytes);
};

// torch.utils.data.DataLoader(shuffle=True, generator=manual_seed(seed), drop_last=True) 의 인덱스 순서.
// persistent=true: 워커 지속(openpi 기본, 워커 > 0) — 기준 시드를 반복자 만들 때 한 번만 뽑는다.
// persistent=false: 워커 0 — 에포크마다 새 반복자라 기준 시드를 에포크마다 뽑는다.
// 근거: torch/utils/data/dataloader.py:697-702 (_base_seed = random_() int64 → 32비트 두 번), 710-711 (_reset 은 안 뽑음),
//       sampler.py:45-70 (randperm, 에포크 끝 나머지 randperm), ATen randperm_cpu (z = random() % (n-i), 교환).
// CPU 에서 torch 와 대조: tools/ft_verify.py order.
class Sampler {
public:
    Sampler(int64_t n, int bs, uint64_t seed, bool shuffle, bool persistent);
    void next(int64_t* out);  // 배치 하나 (bs 개)
    int64_t epoch() const { return epoch_; }

private:
    void start_epoch();
    void randperm(std::vector<int64_t>& r);
    std::mt19937 mt_;
    int64_t n_;
    int bs_;
    bool shuffle_, persistent_, first_ = true;
    std::vector<int64_t> perm_, scratch_;
    int64_t pos_ = 0, epoch_ = -1;
};

struct LoaderStats {
    double batches = 0, wait_next_s = 0, wait_slot_s = 0, fill_s = 0;
};

class Loader;

struct Slot {
    Loader* owner = nullptr;
    int id = 0;
    uint8_t* dev = nullptr;   // GPU: [이미지 3개][이미지 외 블록]
    uint8_t* host = nullptr;  // 고정(pinned) 호스트: 이미지 외 블록
    size_t img_bytes = 0, rest_bytes = 0;
    std::vector<int64_t> req, idx;
    Latch latch;
    int64_t seq = -1;
    std::atomic<int> refs{0};
    DLManagedTensor dl[10];
    int64_t shape[10][4];
};

class Loader {
public:
    Loader(const std::string& table_dir, const uint8_t* lut_host, const std::vector<int>& sizes, const int32_t* start,
           const int32_t* count, const float* weight, const int32_t* split_rows, const int32_t* split_cols, int threads,
           int device, int batch, bool shuffle, uint64_t seed, bool persistent, int nslots);
    ~Loader();

    // 다음 배치 (순서대로). dl[10] = 이미지 3, 이미지 마스크 3, 상태, 토큰, 토큰 마스크, 행동. idx[batch] = 샘플 번호.
    // 반환된 DLManagedTensor 10개가 모두 deleter 로 풀려야 슬롯이 다시 쓰인다.
    Slot* next(int64_t* idx_out);
    void release(Slot* s);
    Engine& engine() { return *eng_; }
    const Table& table() const { return *tab_; }
    LoaderStats stats();
    bool busy_slots();  // JAX 가 아직 들고 있는 슬롯이 있나 (있으면 로더를 지우면 안 된다)

private:
    void produce();
    void fill(Slot& s);
    std::unique_ptr<Table> tab_;
    std::unique_ptr<Engine> eng_;
    Sampler smp_;
    int B_, device_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<int> free_;
    std::map<int64_t, Slot*> ready_;
    int64_t next_in_ = 0, next_out_ = 0;
    bool stop_ = false;
    std::string err_;
    void* copy_stream_ = nullptr;
    size_t o_state_ = 0, o_tok_ = 0, o_tm_ = 0, o_act_ = 0;  // 이미지 외 블록 안 위치
    LoaderStats st_;
    std::thread producer_;
};

}  // namespace ft
