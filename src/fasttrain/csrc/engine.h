// NVDEC 디코딩 엔진 인터페이스. CUDA·ffnvcodec 헤더를 드러내지 않는다. torch 와 무관.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ft {

// 작업 묶음(배치 하나)의 완료 표시. 마지막 작업을 끝낸 엔진 스레드가 finish() 를 부른다.
struct Latch {
    std::atomic<int> left{0};
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    std::string err;
    std::function<void(Latch*)> on_done;  // 있으면 finish() 에서 부른다 (엔진 스레드에서)

    void reset(int n) {
        std::lock_guard<std::mutex> g(m);
        left.store(n);
        done = false;
        err.clear();
    }
    void fail(const std::string& e) {
        std::lock_guard<std::mutex> g(m);
        if (err.empty()) err = e;
    }
    void finish() {
        if (on_done) {
            on_done(this);
            return;
        }
        std::lock_guard<std::mutex> g(m);
        done = true;
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> g(m);
        cv.wait(g, [&] { return done; });
    }
};

class Engine {
public:
    // videos[i] 의 패킷 색인이 indexes[i] (Rust ftprep 가 만든 .ftidx).
    // lut_host: 색 변환 표 (2^24 × 3 바이트, 호스트) — 엔진이 GPU 로 복사해 들고 있는다.
    // 크기 조정은 원래(openpi_client PIL BILINEAR → 224²)와 같은 계획을 해상도마다 여기서 만든다 (pil_resize.h).
    Engine(const std::vector<std::string>& videos, const std::vector<std::string>& indexes, const uint8_t* lut_host,
           int threads, int device);
    ~Engine();

    // 비동기: req [n][2] = (파일 번호, 프레임 번호) 를 작업 큐에 넣는다 (req 는 복사됨).
    // mode 0: 요청 j 의 결과 224x224x3 uint8 을 out + j*150528 (GPU) 에.  mode 1: 색 변환까지만, out + j*W*H*3.
    // latch 는 호출 전에 reset(n) 해 둘 것. 모든 작업이 끝나면 latch->finish().
    void submit(const int64_t* req, int n, uint8_t* out, int mode, Latch* latch);
    // 동기 판 (검증용)
    void run(const int64_t* req, int n, uint8_t* out, int mode);
    // 검증용: GPU uint8 [n][H][W][3] → [n][224][224][3] (크기 조정 커널만)
    void resize(const uint8_t* in, int n, int W, int H, uint8_t* out);

    int device() const;
    std::vector<int64_t> info(int file) const;  // width, height, timescale, 표본 수
    std::vector<double> stats() const;          // 요청 수, 디코딩한 프레임 수, 파서 생성 수, 매핑~해제 초

    struct Impl;

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace ft
