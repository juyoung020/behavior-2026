// ftbench — 네이티브 로더 처리량을 파이썬 없이 잰다 (C ABI 만 사용).
//   ftbench <table_dir> <lut.bin> [batch=32] [threads=6] [slots=6] [batches=200] [step_ms=0]
// 소비 쪽 흉내: 배치를 받자마자 10개 텐서를 돌려준다. step_ms > 0 이면 그 시간만큼 기다린 뒤 돌려준다(학습 스텝 흉내,
// CPU 에서 잠만 잔다 — GPU 는 로더만 쓴다). 결과: 배치/s, 샘플/s, 소비 쪽이 데이터를 기다린 시간(= GPU 가 기다릴 시간).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

extern "C" {
const char* ft_last_error(void);
void* ft_loader_create(const char*, const uint8_t*, int, int, int, int, uint64_t, int, int);
int ft_loader_next(void*, int64_t*, void**);
int ft_loader_stats(void*, double*);
void* ft_loader_engine(void*);
int ft_engine_stats(void*, double*);
void ft_dl_release(void*);
}

static std::vector<char> slurp(const char* p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "못 연다: %s\n", p);
        std::exit(2);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "사용법: ftbench <table_dir> <lut.bin> [batch] [threads] [slots] [batches] [step_ms]\n");
        return 2;
    }
    const int batch = argc > 3 ? std::atoi(argv[3]) : 32;
    const int threads = argc > 4 ? std::atoi(argv[4]) : 6;
    const int slots = argc > 5 ? std::atoi(argv[5]) : 6;
    const int nb = argc > 6 ? std::atoi(argv[6]) : 200;
    const double step_ms = argc > 7 ? std::atof(argv[7]) : 0.0;
    auto lut = slurp(argv[2]);
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    void* L = ft_loader_create(argv[1], (const uint8_t*)lut.data(), threads, 0, batch, 1, 42, 1, slots);
    if (!L) {
        std::fprintf(stderr, "로더 실패: %s\n", ft_last_error());
        return 1;
    }
    std::vector<int64_t> idx(batch);
    void* dl[10];
    const int warm = slots + 2;
    double wait = 0;
    auto t1 = clk::now();
    for (int i = 0; i < warm + nb; ++i) {
        if (i == warm) {
            t1 = clk::now();
            wait = 0;
        }
        auto a = clk::now();
        if (ft_loader_next(L, idx.data(), dl)) {
            std::fprintf(stderr, "next 실패: %s\n", ft_last_error());
            return 1;
        }
        wait += std::chrono::duration<double>(clk::now() - a).count();
        if (step_ms > 0) std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(step_ms));
        for (int k = 0; k < 10; ++k) ft_dl_release(dl[k]);
    }
    const double el = std::chrono::duration<double>(clk::now() - t1).count();
    double es[4];
    ft_engine_stats(ft_loader_engine(L), es);
    std::printf(
        "{\"batch\": %d, \"threads\": %d, \"slots\": %d, \"batches\": %d, \"step_ms\": %.1f, \"startup_s\": %.2f, "
        "\"batches_per_s\": %.2f, \"samples_per_s\": %.1f, \"wait_ms_per_batch\": %.2f, \"frames_per_request\": %.2f}\n",
        batch, threads, slots, nb, step_ms, std::chrono::duration<double>(t1 - t0).count(), nb / el, nb * batch / el,
        1e3 * wait / nb, es[1] / es[0]);
    std::fflush(stdout);
    std::_Exit(0);  // 슬롯·엔진 정리는 건너뛴다 (측정 도구)
}
