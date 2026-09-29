// NVDEC 디코딩 엔진 인터페이스 (CUDA·ffnvcodec 헤더를 드러내지 않는다 → torch 바인딩과 따로 컴파일).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ft {

class Engine {
public:
    // videos[i] 의 패킷 색인이 indexes[i] (Rust ftprep index 가 만든 .ftidx).
    // sizes: 입력 해상도 목록(정사각), start/count: [len(sizes)][224] int32, weight: [len(sizes)][224][8] float32,
    // split_rows/split_cols: [len(sizes)][224] int32 (calib.py 가 원래 크기 조정에서 알아낸 분할 위치)
    // lut_dev: GPU 의 색 변환 표 (2^24 × 3 바이트)
    Engine(const std::vector<std::string>& videos, const std::vector<std::string>& indexes, uintptr_t lut_dev,
           const std::vector<int>& sizes, const int32_t* start, const int32_t* count, const float* weight,
           const int32_t* split_rows, const int32_t* split_cols, int threads, int device);
    ~Engine();

    // req: [n][2] = (파일 번호, 프레임 번호(표시 순서)).
    // mode 0: out = [n][224][224][3] uint8 (GPU)  — 원래 파이프라인의 크기 조정 결과와 같은 것
    // mode 1: out = [n][H][W][3] uint8 (GPU)      — 색 변환까지만 (검증용, 같은 해상도끼리만)
    void run(const int64_t* req, int n, uintptr_t out, int mode);

    // 검증용: GPU uint8 [n][W][W][3] → [n][224][224][3] (크기 조정 커널만)
    void resize(uintptr_t in, int n, int W, uintptr_t out);

    std::vector<int64_t> info(int file) const;   // width, height, timescale, 표본 수
    std::vector<int64_t> pts(int file) const;    // 표시 시각(timescale 단위), 표시 순서
    std::vector<double> stats() const;           // 요청 수, 디코딩한 프레임 수, 파서 생성 수, 매핑 대기 초

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace ft
