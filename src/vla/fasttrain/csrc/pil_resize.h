// 원래 openpi 크기 조정 = openpi_client.image_tools.resize_with_pad (PIL BILINEAR, CPU) 를 그대로 옮기기 위한 계획(호스트).
//
// openpi transforms.py:9 `from openpi_client import image_tools`, :190 ResizeImages → image_tools.py:15-58:
//   ratio = max(W/224, H/224), 크기 = int(H/ratio) × int(W/ratio), Image.resize(BILINEAR), 0 으로 채운 224² 에 붙임
//   (붙이는 위치 max(0, int((224 - 크기)/2))).
// Pillow 11.2.1 src/libImaging/Resample.c (ImagingResampleInner):
//   precompute_coeffs: 출력 칸마다 중심 = (xx + 0.5) * scale, 지지 = 1 * max(scale, 1), 삼각 필터, 합으로 나눔 (double)
//   normalize_coeffs_8bpc: 계수를 22 비트 고정소수점 정수로 (0.5 더해 자름)
//   가로 먼저(필요한 행만) → 8비트 중간 영상 → 세로. 칸마다 ss = 2^21 + Σ 화소 × 계수 (int32), 결과 = clip(ss >> 22)
// 모두 정수 연산이라 GPU 에서 같은 순서로 하면 비트가 같다. 이 파일은 -ffp-contract=off 로 컴파일한다(double 식 그대로).
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace ft {

constexpr int PIL_PRECISION_BITS = 32 - 8 - 2;

struct PilAxis {
    int out = 0, ksize = 0;
    std::vector<int> bounds;  // [out][2] = (시작, 개수)
    std::vector<int> kk;      // [out][ksize] 고정소수점 계수
};

inline double pil_bilinear(double x) {
    if (x < 0.0) x = -x;
    if (x < 1.0) return 1.0 - x;
    return 0.0;
}

// Resample.c precompute_coeffs + normalize_coeffs_8bpc (in0 = 0, in1 = inSize)
inline PilAxis pil_axis(int inSize, int outSize) {
    PilAxis a;
    const float in0 = 0.0f, in1 = (float)inSize;
    double support, scale, filterscale;
    filterscale = scale = (double)(in1 - in0) / outSize;
    if (filterscale < 1.0) filterscale = 1.0;
    support = 1.0 * filterscale;  // bilinear support = 1.0
    const int ksize = (int)std::ceil(support) * 2 + 1;
    std::vector<double> kk((size_t)outSize * ksize);
    a.bounds.resize((size_t)outSize * 2);
    for (int xx = 0; xx < outSize; xx++) {
        double center = in0 + (xx + 0.5) * scale;
        double ww = 0.0;
        double ss = 1.0 / filterscale;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        xmax -= xmin;
        double* k = &kk[(size_t)xx * ksize];
        int x;
        for (x = 0; x < xmax; x++) {
            double w = pil_bilinear((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++)
            if (ww != 0.0) k[x] /= ww;
        for (; x < ksize; x++) k[x] = 0;
        a.bounds[(size_t)xx * 2 + 0] = xmin;
        a.bounds[(size_t)xx * 2 + 1] = xmax;
    }
    a.kk.resize(kk.size());
    for (size_t x = 0; x < kk.size(); x++) {
        if (kk[x] < 0)
            a.kk[x] = (int)(-0.5 + kk[x] * (1 << PIL_PRECISION_BITS));
        else
            a.kk[x] = (int)(0.5 + kk[x] * (1 << PIL_PRECISION_BITS));
    }
    a.out = outSize;
    a.ksize = ksize;
    return a;
}

// 한 입력 크기의 전체 계획 (openpi_client resize_with_pad 의 크기·붙이는 위치 계산 포함)
struct PilPlan {
    int in_w = 0, in_h = 0, out_w = 0, out_h = 0, pad_x = 0, pad_y = 0, target = 224;
    int ybox_first = 0, rows = 0;  // 가로 단계가 계산할 입력 행 [ybox_first, ybox_first + rows)
    PilAxis h, v;                  // v.bounds 의 시작은 ybox_first 만큼 당겨 둠 (Resample.c 와 같이)
};

inline PilPlan pil_plan(int in_w, int in_h, int target = 224) {
    PilPlan p;
    p.in_w = in_w, p.in_h = in_h, p.target = target;
    // image_tools.py:_resize_with_pad_pil — 파이썬 float 연산 그대로 (double)
    double ratio = std::max((double)in_w / target, (double)in_h / target);
    p.out_h = (int)(in_h / ratio);
    p.out_w = (int)(in_w / ratio);
    p.pad_y = std::max(0, (int)((target - p.out_h) / 2.0));
    p.pad_x = std::max(0, (int)((target - p.out_w) / 2.0));
    p.h = pil_axis(in_w, p.out_w);
    p.v = pil_axis(in_h, p.out_h);
    p.ybox_first = p.v.bounds[0];
    const int ybox_last = p.v.bounds[(size_t)p.out_h * 2 - 2] + p.v.bounds[(size_t)p.out_h * 2 - 1];
    p.rows = ybox_last - p.ybox_first;
    for (int i = 0; i < p.out_h; i++) p.v.bounds[(size_t)i * 2] -= p.ybox_first;
    return p;
}

}  // namespace ft
