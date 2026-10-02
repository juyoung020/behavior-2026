// openpi_client.image_tools.resize_with_pad reproduced bit-exactly: PIL Image.resize(BILINEAR) on 8-bit RGB
// (Pillow libImaging/Resample.c: separable passes, 22-bit fixed-point coefficients) + centered zero padding.
#pragma once
#include <cstdint>
#include <vector>

namespace pi05 {

struct ImageView {
  const uint8_t* data = nullptr;  // first channel of pixel (0,0)
  int h = 0, w = 0;
  long long row_stride = 0;   // bytes
  long long pix_stride = 3;   // bytes (3 for RGB, 4 for RGBA; only the first 3 channels are used)
};

// Writes out_h x out_w x 3 into dst (row-major). Same size input -> plain copy of RGB.
void resize_with_pad(const ImageView& src, int out_h, int out_w, uint8_t* dst);

// Resample.c precompute_coeffs + normalize_coeffs_8bpc (22-bit fixed point): bounds [out][2], kk [out][ksize]
int pil_coeffs(int in_size, int out_size, std::vector<int>& bounds, std::vector<int>& kk);

// The same resize_with_pad on the GPU (device source RGB/RGBA with strides -> device 224x224x3): integer arithmetic
// identical to the host version, so the bytes are identical.
void resize_with_pad_gpu(const uint8_t* src, int h, int w, long long row_stride, int pix_stride, int out_h, int out_w,
                         uint8_t* dst, void* stream);

}  // namespace pi05
