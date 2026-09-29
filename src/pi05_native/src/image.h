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

}  // namespace pi05
