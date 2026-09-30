// openpi training-time image augmentation (models/model.py:168-187, augmax 0.4):
//   image in [-1, 1] -> [0, 1];
//   non-wrist cameras: augmax.Chain(RandomCrop(212, 212), Resize(224, 224), Rotate((-5, 5)), ColorJitter(0.3, 0.4, 0.5))
//   wrist cameras:     augmax.Chain(ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5))
//   -> back to [-1, 1]
// ColorJitter keeps augmax's defaults hue = 0.1 and p = 0.5, and its saturation step has no effect (augmax
// colorspace.py:274-275 discards the result). Geometric chains are one bilinear resample (jax.scipy.ndimage
// map_coordinates, order 1, mode 'constant' 0) at T = crop . resize . rotate applied to the output grid.
// The random parameters come from the same JAX keys as openpi (trng.h), so given a key the output matches JAX.
#pragma once
#include <vector>

#include "../../pi05_native/src/common.cuh"
#include "trng.h"

namespace pi05t {

struct AugParams {
  int geo = 0;               // 1: crop + resize + rotate
  float cy = 0, cx = 0;      // crop center offset (pixels)
  float t00 = 1, t01 = 0, t10 = 0, t11 = 1;  // S . R entries (y, x rows)
  float bright = 0, contrast = 0, hue = 0;
  int apply = 0;             // ColorJitter applied (bernoulli 0.5)
  float slant = 1, p1 = 0, p2 = 1;  // contrast curve constants
};

// parameters of one (sample, camera) from the per-sample key split(preprocess_rng, B)[b]
AugParams aug_params(const JaxKey& sample_key, bool wrist);

// images: [n][3 cams][224][224][3], u8 (model.py:118 conversion) or f32 in [-1, 1]; params [n][3]; out f32 [-1, 1]
void augment(const uint8_t* u8, const float* f32, const AugParams* params_dev, float* out, int n, cudaStream_t st);

}  // namespace pi05t
