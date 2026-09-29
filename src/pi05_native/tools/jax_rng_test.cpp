// Checks the JAX noise stream against tools-dumped reference (data/pi05_native/jax_noise_seq.npy: 3 x 32 x 32 f32).
#include <cmath>
#include <cstdio>
#include <vector>
#include "../src/jax_rng.h"
int main(int argc, char** argv) {
  FILE* f = fopen(argc > 1 ? argv[1] : "jax_noise_seq.npy", "rb");
  if (!f) return 2;
  std::vector<char> raw(1 << 16);
  size_t n = fread(raw.data(), 1, raw.size(), f);
  fclose(f);
  const uint16_t hl = *(uint16_t*)&raw[8];
  const float* ref = (const float*)&raw[10 + hl];
  if (n < 10 + hl + 3 * 1024 * 4) return 2;
  pi05::JaxKey rng = pi05::jax_key(0), s;
  int bad = 0;
  double maxd = 0;
  for (int i = 0; i < 3; ++i) {
    pi05::jax_split(rng, &rng, &s);
    float out[1024];
    pi05::jax_normal(s, out, 1024);
    int eq = 0;
    for (int j = 0; j < 1024; ++j) {
      double d = std::fabs(out[j] - ref[i * 1024 + j]);
      maxd = std::max(maxd, d);
      eq += out[j] == ref[i * 1024 + j];
      if (d > 1e-5) ++bad;
    }
    printf("draw %d: key %u %u  first %.7f %.7f  bit-equal %d/1024\n", i, s.k0, s.k1, out[0], out[1], eq);
  }
  printf("max |diff| %.3g, %d values off by > 1e-5 -> %s\n", maxd, bad, bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
