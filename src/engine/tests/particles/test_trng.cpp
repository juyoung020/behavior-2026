// 층 1 시험: torch 난수 (get_rng_state → T.random_quaternion, 뒤 상태, 다음 randint) vs 공식 torch.
//   ./test_trng ~/engine-data/particles/trng
#include <cstdio>
#include <cstring>
#include <string>

#include "core/particles/trng.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string d = std::string(argv[1]) + "/";
  Npy st, af, nn, qu, ri;
  if (!npy_load(d + "state.npy", st) || !npy_load(d + "after.npy", af) || !npy_load(d + "n.npy", nn) || !npy_load(d + "quat.npy", qu) ||
      !npy_load(d + "randint_next.npy", ri))
    return 1;
  const int N = (int)st.shape[0], B = (int)st.shape[1], M = (int)qu.shape[1];
  long bad_q = 0, n_q = 0, bad_s = 0, bad_r = 0;
  for (int i = 0; i < N; ++i) {
    TorchMT m;
    torch_mt_from_bytes(st.as<uint8_t>() + (size_t)i * B, B, m);
    const int n = nn.as<int32_t>()[i];
    float q[64 * 4];
    random_quaternion(m, n, q);
    for (int k = 0; k < 4 * n; ++k, ++n_q) {
      uint32_t a, b;
      memcpy(&a, &q[k], 4);
      memcpy(&b, qu.as<float>() + (size_t)i * M * 4 + k, 4);
      if (a != b && bad_q++ == 0) printf("  첫 다름: 사례 %d 원소 %d\n", i, k);
    }
    TorchMT m2;
    torch_mt_from_bytes(af.as<uint8_t>() + (size_t)i * B, B, m2);
    bad_s += m.left != m2.left || m.next != m2.next || memcmp(m.state, m2.state, sizeof m.state) != 0;
    bad_r += torch_randint64(m, INT64_MIN, INT64_MAX) != ri.as<int64_t>()[i];
  }
  printf("torch 난수 층 1 vs 공식: 사례 %d (상태 %d 바이트)  random_quaternion 원소 %ld 다름 %ld  뒤 상태 다름 %ld  다음 randint 다름 %ld\n", N, B, n_q,
         bad_q, bad_s, bad_r);
  const bool ok = !(bad_q || bad_s || bad_r);
  printf(ok ? "전부 비트 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
