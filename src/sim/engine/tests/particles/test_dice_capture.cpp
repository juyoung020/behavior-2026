// 층 1 시험 (실제 과제 기록): 공식 다지기 한 번마다 격자 중심 → 방향(T.random_quaternion, 기록한 난수 상태에서) → 강체 원점 자세.
//   python3 dice_events_to_npy.py <수확 폴더>; ./test_dice_capture <수확 폴더>/dice_npy
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/common/pmath.h"
#include "core/particles/dice.h"
#include "core/particles/spawn.h"
#include "core/particles/trng.h"
#include "tests/omni/npy.h"

using namespace eng::particles;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  long n_ev = 0, bad_cnt = 0, bad_c = 0, bad_f = 0, n_p = 0;
  for (int e = 0;; ++e) {
    char d[512];
    snprintf(d, sizeof d, "%s/ev_%03d/", argv[1], e);
    Npy lo, hi, r, off, rng, nb, cen, fr, mt;
    if (!npy_load(std::string(d) + "lo.npy", lo)) break;
    npy_load(std::string(d) + "hi.npy", hi), npy_load(std::string(d) + "r.npy", r), npy_load(std::string(d) + "off.npy", off);
    npy_load(std::string(d) + "rng.npy", rng), npy_load(std::string(d) + "nbefore.npy", nb), npy_load(std::string(d) + "centers.npy", cen);
    npy_load(std::string(d) + "frames.npy", fr), npy_load(std::string(d) + "mesh_tf.npy", mt);
    const int nm = (int)mt.shape[0];
    std::vector<Npy> keep(4 * nm);
    std::vector<DiceMesh> ms(nm);
    for (int m = 0; m < nm; ++m) {
      const std::string b = std::string(d) + "dl" + std::to_string(m) + "_";
      npy_load(b + "transform.npy", keep[4 * m]), npy_load(b + "neighbors.npy", keep[4 * m + 1]), npy_load(b + "equations.npy", keep[4 * m + 2]),
          npy_load(b + "misc.npy", keep[4 * m + 3]);
      const double* mi = keep[4 * m + 3].as<double>();
      memcpy(ms[m].tf, mt.as<float>() + 16 * m, 64);
      ms[m].dl = Delaunay3{(int32_t)keep[4 * m].shape[0], keep[4 * m].as<double>(), keep[4 * m + 1].as<int32_t>(), keep[4 * m + 2].as<double>(), mi[0], mi[1],
                           {mi[2], mi[3], mi[4]}, {mi[5], mi[6], mi[7]}};
    }
    const int64_t want = cen.shape[0];
    std::vector<float> c(3 * (want + 4096));
    const int64_t got = dice_grid(lo.as<float>(), hi.as<float>(), r.as<float>()[0], ms.data(), nm, c.data(), want + 4096);
    ++n_ev;
    n_p += want;
    if (got != want) {
      printf("  사건 %d: 개수 우리 %ld 공식 %ld\n", e, (long)got, (long)want);
      ++bad_cnt;
      continue;
    }
    bad_c += memcmp(c.data(), cen.as<float>(), (size_t)want * 12) != 0;
    TorchMT m;
    torch_mt_from_bytes(rng.as<uint8_t>(), (int)rng.count(), m);
    std::vector<float> q(4 * want);
    random_quaternion(m, (int)want, q.data());
    const int64_t n0 = nb.as<int32_t>()[0];
    long bf = 0;
    for (int64_t i = 0; i < want; ++i) {
      Pose7 p = particle_frame_from_center(&c[3 * i], &q[4 * i], off.as<float>());
      // PhysX 는 자세를 넣을 때 쿼터니언을 정규화한다 (NpRigidDynamic::setGlobalPose → getNormalized, 엔진은 common/body.h 가 같은 일)
      const eng::Q nq = eng::normalized(eng::Q{p.q[0], p.q[1], p.q[2], p.q[3]});
      p.q[0] = nq.x, p.q[1] = nq.y, p.q[2] = nq.z, p.q[3] = nq.w;
      const float* f = fr.as<float>() + (n0 + i) * 7;
      const bool bp = memcmp(p.p, f, 12) != 0, bq = memcmp(p.q, f + 3, 16) != 0;
      if ((bp || bq) && getenv("DICE_SHOW"))
        printf("    입자 %ld: 위치 %s 방향 %s | q %.9g %.9g %.9g %.9g / %.9g %.9g %.9g %.9g | p %.9g %.9g %.9g / %.9g %.9g %.9g\n", (long)i, bp ? "다름" : "같음",
               bq ? "다름" : "같음", p.q[0], p.q[1], p.q[2], p.q[3], f[3], f[4], f[5], f[6], p.p[0], p.p[1], p.p[2], f[0], f[1], f[2]);
      bf += bp || bq;
    }
    if (bf) printf("  사건 %d: 원점 자세 다름 %ld / %ld\n", e, bf, (long)want);
    bad_f += bf;
  }
  printf("실제 다지기 재생: 사건 %ld, 입자 %ld  개수 다른 사건 %ld  중심 다른 사건 %ld  원점 자세 다름 %ld\n", n_ev, n_p, bad_cnt, bad_c, bad_f);
  const bool ok = n_ev > 0 && !(bad_cnt || bad_c || bad_f);
  printf(ok ? "전부 비트 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
