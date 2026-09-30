// 층 1 시험 (물리 결합): 새 행위자를 틀 + 우리가 계산한 자세로 만들어 PhysX 가 실제로 넣은 몸체(틀 파일 = 리드 g1_sc 가 뜬 값)와 비트 비교하고,
// sc_scene.h ScScene 에 실제로 넣어 본다(번호·모듈 호출 자체는 리드 g1_sc 가 PhysX 와 대조함).
//   python3 spawn_to_txt.py <수확 폴더>; ./test_spawn_exec <수확 폴더>/spawn_rows.txt
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#include "core/omni/gfmat.h"
#include "core/particles/spawn.h"

using namespace eng;
using namespace eng::particles;

struct NullModules : scene::ScModules {  // 넓은 단계·섬 자리 (값만 돌려줌) — 넣기 흐름이 끝까지 도는지만
  uint64_t next = 0;
  bool bpAdd(const scene::BpOp&) override { return true; }
  bool bpRemove(uint32_t) override { return false; }
  uint64_t islandAddNode(bool, bool) override { return next++; }
  void islandDeactivateNode(uint64_t) override {}
  void islandRemoveNode(uint64_t) override {}
};

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  FILE* f = fopen(argv[1], "r");
  if (!f) return 1;
  std::map<std::string, SpawnTemplate> T;
  scene::ScScene sc;
  NullModules nm;
  long n = 0, bad_pose = 0, bad_body = 0, bad_add = 0, nS = 0, nD = 0;
  char kind[4], path[1024];
  unsigned a;
  float sc_half[3] = {1, 1, 1};
  long variant_ok[8] = {0}, variant_n = 0;
  while (fscanf(f, "%3s %1023s %u", kind, path, &a) == 3) {
    Pose7 pose;
    if (kind[0] == 'S') {
      float sp[3], so[4], ss[3];
      HalfSpec h;
      for (float* x : {sp, sp + 1, sp + 2, so, so + 1, so + 2, so + 3, ss, ss + 1, ss + 2}) fscanf(f, "%f", x);
      for (float& x : h.bb_pos) fscanf(f, "%f", &x);
      for (float& x : h.bb_orn) fscanf(f, "%f", &x);
      for (float& x : h.bb_size) fscanf(f, "%f", &x);
      for (float& x : h.native_bb) fscanf(f, "%f", &x);
      for (float& x : h.base_link_offset) fscanf(f, "%f", &x);
      pose = half_add_pose(sp, so, ss, h);
      {  // 반쪽 척도 (가설 시험용)
        float bp[3], bo[4], bb[3];
        slice_part_bbox(sp, so, ss, h.bb_pos, h.bb_orn, h.bb_size, bp, bo, bb);
        half_scale(bb, h.native_bb, sc_half);
      }
      ++nS;
    } else {
      for (float& x : pose.p) fscanf(f, "%f", &x);
      for (float& x : pose.q) fscanf(f, "%f", &x);
      ++nD;
    }
    auto it = T.find(path);
    if (it == T.end()) {
      std::string err;
      if (!T[path].load(path, &err)) {
        fprintf(stderr, "틀 읽기 실패 %s: %s\n", path, err.c_str());
        return 1;
      }
      it = T.find(path);
    }
    const SpawnTemplate& tp = it->second;
    scene::ScActorIn in;
    Body b;
    // USD 왕복 척도 = 행위자 첫 모양의 볼록 척도(= prim 세계 척도, 회전 없는 척도일 때)
    float usc[3] = {1, 1, 1};
    {
      const auto& A = tp.shared->actors[a];
      const auto& g = tp.shared->shapes[A.shapeStart].geom;
      if (kind[0] == 'S') usc[0] = sc_half[0], usc[1] = sc_half[1], usc[2] = sc_half[2];  // 반쪽: 물체 척도 (bounding_box / nativeBB)
      else if (g.type == 5 /*eCONVEXMESH*/) usc[0] = g.convex.scale.scale.x, usc[1] = g.convex.scale.scale.y, usc[2] = g.convex.scale.scale.z;  // 입자 prim 척도
      if (getenv("SHOW_SCALE")) printf("    척도 %s %u: %.9g %.9g %.9g  (회전 %.9g %.9g %.9g %.9g) 종류 %d\n", kind, a, usc[0], usc[1], usc[2], g.convex.scale.rotation.x,
                                       g.convex.scale.rotation.y, g.convex.scale.rotation.z, g.convex.scale.rotation.w, g.type);
    }
    if (!actor_from_template(tp, a, pose, in, b, usc, kind[0] == 'D')) {
      ++bad_add;
      continue;
    }
    const Body& want = tp.f.bodies[tp.shared->actors[a].body];
    const bool okp = memcmp(&b.body2World, &want.body2World, sizeof(Tf)) == 0;
    if (!okp && (!bad_pose || getenv("SHOW_ALL")))
      printf("  첫 자세 다름 (%s 행위자 %u, %s): 우리 q %.9g %.9g %.9g %.9g p %.9g %.9g %.9g / PhysX q %.9g %.9g %.9g %.9g p %.9g %.9g %.9g\n", path, a, kind,
             b.body2World.q.x, b.body2World.q.y, b.body2World.q.z, b.body2World.q.w, b.body2World.p.x, b.body2World.p.y, b.body2World.p.z, want.body2World.q.x,
             want.body2World.q.y, want.body2World.q.z, want.body2World.q.w, want.body2World.p.x, want.body2World.p.y, want.body2World.p.z);
    if (!okp && getenv("SHOW_ALL"))
      printf("    입력 자세 p %.9g %.9g %.9g q %.9g %.9g %.9g %.9g 척도 %.9g %.9g %.9g  body2Actor q %.9g %.9g %.9g %.9g p %.9g %.9g %.9g\n", pose.p[0], pose.p[1],
             pose.p[2], pose.q[0], pose.q[1], pose.q[2], pose.q[3], usc[0], usc[1], usc[2], b.body2Actor.q.x, b.body2Actor.q.y, b.body2Actor.q.z, b.body2Actor.q.w,
             b.body2Actor.p.x, b.body2Actor.p.y, b.body2Actor.p.z);
    bad_pose += !okp;
    if (getenv("DUMP_ROWS")) {  // pxr 파이썬 가설 시험용: 입력 자세·척도·body2Actor·PhysX 자세
      static FILE* dr = fopen(getenv("DUMP_ROWS"), "w");
      fprintf(dr, "%s %u %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n", kind, a, pose.p[0], pose.p[1],
              pose.p[2], pose.q[0], pose.q[1], pose.q[2], pose.q[3], usc[0], usc[1], usc[2], b.body2Actor.q.x, b.body2Actor.q.y, b.body2Actor.q.z,
              b.body2Actor.q.w, want.body2World.q.x, want.body2World.q.y, want.body2World.q.z, want.body2World.q.w, okp ? 1.f : 0.f, 0.f, 0.f);
      fflush(dr);
    }
    if (getenv("RAWQ_PROBE")) {  // 가설: USD 왕복 없이 입력 q 그대로 / 정규화만
      const Tf r0 = Tf{Q{pose.q[0], pose.q[1], pose.q[2], pose.q[3]}, b.body2World.p};
      const Tf r1 = Tf{normalized(r0.q), b.body2World.p};
      const Tf g0 = Tf{r0.q, normalized(pose7_tf(pose)).p} * b.body2Actor, g1 = Tf{r1.q, normalized(pose7_tf(pose)).p} * b.body2Actor;
      namespace gf = eng::omni::gf;
      bool hs[2];
      for (int v = 0; v < 2; ++v) {  // 척도 없는 행렬: v=0 ExtractRotationQuat 만, v=1 RemoveScaleShear 거침
        const float pq[4] = {pose.q[0], pose.q[1], pose.q[2], pose.q[3]}, pp[3] = {pose.p[0], pose.p[1], pose.p[2]};
        gf::M4 M = gf::from_physx_pose(pp, pq);
        double q[4];
        gf::extract_rotation_quat(v ? gf::remove_scale_shear(M) : M, q);
        const Tf g = Tf{normalized(Q{(float)q[0], (float)q[1], (float)q[2], (float)q[3]}), normalized(pose7_tf(pose)).p} * b.body2Actor;
        hs[v] = memcmp(&g.q, &want.body2World.q, 16) == 0;
      }
      printf("    %s %u: 왕복 %s  raw q %s  정규화 q %s  척도없음 ERQ %s  척도없음 RSS %s\n", kind, a, okp ? "O" : "X", memcmp(&g0.q, &want.body2World.q, 16) ? "X" : "O",
             memcmp(&g1.q, &want.body2World.q, 16) ? "X" : "O", hs[0] ? "O" : "X", hs[1] ? "O" : "X");
    }
    if (getenv("GF_VARIANTS") && !okp) {  // 틀린 것만: USD 왕복 방식 후보  // 반쪽: USD 왕복 방식 후보
      namespace gf = eng::omni::gf;
      const Tf ap0 = normalized(pose7_tf(pose));
      for (int v = 0; v < 8; ++v) {
        const bool rawq = v & 1, rt = (v >> 1) & 1, fmat = (v >> 2) & 1;
        const float pq[4] = {rawq ? pose.q[0] : ap0.q.x, rawq ? pose.q[1] : ap0.q.y, rawq ? pose.q[2] : ap0.q.z, rawq ? pose.q[3] : ap0.q.w};
        const float pp[3] = {pose.p[0], pose.p[1], pose.p[2]};
        gf::M4 M = gf::from_physx_pose(pp, pq);
        if (fmat) {  // float 로 계산한 회전 행렬 (Fabric/float 경로 가설)
          const float x = pq[0], y = pq[1], z = pq[2], w = pq[3];
          const float R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (z * x - y * w), 2 * (x * y - z * w), 1 - 2 * (z * z + x * x),
                              2 * (y * z + x * w), 2 * (z * x + y * w), 2 * (y * z - x * w), 1 - 2 * (y * y + x * x)};
          for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) M.m[r][c] = (double)(R[3 * r + c] * usc[r]);
        } else {
          for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) M.m[r][c] *= (double)usc[r];
        }
        double q[4];
        gf::extract_rotation_quat(rt ? gf::rt_remove_scale_shear(M) : gf::remove_scale_shear(M), q);
        const Tf g = Tf{normalized(Q{(float)q[0], (float)q[1], (float)q[2], (float)q[3]}), ap0.p} * b.body2Actor;
        const bool hit = memcmp(&g, &want.body2World, sizeof(Tf)) == 0;
        variant_ok[v] += hit;
        if (hit) printf("    %s %u: 후보 [%d%d%d] 맞음\n", kind, a, v & 1, (v >> 1) & 1, (v >> 2) & 1);
      }
      ++variant_n;
    }
    if (getenv("GF_PROBE")) {
      if (kind[0] == 'D') sc_half[0] = sc_half[1] = sc_half[2] = 1.0f;  // 가설: USD 행렬(척도·회전·이동) → RemoveScaleShear → ExtractRotationQuat → float
      namespace gf = eng::omni::gf;
      Tf ap = normalized(pose7_tf(pose));
      const float pq[4] = {ap.q.x, ap.q.y, ap.q.z, ap.q.w}, pp[3] = {ap.p.x, ap.p.y, ap.p.z};
      static float lastsc[3];
      (void)lastsc;
      for (int v = 0; v < 2; ++v) {
        gf::M4 M = gf::from_physx_pose(pp, pq);
        const float s3 = sc_half[0];
        for (int r = 0; r < 3; ++r)
          for (int c = 0; c < 3; ++c) M.m[r][c] *= (double)sc_half[r];
        (void)s3;
        const gf::M4 R = v == 0 ? gf::remove_scale_shear(M) : gf::rt_remove_scale_shear(M);
        double q[4];
        gf::extract_rotation_quat(R, q);
        Tf g{normalized(Q{(float)q[0], (float)q[1], (float)q[2], (float)q[3]}), ap.p};
        g = g * b.body2Actor;
        printf("    가설 %s: %s  %.17g %.17g %.17g %.17g -> %.9g %.9g %.9g %.9g / %.9g %.9g %.9g %.9g\n", v == 0 ? "pxr" : "usdrt",
               memcmp(&g.q, &want.body2World.q, 16) == 0 ? "같음" : "다름", q[0], q[1], q[2], q[3], g.q.x, g.q.y, g.q.z, g.q.w, want.body2World.q.x,
               want.body2World.q.y, want.body2World.q.z, want.body2World.q.w);
      }
    }
    bad_body += memcmp(&b, &want, sizeof(Body)) != 0 && okp;  // 자세 말고 다른 칸 (속도 등 — 틀은 창 끝 값)
    bad_add += sc.addActor(in, nm) < 0;
    ++n;
  }
  fclose(f);
  if (variant_n) {
    printf("  반쪽 USD 왕복 후보 (정규화 전 q, usdrt, float 행렬) 맞은 수 / %ld:", variant_n);
    for (int v = 0; v < 8; ++v) printf(" [%d%d%d]%ld", v & 1, (v >> 1) & 1, (v >> 2) & 1, variant_ok[v]);
    printf("\n");
  }
  printf("물리 결합 넣기: 행위자 %ld (반쪽 %ld, 다진 입자 %ld)  body2World 다름 %ld  다른 몸체 칸 다름 %ld  넣기 실패 %ld  (ScScene 행위자 %zu)\n", n, nS, nD,
         bad_pose, bad_body, bad_add, sc.actors.size());
  const bool ok = n > 0 && !(bad_pose || bad_add);
  printf(ok ? "자세 비트 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
