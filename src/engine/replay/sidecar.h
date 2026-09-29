// OVD 옆에 두는 보조 파일 두 개 (둘 다 평가기 쪽 capture 가 만든다; 자체 시험에서는 ovd_selftest 가 만든다)
//   convex.bin   볼록 메시 원본 (PxConvexMesh 의 getVertices / getIndexBuffer / getPolygonData 그대로)
//                "CVX1" u32 n  { u32 nv, u32 ni, u32 np, f32 v[3nv], u8 idx[ni], np x (f32 plane[4], u16 nverts, u16 base) }
//   filters.txt  omni.physx 충돌 거르개 표:  "group a b" / "pair a b" / "inverted 0|1" / "contact_report 0|1"
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "omni_filter.h"

namespace engine {

struct ConvexData {
  std::vector<float> verts;       // 3 * nv
  std::vector<uint8_t> indices;
  struct Poly { float plane[4]; uint16_t nverts, base; };
  std::vector<Poly> polys;
};

inline uint64_t hash_bytes(const void* p, size_t n) {  // FNV-1a 64
  const uint8_t* b = static_cast<const uint8_t*>(p);
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}

inline bool write_convex_bin(const std::string& path, const std::vector<ConvexData>& v) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;
  uint32_t n = uint32_t(v.size());
  fwrite("CVX1", 1, 4, f);
  fwrite(&n, 4, 1, f);
  for (const auto& c : v) {
    uint32_t nv = uint32_t(c.verts.size() / 3), ni = uint32_t(c.indices.size()), np = uint32_t(c.polys.size());
    fwrite(&nv, 4, 1, f); fwrite(&ni, 4, 1, f); fwrite(&np, 4, 1, f);
    fwrite(c.verts.data(), 4, c.verts.size(), f);
    fwrite(c.indices.data(), 1, c.indices.size(), f);
    for (const auto& p : c.polys) { fwrite(p.plane, 4, 4, f); fwrite(&p.nverts, 2, 1, f); fwrite(&p.base, 2, 1, f); }
  }
  fclose(f);
  return true;
}

// 꼭짓점 배열 바이트 해시 -> 메시 (OVD 의 PxConvexMesh.verts 와 맞춰 찾는다)
inline bool read_convex_bin(const std::string& path, std::unordered_map<uint64_t, ConvexData>& out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[4];
  uint32_t n = 0;
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CVX1", 4) || fread(&n, 4, 1, f) != 1) { fclose(f); return false; }
  for (uint32_t i = 0; i < n; ++i) {
    ConvexData c;
    uint32_t nv, ni, np;
    if (fread(&nv, 4, 1, f) != 1 || fread(&ni, 4, 1, f) != 1 || fread(&np, 4, 1, f) != 1) { fclose(f); return false; }
    c.verts.resize(size_t(nv) * 3); c.indices.resize(ni); c.polys.resize(np);
    if (fread(c.verts.data(), 4, c.verts.size(), f) != c.verts.size()) { fclose(f); return false; }
    if (ni && fread(c.indices.data(), 1, ni, f) != ni) { fclose(f); return false; }
    for (auto& p : c.polys) {
      if (fread(p.plane, 4, 4, f) != 4 || fread(&p.nverts, 2, 1, f) != 1 || fread(&p.base, 2, 1, f) != 1) { fclose(f); return false; }
    }
    uint64_t h = hash_bytes(c.verts.data(), c.verts.size() * 4);
    out.emplace(h, std::move(c));
  }
  fclose(f);
  return true;
}

inline bool read_filters(const std::string& path, FilterSpec& s) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;
  char key[64];
  unsigned a = 0, b = 0;
  char line[4096];
  auto group_of = [&](const std::string& path) -> FilterSpec::Group& {
    for (auto& g : s.groups) if (g.path == path) return g;
    s.groups.push_back(FilterSpec::Group{path, {}, {}});
    return s.groups.back();
  };
  while (fgets(line, sizeof line, f)) {
    // 경로 기반: "groupfilter <G> <F>" / "groupinclude <G> <prim>" / "rel <A> <B>"
    char p1[2048], p2[2048];
    if (sscanf(line, "groupfilter %2047s %2047s", p1, p2) == 2) { group_of(p1).filtered.push_back(p2); continue; }
    if (sscanf(line, "groupinclude %2047s %2047s", p1, p2) == 2) { group_of(p1).includes.push_back(p2); continue; }
    if (sscanf(line, "rel %2047s %2047s", p1, p2) == 2) { s.rels.push_back({p1, p2}); continue; }
    if (sscanf(line, "%63s %u %u", key, &a, &b) >= 2) {
      if (!strcmp(key, "group")) s.group_pairs.insert(pair_key(a, b));
      else if (!strcmp(key, "pair")) s.filtered_pairs.insert(pair_key(a, b));
      else if (!strcmp(key, "inverted")) s.inverted_group_filter = a != 0;
      else if (!strcmp(key, "contact_report")) s.any_contact_report = a != 0;
    }
  }
  fclose(f);
  return true;
}

inline bool write_filters(const std::string& path, const FilterSpec& s) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return false;
  for (uint64_t k : s.group_pairs) fprintf(f, "group %u %u\n", unsigned(k >> 32), unsigned(k & 0xffffffffu));
  for (uint64_t k : s.filtered_pairs) fprintf(f, "pair %u %u\n", unsigned(k >> 32), unsigned(k & 0xffffffffu));
  fprintf(f, "inverted %d\ncontact_report %d\n", int(s.inverted_group_filter), int(s.any_contact_report));
  fclose(f);
  return true;
}

// ------------------------------------------------------------------------------------------------ 곁기록(sidelog)
// OVD 에 안 남는 호출 목록. 평가기 쪽은 capture/physx_capture.py -> capture/export_sidecar.py 가, 자체 시험은 ovd_selftest 가 쓴다.
//   "SLG1" u32 nviews { u32 kind(0 관절체 뷰/1 강체 뷰/2 프림 하나), u32 nprims, [u32 len, chars]*, u32 max_dofs, [u32 nd, i8 sign*nd]* }
//          u32 ncalls { u64 after, u32 view, u32 len, chars method, u32 nidx, u32 idx*, u32 ndata, f32 data* }
struct SideView {
  uint32_t kind = 0;  // 0 관절체 뷰, 1 강체 뷰, 2 psi(프림 하나)
  std::vector<std::string> prims;
  uint32_t max_dofs = 0;
  std::vector<std::vector<int8_t>> signs;  // prim 별 dof 부호 (isDofBody0Parent)
};
struct SideCall {
  uint64_t after = 0;  // 이 번호의 fetchResults 뒤, 다음 simulate 전
  uint32_t view = 0;
  std::string method;
  std::vector<uint32_t> idx;
  std::vector<float> data;
};
inline bool read_sidelog(const std::string& path, std::vector<SideView>& views, std::vector<SideCall>& calls) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  auto rd = [&](void* p, size_t n) { return fread(p, 1, n, f) == n; };
  auto rstr = [&](std::string& s) {
    uint32_t n = 0;
    if (!rd(&n, 4)) return false;
    s.resize(n);
    return n == 0 || rd(&s[0], n);
  };
  char magic[4];
  uint32_t nv = 0;
  if (!rd(magic, 4) || memcmp(magic, "SLG1", 4) || !rd(&nv, 4)) { fclose(f); return false; }
  views.resize(nv);
  for (auto& v : views) {
    uint32_t np = 0;
    rd(&v.kind, 4); rd(&np, 4);
    v.prims.resize(np);
    for (auto& s : v.prims) rstr(s);
    rd(&v.max_dofs, 4);
    v.signs.resize(np);
    for (auto& s : v.signs) {
      uint32_t nd = 0;
      rd(&nd, 4);
      s.resize(nd);
      if (nd) rd(s.data(), nd);
    }
  }
  uint32_t nc = 0;
  rd(&nc, 4);
  calls.resize(nc);
  for (auto& c : calls) {
    uint32_t ni = 0, nd = 0;
    rd(&c.after, 8); rd(&c.view, 4); rstr(c.method);
    rd(&ni, 4); c.idx.resize(ni); if (ni) rd(c.idx.data(), 4 * ni);
    rd(&nd, 4); c.data.resize(nd); if (nd) rd(c.data.data(), 4 * nd);
  }
  fclose(f);
  return true;
}


inline bool write_sidelog(const std::string& path, const std::vector<SideView>& views, const std::vector<SideCall>& calls) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;
  auto wr = [&](const void* p, size_t n) { fwrite(p, 1, n, f); };
  auto wstr = [&](const std::string& s) { uint32_t n = uint32_t(s.size()); wr(&n, 4); wr(s.data(), n); };
  wr("SLG1", 4);
  uint32_t nv = uint32_t(views.size());
  wr(&nv, 4);
  for (auto& v : views) {
    uint32_t np = uint32_t(v.prims.size());
    wr(&v.kind, 4); wr(&np, 4);
    for (auto& s : v.prims) wstr(s);
    wr(&v.max_dofs, 4);
    for (uint32_t i = 0; i < np; ++i) {
      uint32_t nd = i < v.signs.size() ? uint32_t(v.signs[i].size()) : 0;
      wr(&nd, 4);
      if (nd) wr(v.signs[i].data(), nd);
    }
  }
  uint32_t nc = uint32_t(calls.size());
  wr(&nc, 4);
  for (auto& c : calls) {
    uint32_t ni = uint32_t(c.idx.size()), nd = uint32_t(c.data.size());
    wr(&c.after, 8); wr(&c.view, 4); wstr(c.method);
    wr(&ni, 4); if (ni) wr(c.idx.data(), 4 * ni);
    wr(&nd, 4); if (nd) wr(c.data.data(), 4 * nd);
  }
  fclose(f);
  return true;
}

}  // namespace engine
