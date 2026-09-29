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
  char line[256];
  while (fgets(line, sizeof line, f)) {
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

}  // namespace engine
