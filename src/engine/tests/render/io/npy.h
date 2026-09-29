// .npy 읽기·쓰기 (호스트 전용, 손으로 짬). 렌더 모듈의 파일 경계는 전부 .npy 한 장 = 배열 하나다:
//   변환기 출력(tests/render/capture/export_render_scene.py) -> 렌더러 입력, 렌더러 출력 -> 비교 도구(render_compare).
// 지원: 버전 1.0/2.0/3.0 머리, 작은 끝(little-endian), C 순서, dtype <f4 <f8 <i4 <i8 |u1 |b1 <u4 <u2 (구조체 dtype 없음).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

struct Array {
  std::string dtype;            // 예: "<f4"
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;    // 원시 바이트
  int64_t size() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
  int itemsize() const { return std::atoi(dtype.c_str() + 2); }
  template <class T> const T* as() const { return reinterpret_cast<const T*>(data.data()); }
  template <class T> T* as() { return reinterpret_cast<T*>(data.data()); }
};

inline std::string header_value(const std::string& h, const std::string& key) {
  const size_t k = h.find("'" + key + "'");
  if (k == std::string::npos) throw std::runtime_error("npy: 머리에 " + key + " 없음");
  size_t p = h.find(':', k) + 1;
  while (p < h.size() && h[p] == ' ') ++p;
  if (h[p] == '\'') { const size_t e = h.find('\'', p + 1); return h.substr(p + 1, e - p - 1); }
  if (h[p] == '(') { const size_t e = h.find(')', p); return h.substr(p + 1, e - p - 1); }
  const size_t e = h.find_first_of(",}", p);
  return h.substr(p, e - p);
}

inline Array load(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("npy: 못 엶 " + path);
  unsigned char magic[8];
  if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "\x93NUMPY", 6) != 0) { std::fclose(f); throw std::runtime_error("npy: 형식 아님 " + path); }
  uint32_t hlen = 0;
  if (magic[6] == 1) { uint16_t h16; if (std::fread(&h16, 2, 1, f) != 1) { std::fclose(f); throw std::runtime_error("npy: 머리"); } hlen = h16; }
  else { if (std::fread(&hlen, 4, 1, f) != 1) { std::fclose(f); throw std::runtime_error("npy: 머리"); } }
  std::string h(hlen, '\0');
  if (std::fread(&h[0], 1, hlen, f) != hlen) { std::fclose(f); throw std::runtime_error("npy: 머리"); }
  Array a;
  a.dtype = header_value(h, "descr");
  if (header_value(h, "fortran_order").find("True") != std::string::npos) { std::fclose(f); throw std::runtime_error("npy: Fortran 순서 안 됨 " + path); }
  const std::string sh = header_value(h, "shape");
  size_t p = 0;
  while (p < sh.size()) {
    while (p < sh.size() && (sh[p] == ' ' || sh[p] == ',')) ++p;
    if (p >= sh.size()) break;
    a.shape.push_back(std::strtoll(sh.c_str() + p, nullptr, 10));
    while (p < sh.size() && sh[p] != ',') ++p;
  }
  if (a.dtype[0] == '>') { std::fclose(f); throw std::runtime_error("npy: 큰 끝 안 됨 " + path); }
  const int64_t bytes = a.size() * a.itemsize();
  a.data.resize(size_t(bytes));
  if (bytes && std::fread(a.data.data(), 1, size_t(bytes), f) != size_t(bytes)) { std::fclose(f); throw std::runtime_error("npy: 자료 짧음 " + path); }
  std::fclose(f);
  return a;
}

inline void save(const std::string& path, const std::string& dtype, const std::vector<int64_t>& shape, const void* data) {
  std::string sh = "(";
  for (size_t i = 0; i < shape.size(); ++i) sh += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
  sh += ")";
  std::string h = "{'descr': '" + dtype + "', 'fortran_order': False, 'shape': " + sh + ", }";
  const size_t total = 10 + h.size() + 1;
  h.append((64 - total % 64) % 64, ' ');
  h += '\n';
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("npy: 못 씀 " + path);
  const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  const uint16_t hl = uint16_t(h.size());
  std::fwrite(magic, 1, 8, f);
  std::fwrite(&hl, 2, 1, f);
  std::fwrite(h.data(), 1, h.size(), f);
  int64_t n = 1;
  for (auto s : shape) n *= s;
  std::fwrite(data, size_t(std::atoi(dtype.c_str() + 2)), size_t(n), f);
  std::fclose(f);
}

}  // namespace npy
