// 시험용 .npy 읽기 (numpy 1.0 형식, 작은 끝, C 순서). 정답지 파이썬이 np.save 로 쓴 배열을 그대로 읽는다.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct Npy {
  std::string descr;           // '<f4', '<i4', '|u1', '<u8' ...
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;
  size_t count() const {
    size_t n = 1;
    for (auto s : shape) n *= (size_t)s;
    return n;
  }
  template <class T>
  const T* as() const {
    return reinterpret_cast<const T*>(data.data());
  }
  template <class T>
  std::vector<T> vec() const {
    std::vector<T> v(count());
    if (!v.empty()) memcpy(v.data(), data.data(), v.size() * sizeof(T));
    return v;
  }
};

inline bool npy_load(const std::string& path, Npy& out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[6];
  if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); throw std::runtime_error("bad npy " + path); }
  uint8_t ver[2];
  fread(ver, 1, 2, f);
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint16_t h;
    fread(&h, 2, 1, f);
    hlen = h;
  } else {
    fread(&hlen, 4, 1, f);
  }
  std::string hdr(hlen, '\0');
  fread(&hdr[0], 1, hlen, f);
  auto p = hdr.find("'descr':");
  auto q1 = hdr.find('\'', p + 8);
  auto q2 = hdr.find('\'', q1 + 1);
  out.descr = hdr.substr(q1 + 1, q2 - q1 - 1);
  if (hdr.find("'fortran_order': True") != std::string::npos) { fclose(f); throw std::runtime_error("fortran npy"); }
  auto s1 = hdr.find('(', hdr.find("'shape':"));
  auto s2 = hdr.find(')', s1);
  std::string sh = hdr.substr(s1 + 1, s2 - s1 - 1);
  out.shape.clear();
  size_t i = 0;
  while (i < sh.size()) {
    while (i < sh.size() && (sh[i] == ' ' || sh[i] == ',')) ++i;
    if (i >= sh.size()) break;
    size_t j = i;
    while (j < sh.size() && sh[j] != ',') ++j;
    out.shape.push_back(std::stoll(sh.substr(i, j - i)));
    i = j;
  }
  size_t esz = (size_t)std::stoi(out.descr.substr(2));
  out.data.resize(out.count() * esz);
  if (!out.data.empty() && fread(out.data.data(), 1, out.data.size(), f) != out.data.size()) {
    fclose(f);
    throw std::runtime_error("short npy " + path);
  }
  fclose(f);
  return true;
}
