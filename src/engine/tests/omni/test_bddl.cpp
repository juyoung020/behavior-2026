// omni 시험 1: BDDL 목표 판정·q_score — 우리 C++ (core/omni/bddl.h) = 공식 bddl3 (gen_bddl_ref.py 가 적은 정답).
//   test_bddl <정답 폴더(~/engine-data/omni/bddl)>
// 비교: 원자 집합, ground option 수·정규형 해시, 시행마다 성공 여부·HEAD 별 참/거짓·q_score(double 비트).
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/omni/bddl.h"

using namespace eng::omni::bddl;

static uint64_t fnv1a64(const std::string& s) {
  uint64_t h = 0xCBF29CE484222325ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001B3ull;
  }
  return h;
}

struct RefTrial {
  std::string bits;
  int success;
  std::string heads;
  uint64_t q_bits;
};
struct Ref {
  std::string task, problem;
  std::vector<Atom> atoms;
  int n_heads = 0, n_opts = 0;
  uint64_t opthash = 0;
  std::vector<RefTrial> trials;
};

static bool load_ref(const std::string& path, Ref& r) {
  std::ifstream f(path);
  if (!f) return false;
  std::string line;
  std::getline(f, line);
  r.task = line.substr(5);
  std::getline(f, line);  // PROBLEM_BEGIN
  std::ostringstream pb;
  while (std::getline(f, line) && line != "PROBLEM_END") pb << line << "\n";
  r.problem = pb.str();
  auto kv = [&](const char* key) {
    std::getline(f, line);
    if (line.compare(0, strlen(key), key) != 0) throw std::runtime_error(std::string("expected ") + key);
    return line.substr(strlen(key) + 1);
  };
  const int na = std::stoi(kv("ATOMS"));
  for (int i = 0; i < na; ++i) {
    std::getline(f, line);
    std::istringstream ls(line);
    Atom a;
    ls >> a.pred;
    std::string x;
    while (ls >> x) a.args.push_back(x);
    r.atoms.push_back(a);
  }
  r.n_heads = std::stoi(kv("HEADS"));
  r.n_opts = std::stoi(kv("NOPTIONS"));
  r.opthash = std::stoull(kv("OPTHASH"), nullptr, 16);
  const int nt = std::stoi(kv("TRIALS"));
  for (int i = 0; i < nt; ++i) {
    std::getline(f, line);
    std::istringstream ls(line);
    RefTrial t;
    std::string qhex;
    ls >> t.bits >> t.success >> t.heads >> qhex;
    // struct.pack("<d").hex() = 작은 끝 바이트 순서 16진
    uint64_t v = 0;
    for (int b = 7; b >= 0; --b) v = (v << 8) | std::stoul(qhex.substr(2 * b, 2), nullptr, 16);
    t.q_bits = v;
    r.trials.push_back(t);
  }
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: test_bddl <ref dir>\n");
    return 2;
  }
  const std::string dir = argv[1];
  std::ifstream idx(dir + "/index.txt");
  std::string name;
  long long n_cmp = 0, n_bad = 0, n_tasks = 0, n_opts_total = 0;
  double t_eval = 0.0;
  while (std::getline(idx, name)) {
    if (name.empty()) continue;
    Ref r;
    if (!load_ref(dir + "/" + name + ".txt", r)) {
      printf("%s: 정답 파일 없음\n", name.c_str());
      ++n_bad;
      continue;
    }
    ++n_tasks;
    Compiled c = compile_problem(parse_problem(r.problem));
    long long bad = 0;
    const char* first = nullptr;
    // 원자 대응: 정답 번호 -> 우리 번호
    std::map<Atom, int> ours;
    for (size_t i = 0; i < c.atoms.size(); ++i) ours[c.atoms[i]] = (int)i;
    std::vector<int> ref2our(r.atoms.size(), -1), our2ref(c.atoms.size(), -1);
    for (size_t i = 0; i < r.atoms.size(); ++i) {
      auto it = ours.find(r.atoms[i]);
      if (it != ours.end()) {
        ref2our[i] = it->second;
        our2ref[it->second] = (int)i;
      }
    }
    ++n_cmp;
    bool atoms_ok = r.atoms.size() == c.atoms.size();
    for (int v : ref2our) atoms_ok = atoms_ok && v >= 0;
    if (!atoms_ok) { ++bad; first = first ? first : "atoms"; }
    ++n_cmp;
    if ((int)c.heads.size() != r.n_heads) { ++bad; first = first ? first : "heads"; }
    ++n_cmp;
    const int n_opts = (int)c.opt_start.size() - 1;
    n_opts_total += n_opts;
    if (n_opts != r.n_opts) { ++bad; first = first ? first : "n_options"; }
    // 정규형 해시
    if (atoms_ok) {
      std::vector<std::vector<std::pair<int, int>>> canon;
      for (int o = 0; o < n_opts; ++o) {
        std::vector<std::pair<int, int>> lits;
        for (int k = c.opt_start[o]; k < c.opt_start[o + 1]; ++k)
          lits.emplace_back(c.opt_lits[k].neg, our2ref[c.opt_lits[k].atom]);
        std::sort(lits.begin(), lits.end());
        canon.push_back(std::move(lits));
      }
      std::sort(canon.begin(), canon.end());
      std::string text;
      for (size_t o = 0; o < canon.size(); ++o) {
        if (o) text += ";";
        for (size_t k = 0; k < canon[o].size(); ++k) {
          if (k) text += ",";
          text += std::to_string(canon[o][k].first) + ":" + std::to_string(canon[o][k].second);
        }
      }
      ++n_cmp;
      if (fnv1a64(text) != r.opthash) { ++bad; first = first ? first : "option_hash"; }
    }
    // 시행
    std::vector<uint8_t> now(c.atoms.size()), init(c.atoms.size()), node_val(c.nodes.size()), head_val(c.heads.size());
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t t = 0; t < r.trials.size() && atoms_ok; ++t) {
      const RefTrial& tr = r.trials[t];
      const RefTrial& prev = r.trials[t == 0 ? 0 : t - 1];
      for (size_t i = 0; i < r.atoms.size(); ++i) {
        now[ref2our[i]] = tr.bits[i] == '1';
        init[ref2our[i]] = prev.bits[i] == '1';
      }
      const bool ok = eval_goal(c.nodes.data(), (int)c.nodes.size(), c.kids.data(), c.heads.data(), (int)c.heads.size(),
                                now.data(), node_val.data(), head_val.data());
      ++n_cmp;
      if ((int)ok != tr.success) { ++bad; first = first ? first : "success"; }
      for (size_t h = 0; h < c.heads.size(); ++h) {
        ++n_cmp;
        if ((head_val[h] ? '1' : '0') != tr.heads[h]) { ++bad; first = first ? first : "head"; }
      }
      const double q = q_score(ok, c.opt_start.data(), n_opts, c.opt_lits.data(), now.data(), init.data());
      uint64_t qb;
      memcpy(&qb, &q, 8);
      ++n_cmp;
      if (qb != tr.q_bits) {
        ++bad;
        first = first ? first : "q_score";
      }
    }
    t_eval += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    n_bad += bad;
    if (bad) printf("%-45s 원자 %3zu HEAD %2zu 선택지 %7d 시행 %3zu  다름 %lld (첫: %s)\n", name.c_str(), c.atoms.size(),
                    c.heads.size(), n_opts, r.trials.size(), bad, first);
  }
  printf("과제 %lld, 비교 %lld, 다름 %lld, ground option 합 %lld, 판정+q_score 시간 %.3f s\n", n_tasks, n_cmp, n_bad,
         n_opts_total, t_eval);
  return n_bad ? 1 : 0;
}
