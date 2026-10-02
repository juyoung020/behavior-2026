// Tokenizer check against Python sentencepiece cases (tools/make_tokenizer_cases.py).
// usage: tok_test <weights.pi05w | tokenizer.model> <cases.tsv>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/tokenizer.h"
#include "../src/weights.h"

static std::string unhex(const std::string& h) {
  std::string s;
  for (size_t i = 0; i + 1 < h.size(); i += 2) s += char(std::stoi(h.substr(i, 2), nullptr, 16));
  return s;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: tok_test <weights.pi05w|tokenizer.model> <cases.tsv>\n"); return 2; }
  std::string path = argv[1], err;
  std::vector<uint8_t> model;
  pi05::WeightFile wf;
  if (path.size() > 6 && path.substr(path.size() - 6) == ".pi05w") {
    if (!wf.open(path, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const pi05::TensorInfo* t = wf.find("tokenizer.model");
    if (!t) { fprintf(stderr, "no tokenizer.model in %s\n", path.c_str()); return 1; }
    model.resize(t->nbytes);
    if (!wf.read(*t, model.data(), &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  } else {
    std::ifstream f(path, std::ios::binary);
    model.assign(std::istreambuf_iterator<char>(f), {});
  }
  pi05::Tokenizer tok;
  if (!tok.load(model.data(), model.size(), &err)) { fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
  std::ifstream in(argv[2]);
  std::string line;
  int n = 0, bad = 0;
  double us = 0;
  while (std::getline(in, line)) {
    size_t tab = line.find('\t');
    std::string text = unhex(line.substr(0, tab));
    std::vector<int> want;
    std::istringstream ss(line.substr(tab + 1));
    for (int v; ss >> v;) want.push_back(v);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<int> got = tok.encode(text, true);
    us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    ++n;
    if (got != want) {
      if (bad++ < 5) {
        fprintf(stderr, "MISMATCH case %d text=%s\n  want:", n, line.substr(0, tab).c_str());
        for (int v : want) fprintf(stderr, " %d", v);
        fprintf(stderr, "\n  got: ");
        for (int v : got) fprintf(stderr, " %d", v);
        fprintf(stderr, "\n");
      }
    }
  }
  printf("tokenizer: %d cases, %d mismatches, %.1f us/encode\n", n, bad, us / (n ? n : 1));
  return bad ? 1 : 0;
}
