// BDDL 목표 판정 (손으로 짬). 공식 bddl3 (BEHAVIOR-1K/bddl3/bddl) 과 참/거짓·q_score 가 완전히 같게 옮겼다.
//   원본: parsing.py:32 scan_tokens, :222 parse_problem, :309 package_predicates
//         condition_evaluation.py (Conjunction·Disjunction·Universal·Existential·NQuantifier·ForPairs·ForNPairs·
//         Negation·Implication·HEAD, get_ground_state_options:741), predicates.py:62 Predicate
//         OmniGibson metrics/task_metric.py:6 compute_q_score, tasks/behavior_task.py:176 get_goal_option_satisfaction
// 구조: 문자열 파싱·컴파일(양화사 펼치기)·ground option 만들기는 과제를 불러올 때 한 번(호스트, std 컨테이너).
//       판정(매 스텝)은 평평한 배열 프로그램(자식이 부모보다 앞, 후위 순서)을 한 줄로 훑는 EHD 함수 → 층 2(판 N 개) 그대로.
// 파이썬과 다른 점(결과는 같음): 파이썬 최상위 scope 는 set 이라 양화사 자식 순서가 PYTHONHASHSEED 따라 바뀐다.
//   all/any/개수/ForPairs 의 행·열 "any 개수" 는 순서와 무관하고, ground option 은 집합이 같아 q_score(최댓값)도 같다.
//   우리는 :objects 선언 순서로 돈다(결정적).
#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__CUDACC__)
#define OEHD __host__ __device__ __forceinline__
#else
#define OEHD inline
#endif

namespace eng {
namespace omni {
namespace bddl {

// ---------------- S-식 (parsing.py:32 scan_tokens) ----------------
struct SExp {
  bool is_list = false;
  std::string tok;            // is_list == false 일 때
  std::vector<SExp> items;    // is_list == true 일 때
};

// ';' 부터 줄 끝까지 지우고, 전부 소문자로, "(" ")" 와 공백 아닌 덩어리로 자른다.
inline SExp scan_tokens(const std::string& raw) {
  std::string s;
  s.reserve(raw.size());
  bool comment = false;
  for (char ch : raw) {
    if (ch == '\n') comment = false;
    if (ch == ';') comment = true;
    if (!comment) s.push_back(ch);
  }
  // str.lower(): 여기 나오는 문자는 ASCII 뿐 (BDDL 이름)
  for (char& ch : s) ch = (char)std::tolower((unsigned char)ch);
  std::vector<std::vector<SExp>> stack;
  std::vector<SExp> tokens;
  size_t i = 0, n = s.size();
  while (i < n) {
    const char c = s[i];
    if (c == '(') {
      stack.push_back(std::move(tokens));
      tokens.clear();
      ++i;
    } else if (c == ')') {
      if (stack.empty()) throw std::runtime_error("Missing open parenthesis");
      SExp lst;
      lst.is_list = true;
      lst.items = std::move(tokens);
      tokens = std::move(stack.back());
      stack.pop_back();
      tokens.push_back(std::move(lst));
      ++i;
    } else if (std::isspace((unsigned char)c)) {
      ++i;
    } else {
      size_t j = i;
      while (j < n && s[j] != '(' && s[j] != ')' && !std::isspace((unsigned char)s[j])) ++j;
      SExp t;
      t.tok = s.substr(i, j - i);
      tokens.push_back(std::move(t));
      i = j;
    }
  }
  if (!stack.empty()) throw std::runtime_error("Missing close parenthesis");
  if (tokens.size() != 1) throw std::runtime_error("Malformed expression");
  return tokens[0];
}

// ---------------- 문제 파싱 (parsing.py:222 parse_problem) ----------------
struct Problem {
  std::string name = "unknown";
  // objects: 파이썬 dict (삽입 순서). 같은 범주가 다시 나오면 값만 바뀌고 자리는 그대로.
  std::vector<std::string> categories;
  std::map<std::string, std::vector<std::string>> objects;
  std::vector<SExp> init;
  std::vector<SExp> goal;  // package_predicates: 최상위 and 를 벗긴 조건 목록
};

inline Problem parse_problem(const std::string& text) {
  SExp top = scan_tokens(text);
  if (!top.is_list || top.items.empty() || top.items[0].is_list || top.items[0].tok != "define")
    throw std::runtime_error("problem does not match problem pattern");
  Problem p;
  std::vector<SExp> toks(top.items.begin() + 1, top.items.end());
  while (!toks.empty()) {  // tokens.pop(): 뒤에서부터
    SExp group = std::move(toks.back());
    toks.pop_back();
    if (!group.is_list || group.items.empty()) continue;
    const std::string t = group.items[0].is_list ? std::string() : group.items[0].tok;
    if (t == "problem") {
      p.name = group.items.back().tok;
    } else if (t == ":objects") {
      std::vector<std::string> object_list;
      size_t k = 1;
      auto put = [&](const std::string& cat, std::vector<std::string> v) {
        if (!p.objects.count(cat)) p.categories.push_back(cat);
        p.objects[cat] = std::move(v);
      };
      while (k < group.items.size()) {
        if (group.items[k].tok == "-") {
          put(group.items[k + 1].tok, object_list);
          object_list.clear();
          k += 2;
        } else {
          object_list.push_back(group.items[k].tok);
          ++k;
        }
      }
      if (!object_list.empty()) {
        if (!p.objects.count("object")) { p.categories.push_back("object"); p.objects["object"] = {}; }
        auto& o = p.objects["object"];
        o.insert(o.end(), object_list.begin(), object_list.end());
      }
    } else if (t == ":init") {
      p.init.assign(group.items.begin() + 1, group.items.end());
    } else if (t == ":goal") {
      const SExp& g = group.items[1];
      if (!g.is_list) throw std::runtime_error("Error with goals");
      if (!g.items.empty() && !g.items[0].is_list && g.items[0].tok == "and")
        p.goal.assign(g.items.begin() + 1, g.items.end());
      else
        p.goal.push_back(g);
    }
  }
  return p;
}

// ---------------- 평평한 판정 프로그램 ----------------
enum Op : uint8_t { OP_ATOM = 0, OP_AND = 1, OP_OR = 2, OP_NOT = 3, OP_IMPLY = 4, OP_COUNT_EQ = 5, OP_PAIRS = 6 };

// 노드는 후위 순서(자식이 먼저). 자식 목록은 kids[start .. start+count).
// OP_PAIRS: 자식이 rows x cols 행 우선, cols = count / rows. need = L(ForPairs) 또는 N(ForNPairs).
struct Node {
  uint8_t op;
  int32_t arg;    // ATOM: 원자 번호 / COUNT_EQ: N / PAIRS: need (<0 이면 L=min(rows, cols))
  int32_t start;  // kids 시작
  int32_t count;  // 자식 수
  int32_t rows;   // PAIRS 행 수
};

struct Atom {
  std::string pred;               // 토큰 (ontop, inside, ...)
  std::vector<std::string> args;  // 원자 인자 (범주 변수는 이미 풀림)
  bool operator<(const Atom& o) const { return pred != o.pred ? pred < o.pred : args < o.args; }
  bool operator==(const Atom& o) const { return pred == o.pred && args == o.args; }
};

// 리터럴: 원자 + 부정 겹수 (Negation 의 ground option 은 이미 not 인 조건에 not 을 또 씌운다 → 겹수 2 가능)
struct Lit {
  int32_t atom;
  int32_t neg;
  bool operator<(const Lit& o) const { return neg != o.neg ? neg < o.neg : atom < o.atom; }  // 파이썬 (neg, id) 순서
  bool operator==(const Lit& o) const { return atom == o.atom && neg == o.neg; }
};
using Option = std::vector<Lit>;

struct Compiled {
  std::vector<Atom> atoms;
  std::vector<Node> nodes;
  std::vector<int32_t> kids;
  std::vector<int32_t> heads;  // 목표 조건(HEAD)마다 뿌리 노드 번호
  // ground option: CSR (opt_start[i] .. opt_start[i+1])
  std::vector<int32_t> opt_start;
  std::vector<Lit> opt_lits;
};

// ---------------- 컴파일 (condition_evaluation.py) ----------------
class Compiler {
 public:
  Compiler(const Problem& p) : P(p) {
    for (const auto& c : P.categories)
      for (const auto& inst : P.objects.at(c)) instances.push_back(inst);
    // create_scope 는 set: 같은 이름이 두 범주에 있으면 한 번만
    std::vector<std::string> uniq;
    for (auto& s : instances)
      if (std::find(uniq.begin(), uniq.end(), s) == uniq.end()) uniq.push_back(s);
    instances.swap(uniq);
  }

  Compiled compile_goal() {
    Compiled out;
    C = &out;
    std::vector<std::vector<Option>> head_opts;
    for (const SExp& cond : P.goal) {
      Bindings b;
      std::vector<Option> opts;
      const int root = compile(cond, b, &opts);
      out.heads.push_back(root);
      head_opts.push_back(std::move(opts));
    }
    ground(head_opts, out);
    C = nullptr;
    return out;
  }

 private:
  using Bindings = std::vector<std::pair<std::string, std::string>>;  // 양화사 변수 -> 인스턴스
  const Problem& P;
  std::vector<std::string> instances;
  Compiled* C = nullptr;
  std::map<Atom, int32_t> atom_ids;

  static std::string strip_q(const std::string& s) {  // str.strip("?")
    size_t a = 0, b = s.size();
    while (a < b && s[a] == '?') ++a;
    while (b > a && s[b - 1] == '?') --b;
    return s.substr(a, b - a);
  }
  const std::vector<std::string>& cat(const std::string& c) const {
    auto it = P.objects.find(c);
    if (it == P.objects.end()) throw std::runtime_error("unknown category " + c);  // 파이썬 KeyError
    return it->second;
  }
  bool in_cat(const std::string& c, const std::string& inst) const {
    const auto& v = cat(c);
    return std::find(v.begin(), v.end(), inst) != v.end();
  }
  int32_t atom_id(const Atom& a) {
    auto it = atom_ids.find(a);
    if (it != atom_ids.end()) return it->second;
    const int32_t id = (int32_t)C->atoms.size();
    C->atoms.push_back(a);
    atom_ids[a] = id;
    return id;
  }
  int push(Node n) {
    C->nodes.push_back(n);
    return (int)C->nodes.size() - 1;
  }
  int push_list(uint8_t op, int32_t arg, const std::vector<int>& ch, int32_t rows = 0) {
    Node n{op, arg, (int32_t)C->kids.size(), (int32_t)ch.size(), rows};
    for (int c : ch) C->kids.push_back(c);
    return push(n);
  }

  // ground option 조합 도우미
  static std::vector<Option> product_chain(const std::vector<const std::vector<Option>*>& lists) {
    // itertools.product(*lists) 후 각 조합을 이어 붙임 (첫 목록이 가장 느리게 바뀜)
    std::vector<Option> out;
    for (auto* l : lists)
      if (l->empty()) return out;
    std::vector<size_t> idx(lists.size(), 0);
    if (lists.empty()) { out.push_back({}); return out; }
    while (true) {
      Option o;
      for (size_t k = 0; k < lists.size(); ++k) {
        const Option& part = (*lists[k])[idx[k]];
        o.insert(o.end(), part.begin(), part.end());
      }
      out.push_back(std::move(o));
      int k = (int)lists.size() - 1;
      while (k >= 0) {
        if (++idx[k] < lists[k]->size()) break;
        idx[k] = 0;
        --k;
      }
      if (k < 0) break;
    }
    return out;
  }
  static std::vector<Option> negate(const std::vector<Option>& child) {
    // Negation.get_ground_options: 각 선택지의 조건마다 not 을 씌우고, 선택지마다 하나씩 고르는 곱
    std::vector<Option> neg_lists;
    for (const Option& o : child) {
      Option n = o;
      for (Lit& l : n) l.neg += 1;
      neg_lists.push_back(std::move(n));
    }
    std::vector<Option> out;
    for (auto& l : neg_lists)
      if (l.empty()) return out;  // product 에 빈 목록 -> 결과 없음
    std::vector<size_t> idx(neg_lists.size(), 0);
    if (neg_lists.empty()) { out.push_back({}); return out; }
    while (true) {
      Option o;
      for (size_t k = 0; k < neg_lists.size(); ++k) o.push_back(neg_lists[k][idx[k]]);
      out.push_back(std::move(o));
      int k = (int)neg_lists.size() - 1;
      while (k >= 0) {
        if (++idx[k] < neg_lists[k].size()) break;
        idx[k] = 0;
        --k;
      }
      if (k < 0) break;
    }
    return out;
  }
  // itertools.permutations(range(n), r) 순서
  static void permutations(int n, int r, std::vector<std::vector<int>>& out) {
    std::vector<int> cur;
    std::vector<char> used(n, 0);
    std::vector<std::vector<int>> res;
    struct R {
      static void go(int n, int r, std::vector<int>& cur, std::vector<char>& used, std::vector<std::vector<int>>& res) {
        if ((int)cur.size() == r) { res.push_back(cur); return; }
        for (int i = 0; i < n; ++i) {
          if (used[i]) continue;
          used[i] = 1; cur.push_back(i);
          go(n, r, cur, used, res);
          cur.pop_back(); used[i] = 0;
        }
      }
    };
    if (r <= n) R::go(n, r, cur, used, res);
    out.swap(res);
  }

  std::string resolve(const std::string& raw, const Bindings& b) const {
    const std::string s = strip_q(raw);
    for (auto it = b.rbegin(); it != b.rend(); ++it)
      if (it->first == s) return it->second;
    return s;
  }

  // 한 조건을 컴파일: 노드 번호를 돌려주고, opts 에 ground option 을 채운다.
  int compile(const SExp& e, const Bindings& b, std::vector<Option>* opts) {
    if (!e.is_list || e.items.empty() || e.items[0].is_list) throw std::runtime_error("bad condition");
    const std::string& t = e.items[0].tok;
    const size_t nb = e.items.size() - 1;  // body 길이
    auto body = [&](size_t i) -> const SExp& { return e.items[1 + i]; };

    if (t == "and" || t == "or") {
      std::vector<int> ch;
      std::vector<std::vector<Option>> co(nb);
      for (size_t i = 0; i < nb; ++i) ch.push_back(compile(body(i), b, &co[i]));
      if (t == "and") {
        std::vector<const std::vector<Option>*> l;
        for (auto& x : co) l.push_back(&x);
        *opts = product_chain(l);
      } else {
        opts->clear();
        for (auto& x : co) opts->insert(opts->end(), x.begin(), x.end());
      }
      return push_list(t == "and" ? OP_AND : OP_OR, 0, ch);
    }
    if (t == "forall" || t == "exists" || t == "forn") {
      const bool is_n = (t == "forn");
      const SExp& iterable = body(is_n ? 1 : 0);
      const SExp& sub = body(is_n ? 2 : 1);
      const std::string label = strip_q(iterable.items[0].tok);
      if (iterable.items[1].tok != "-") throw std::runtime_error("Middle was not a hyphen");
      const std::string& category = iterable.items[2].tok;
      std::vector<int> ch;
      std::vector<std::vector<Option>> co;
      for (const std::string& inst : instances) {
        if (!in_cat(category, inst)) continue;
        Bindings nbind = b;
        nbind.emplace_back(label, inst);
        co.emplace_back();
        ch.push_back(compile(sub, nbind, &co.back()));
      }
      if (t == "forall") {
        std::vector<const std::vector<Option>*> l;
        for (auto& x : co) l.push_back(&x);
        *opts = product_chain(l);
        return push_list(OP_AND, 0, ch);
      }
      if (t == "exists") {
        opts->clear();
        for (auto& x : co) opts->insert(opts->end(), x.begin(), x.end());
        return push_list(OP_OR, 0, ch);
      }
      // forn: body = [[N], iterable, sub], N = int(N[0])
      const int N = std::stoi(body(0).items[0].tok);
      // get_ground_options: product 의 각 조합 option(튜플) 에서 combinations(option, N)
      std::vector<std::vector<size_t>> prod;
      {
        std::vector<size_t> idx(co.size(), 0);
        bool empty = false;
        for (auto& x : co) if (x.empty()) empty = true;
        if (!empty) {
          while (true) {
            prod.push_back(idx);
            int k = (int)co.size() - 1;
            while (k >= 0) {
              if (++idx[k] < co[k].size()) break;
              idx[k] = 0;
              --k;
            }
            if (k < 0) break;
          }
        }
      }
      opts->clear();
      for (auto& pi : prod) {
        // itertools.combinations(range(len), N) 사전순
        const int m = (int)pi.size();
        if (N > m) continue;
        std::vector<int> c(N);
        for (int i = 0; i < N; ++i) c[i] = i;
        while (true) {
          Option o;
          for (int i = 0; i < N; ++i) {
            const Option& part = co[c[i]][pi[c[i]]];
            o.insert(o.end(), part.begin(), part.end());
          }
          opts->push_back(std::move(o));
          int i = N - 1;
          while (i >= 0 && c[i] == i + m - N) --i;
          if (i < 0) break;
          ++c[i];
          for (int j = i + 1; j < N; ++j) c[j] = c[j - 1] + 1;
        }
      }
      return push_list(OP_COUNT_EQ, N, ch);
    }
    if (t == "forpairs" || t == "fornpairs") {
      const bool is_n = (t == "fornpairs");
      const SExp& it1 = body(is_n ? 1 : 0);
      const SExp& it2 = body(is_n ? 2 : 1);
      const SExp& sub = body(is_n ? 3 : 2);
      const std::string l1 = strip_q(it1.items[0].tok), l2 = strip_q(it2.items[0].tok);
      const std::string& c1 = it1.items[2].tok;
      const std::string& c2 = it2.items[2].tok;
      std::vector<std::vector<int>> grid;
      std::vector<std::vector<std::vector<Option>>> gopt;
      for (const std::string& o1 : instances) {
        if (!in_cat(c1, o1)) continue;
        grid.emplace_back();
        gopt.emplace_back();
        for (const std::string& o2 : instances) {
          if (!in_cat(c2, o2) || o1 == o2) continue;
          Bindings nbind = b;
          nbind.emplace_back(l1, o1);
          nbind.emplace_back(l2, o2);
          gopt.back().emplace_back();
          grid.back().push_back(compile(sub, nbind, &gopt.back().back()));
        }
      }
      const int M = (int)grid.size();
      if (M == 0) throw std::runtime_error("forpairs with no rows (python IndexError)");
      const int Nc = (int)grid[0].size();
      for (auto& r : grid)
        if ((int)r.size() != Nc) throw std::runtime_error("ragged forpairs (numpy ValueError)");
      std::vector<int> ch;
      for (auto& r : grid) ch.insert(ch.end(), r.begin(), r.end());
      opts->clear();
      if (!is_n) {
        // ForPairs.get_ground_options: all_G_choices 반복자가 첫 lchoice 에서 다 소모된다 → lchoice = (0..L-1) 하나뿐
        const int L = std::min(M, Nc), G = std::max(M, Nc);
        std::vector<std::vector<int>> gch;
        permutations(G, L, gch);
        for (auto& g : gch) {
          std::vector<const std::vector<Option>*> l;
          for (int k = 0; k < L; ++k) l.push_back(M < Nc ? &gopt[k][g[k]] : &gopt[g[k]][k]);
          auto part = product_chain(l);
          opts->insert(opts->end(), part.begin(), part.end());
        }
        return push_list(OP_PAIRS, -1, ch, M);
      }
      const int N = std::stoi(body(0).items[0].tok);
      if (N > std::min(M, Nc)) throw std::runtime_error("ForNPairs asks for more pairs than instances available");
      // ForNPairs.get_ground_options: all_Q_choices 가 첫 pchoice 에서 소모 → pchoice = (0..N-1) 하나뿐
      std::vector<std::vector<int>> qch;
      permutations(Nc, N, qch);
      for (auto& q : qch) {
        std::vector<const std::vector<Option>*> l;
        for (int k = 0; k < N; ++k) l.push_back(&gopt[k][q[k]]);
        auto part = product_chain(l);
        opts->insert(opts->end(), part.begin(), part.end());
      }
      return push_list(OP_PAIRS, N, ch, M);
    }
    if (t == "not") {
      std::vector<Option> co;
      const int c = compile(body(0), b, &co);
      *opts = negate(co);
      return push_list(OP_NOT, 0, {c});
    }
    if (t == "imply") {
      std::vector<Option> ca, cc;
      const int a = compile(body(0), b, &ca);
      const int c = compile(body(1), b, &cc);
      *opts = negate(ca);
      opts->insert(opts->end(), cc.begin(), cc.end());
      return push_list(OP_IMPLY, 0, {a, c});
    }
    // 술어 (predicates.py:62): 인자 ? 벗기고 양화사 변수면 묶인 인스턴스로
    Atom at;
    at.pred = t;
    for (size_t i = 0; i < nb; ++i) at.args.push_back(resolve(body(i).tok, b));
    const int32_t id = atom_id(at);
    opts->assign(1, Option{Lit{id, 0}});
    return push(Node{OP_ATOM, id, 0, 0, 0});
  }

  // get_ground_state_options (condition_evaluation.py:741)
  void ground(const std::vector<std::vector<Option>>& head_opts, Compiled& out) {
    std::vector<const std::vector<Option>*> l;
    for (auto& h : head_opts) l.push_back(&h);
    std::vector<Option> all = product_chain(l);
    std::vector<Option> ok;
    for (Option& o : all) {
      bool consistent = true;
      for (size_t i = 0; i < o.size() && consistent; ++i)
        for (size_t j = i + 1; j < o.size(); ++j)
          if (o[i].atom == o[j].atom && (o[i].neg == o[j].neg + 1 || o[j].neg == o[i].neg + 1)) {
            consistent = false;
            break;
          }
      if (consistent) ok.push_back(std::move(o));
    }
    std::stable_sort(ok.begin(), ok.end(), [](const Option& a, const Option& b) { return a.size() < b.size(); });
    out.opt_start.assign(1, 0);
    for (auto& o : ok) {
      out.opt_lits.insert(out.opt_lits.end(), o.begin(), o.end());
      out.opt_start.push_back((int32_t)out.opt_lits.size());
    }
  }
};

inline Compiled compile_problem(const Problem& p) { return Compiler(p).compile_goal(); }

// ---------------- 판정 (매 스텝, EHD) ----------------
// atom_val[원자] (0/1) → node_val[노드]. 후위 순서라 앞에서부터 한 번 훑으면 된다.
OEHD void eval_nodes(const Node* nodes, int n_nodes, const int32_t* kids, const uint8_t* atom_val, uint8_t* node_val) {
  for (int i = 0; i < n_nodes; ++i) {
    const Node nd = nodes[i];
    uint8_t v = 0;
    switch (nd.op) {
      case OP_ATOM: v = atom_val[nd.arg] ? 1 : 0; break;
      case OP_AND: {  // all([]) = True
        v = 1;
        for (int k = 0; k < nd.count; ++k) v &= node_val[kids[nd.start + k]];
      } break;
      case OP_OR: {  // any([]) = False
        v = 0;
        for (int k = 0; k < nd.count; ++k) v |= node_val[kids[nd.start + k]];
      } break;
      case OP_NOT: v = node_val[kids[nd.start]] ? 0 : 1; break;
      case OP_IMPLY: v = (!node_val[kids[nd.start]]) || node_val[kids[nd.start + 1]]; break;
      case OP_COUNT_EQ: {  // sum(child_values) == N
        int s = 0;
        for (int k = 0; k < nd.count; ++k) s += node_val[kids[nd.start + k]];
        v = (s == nd.arg);
      } break;
      case OP_PAIRS: {  // ForPairs / ForNPairs.evaluate: 행 any 개수 >= need 그리고 열 any 개수 >= need (열은 위치 기준)
        const int R = nd.rows, Cn = nd.rows ? nd.count / nd.rows : 0;
        const int need = nd.arg >= 0 ? nd.arg : (R < Cn ? R : Cn);
        int rows_any = 0, cols_any = 0;
        for (int r = 0; r < R; ++r) {
          int a = 0;
          for (int c = 0; c < Cn; ++c) a |= node_val[kids[nd.start + r * Cn + c]];
          rows_any += a;
        }
        for (int c = 0; c < Cn; ++c) {
          int a = 0;
          for (int r = 0; r < R; ++r) a |= node_val[kids[nd.start + r * Cn + c]];
          cols_any += a;
        }
        v = (rows_any >= need) && (cols_any >= need);
      } break;
    }
    node_val[i] = v;
  }
}

// evaluate_state: 모든 HEAD 가 참이면 성공. head_val 에 HEAD 마다 결과.
OEHD bool eval_goal(const Node* nodes, int n_nodes, const int32_t* kids, const int32_t* heads, int n_heads,
                    const uint8_t* atom_val, uint8_t* node_val, uint8_t* head_val) {
  eval_nodes(nodes, n_nodes, kids, atom_val, node_val);
  bool all = true;
  for (int h = 0; h < n_heads; ++h) {
    head_val[h] = node_val[heads[h]];
    all = all && head_val[h];
  }
  return all;
}

OEHD bool lit_val(const Lit& l, const uint8_t* atom_val) { return (atom_val[l.atom] != 0) != ((l.neg & 1) != 0); }

// compute_q_score (task_metric.py:6): 성공이면 1.0, 아니면 선택지마다 (처음 거짓 → 지금 참) 개수 / 길이 의 최댓값.
// 나눗셈은 파이썬 int/int (double 올바른 반올림) 과 같다.
OEHD double q_score(bool success, const int32_t* opt_start, int n_opts, const Lit* lits, const uint8_t* now_atoms,
                    const uint8_t* init_atoms) {
  if (success) return 1.0;
  if (n_opts == 0) return 0.0;
  double best = 0.0;
  bool first = true;
  for (int o = 0; o < n_opts; ++o) {
    const int a = opt_start[o], b = opt_start[o + 1];
    double s;
    if (b == a) {
      s = 0.0;
    } else {
      long long newly = 0;
      for (int k = a; k < b; ++k) newly += (!lit_val(lits[k], init_atoms)) && lit_val(lits[k], now_atoms);
      s = (double)newly / (double)(b - a);
    }
    if (first || s > best) best = s;  // max(): 같으면 앞의 것 (값이 같으니 결과 같음)
    first = false;
  }
  return best;
}

}  // namespace bddl
}  // namespace omni
}  // namespace eng
