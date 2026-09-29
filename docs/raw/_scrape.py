"""BEHAVIOR Challenge 2026 원문 보관본 스크래퍼.

한 번 실행으로 대회 사이트(behavior.stanford.edu/challenge/ 전체 + 본문이 가리키는 관련 페이지),
GitHub README, HuggingFace 데이터셋 카드·메타, 리더보드(HF Space), 등록 폼 문항, 관련 논문 초록을 받아
이 폴더(docs/raw/)에 페이지당 md 파일 1개로 저장한다. 각 파일 맨 위에 원본 URL 과 받은 시각을 적는다.

    python _scrape.py            # 다시 돌리면 같은 이름으로 덮어쓴다 (공지 갱신 확인용)

대용량(데모·에셋)은 받지 않는다. 받는 것은 전부 수십~수백 KB 짜리 텍스트다.
"""

import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

OUT = Path(__file__).resolve().parent
KST = timezone(timedelta(hours=9))
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0 (behavior-2026-docs-archiver/1.0; personal study)"
SITE = "https://behavior.stanford.edu/"
CHALLENGE = SITE + "challenge/"
LB_SPACE = "https://behavior-1k-2026-challenge-leaderboard.hf.space"
INDEX = []  # (file, url, status, note)


# ---------------------------------------------------------------- 공통
def now_str():
    t = datetime.now(KST)
    return f"{t:%Y-%m-%d %H:%M} KST ({t.astimezone(timezone.utc):%Y-%m-%d %H:%M} UTC)"


def get(url, **kw):
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=60, **kw)
            time.sleep(0.3)
            return r
        except requests.RequestException as e:
            err = e
            time.sleep(2)
    raise err


def save(name, url, title, body, note=""):
    header = [f"# {title}", "", f"> 원본: {url}", f"> 받은 시각: {now_str()}"]
    if note:
        header.append(f"> 비고: {note}")
    header += [
        "> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.",
        "",
        "---",
        "",
    ]
    path = OUT / name
    path.write_text("\n".join(header) + body.strip() + "\n", encoding="utf-8")
    INDEX.append((name, url, "ok", title))
    print("saved", name)


def fail(name, url, why):
    INDEX.append((name, url, "실패", why))
    print("FAIL", name, url, why, file=sys.stderr)


def slug(s):
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"\.html?$", "", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s


# ---------------------------------------------------------------- HTML -> Markdown
SKIP_TAGS = {"script", "style", "nav", "button", "svg", "noscript", "header", "footer", "form", "input", "select"}


def _inline(node, base):
    return _md(node, base, inline=True)


def _collapse(text):
    return re.sub(r"[ \t\r\n]+", " ", text)


def _table(node, base):
    rows = []
    for tr in node.find_all("tr"):
        cells = [
            _collapse(_inline(td, base)).strip().replace("|", "\\|") or " " for td in tr.find_all(["td", "th"])
        ]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [" "] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n\n" + "\n".join(out) + "\n\n"


def _list(node, base, depth):
    out = []
    ordered = node.name == "ol"
    for i, li in enumerate(node.find_all("li", recursive=False), 1):
        bullet = f"{i}." if ordered else "-"
        parts, nested = [], []
        for c in li.children:
            if isinstance(c, Tag) and c.name in ("ul", "ol"):
                nested.append(_list(c, base, depth + 1))
            else:
                parts.append(_md(c, base, inline=False) if isinstance(c, Tag) else _collapse(str(c)))
        text = re.sub(r"\n{2,}", "\n", "".join(parts)).strip()
        text = text.replace("\n", "\n" + "  " * (depth + 1))
        out.append("  " * depth + f"{bullet} {text}")
        out += [n.rstrip("\n") for n in nested]
    return ("\n" if depth else "\n\n") + "\n".join(out) + ("\n" if depth else "\n\n")


def _md(node, base, inline=False):
    if isinstance(node, NavigableString):
        if node.__class__.__name__ in ("Comment", "Doctype"):
            return ""
        return _collapse(str(node))
    if not isinstance(node, Tag):
        return ""
    name = node.name
    cls = " ".join(node.get("class") or [])
    if name in SKIP_TAGS:
        return ""
    if name == "a" and ("headerlink" in cls or (node.get("href") or "").startswith("#__codelineno")):
        return ""
    if "md-clipboard" in cls or "linenos" in cls or "md-source-file" in cls:
        return ""
    kids = lambda: "".join(_md(c, base, inline) for c in node.children)  # noqa: E731
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return f"\n\n{'#' * int(name[1])} {_collapse(kids()).strip()}\n\n"
    if name == "p":
        return f"\n\n{kids().strip()}\n\n" if not inline else kids()
    if name == "br":
        return "  \n"
    if name == "hr":
        return "\n\n---\n\n"
    if name in ("strong", "b"):
        t = kids().strip()
        return f"**{t}**" if t else ""
    if name in ("em", "i"):
        t = kids().strip()
        return f"*{t}*" if t else ""
    if name == "code" and node.find_parent("pre") is None:
        t = node.get_text()
        return f"`{t}`" if "`" not in t else f"`` {t} ``"
    if name == "pre":
        code = node.find("code") or node
        lang = ""
        for c in (code.get("class") or []) + (node.parent.get("class") or []):
            if c.startswith("language-"):
                lang = c[9:]
        text = code.get_text().rstrip("\n")
        return f"\n\n```{lang}\n{text}\n```\n\n"
    if name in ("ul", "ol"):
        return _list(node, base, 0)
    if name == "table":
        return _table(node, base)
    if name == "img":
        src = node.get("src") or ""
        return f"![{node.get('alt', '')}]({urljoin(base, src)})"
    if name == "a":
        href = node.get("href") or ""
        t = _collapse(kids()).strip()
        if not t:
            return ""
        if not href or href.startswith("#") or href.startswith("javascript"):
            return t
        return f"[{t}]({urljoin(base, href)})"
    if name == "details" or (name == "div" and "admonition" in cls):
        title_el = node.find("summary") if name == "details" else node.find(class_="admonition-title")
        title = _collapse(title_el.get_text()).strip() if title_el else ""
        body = "".join(_md(c, base) for c in node.children if c is not title_el).strip()
        quoted = "\n".join("> " + ln if ln.strip() else ">" for ln in body.splitlines())
        return f"\n\n> **{title}**\n>\n{quoted}\n\n"
    if name == "div" and "tabbed-set" in cls:
        labels = [_collapse(lb.get_text()).strip() for lb in node.select(".tabbed-labels label")]
        blocks = node.select(".tabbed-content > .tabbed-block")
        out = []
        for i, blk in enumerate(blocks):
            lab = labels[i] if i < len(labels) else f"탭 {i + 1}"
            out.append(f"\n\n**[탭: {lab}]**\n\n" + "".join(_md(c, base) for c in blk.children))
        return "".join(out)
    if name in ("div", "section", "article", "main", "span", "label", "summary", "figure", "figcaption", "dl", "dd", "dt", "tbody", "thead", "sup", "sub", "small", "u", "center", "font", "abbr", "cite", "blockquote", "li", "tr", "td", "th"):
        inner = kids()
        if name in ("div", "section", "article", "main", "figure", "dl", "dd", "dt", "blockquote") and not inline:
            return f"\n{inner}\n"
        return inner
    return kids()


def html_to_md(html, base, selector=None):
    soup = BeautifulSoup(html, "lxml")
    root = None
    if selector:
        root = soup.select_one(selector)
    if root is None:
        root = soup.select_one("article.md-content__inner") or soup.find("article") or soup.find("main") or soup.body
    md = _md(root, base)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    title = soup.title.get_text().strip() if soup.title else base
    return title, md.strip(), soup, root


# ---------------------------------------------------------------- 1. 대회 사이트 크롤
def gallery_tasks_md(html):
    i = html.find("const tasks = [")
    if i < 0:
        return ""
    i += len("const tasks = ")
    depth = 0
    for j in range(i, len(html)):
        if html[j] == "[":
            depth += 1
        elif html[j] == "]":
            depth -= 1
            if depth == 0:
                break
    tasks = json.loads(html[i : j + 1])
    lines = [
        "",
        "## (페이지 내장 데이터) 과제 목록",
        "",
        "갤러리 페이지는 JavaScript 로 아래 데이터를 그린다. 페이지 HTML 에 들어 있는 `const tasks = [...]` 를 그대로 표로 옮긴 것.",
        "duration 단위는 페이지에 명시되지 않음(초로 보임).",
        "",
        "| # | id | name | rooms | duration | scene_model | instruction | video |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n, t in enumerate(tasks):
        lines.append(
            f"| {n} | `{t.get('id')}` | {t.get('name')} | {', '.join(t.get('rooms', []))} | {t.get('duration')} | "
            f"{t.get('scene_model')} | {(t.get('instruction') or '').replace('|', '/')} | {t.get('video', '')} |"
        )
    return "\n".join(lines)


def crawl_site():
    seen, queue, related = set(), [CHALLENGE + "index.html"], set()
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        r = get(url)
        name = "site_" + slug(url.replace(SITE, "")) + ".md"
        if r.status_code != 200:
            fail(name, url, f"HTTP {r.status_code}")
            continue
        r.encoding = "utf-8"
        title, md, soup, root = html_to_md(r.text, url)
        if url.endswith("tasks/index.html"):
            md += "\n\n" + gallery_tasks_md(r.text)
        save(name, url, title, md)
        # 사이트 전체 링크 중 challenge/ 아래는 모두 따라간다 (내비게이션 포함)
        for a in soup.find_all("a", href=True):
            u = urldefrag(urljoin(url, a["href"]))[0]
            if u.startswith(CHALLENGE) and u.endswith(".html") and u not in seen:
                queue.append(u)
        # 본문(article)이 가리키는 challenge 밖 사이트 페이지는 한 단계만 받는다
        for a in root.find_all("a", href=True):
            u = urldefrag(urljoin(url, a["href"]))[0]
            if u.startswith(SITE) and u.endswith(".html") and not u.startswith(CHALLENGE):
                related.add(u)
    # 대회 준비에 직접 쓰이는 사이트 문서(FAQ·알려진 문제·설치·로봇·컨트롤러 등)
    related |= {
        SITE + p
        for p in (
            "other/faq.html",
            "other/known_issues.html",
            "getting_started/installation.html",
            "getting_started/quickstart.html",
            "getting_started/important_concepts.html",
            "omnigibson/robots.html",
            "omnigibson/controllers.html",
            "omnigibson/sensors.html",
            "omnigibson/tasks.html",
            "behavior_components/behavior_tasks.html",
            "behavior_components/joylo.html",
            "behavior_components/scenes.html",
            "tutorials/custom_robot_import.html",
            "tutorials/customizing_robots.html",
            "tutorials/running_on_a_compute_cluster.html",
            "tutorials/remote_streaming.html",
            "other/speed_optimization.html",
        )
    }
    for url in sorted(related - seen):
        r = get(url)
        name = "site_" + slug(url.replace(SITE, "")) + ".md"
        if r.status_code != 200:
            fail(name, url, f"HTTP {r.status_code}")
            continue
        r.encoding = "utf-8"
        title, md, _, _ = html_to_md(r.text, url)
        save(name, url, title, md, note="대회 페이지 본문이 링크하거나 대회 준비에 직접 쓰이는 사이트 문서")


# ---------------------------------------------------------------- 2. GitHub
GITHUB_FILES = [
    ("wensi-ai/openpi", "behavior", "README.md", "π0.5 베이스라인 포크 README"),
    ("wensi-ai/openpi", "behavior", "docs/b1k.md", "π0.5 베이스라인 BEHAVIOR 문서"),
    ("wensi-ai/Isaac-GR00T", "behavior", "README.md", "GR00T 베이스라인 포크 README"),
    ("wensi-ai/Isaac-GR00T", "behavior", "getting_started/b1k.md", "GR00T 베이스라인 BEHAVIOR 문서"),
    ("StanfordVL/BEHAVIOR-1K", "main", "README.md", "BEHAVIOR-1K 레포 README"),
    ("IliaLarchenko/behavior-1k-solution", "main", "README.md", "2025 1위 공개 솔루션 README"),
    ("mli0603/openpi-comet", "main", "README.md", "2025 2위 공개 솔루션 README"),
    ("starVLA/starVLA", "starVLA_dev", "examples/simBenchmarks/Behavior/README.md", "2025 14위 StarVLA BEHAVIOR README"),
]


def fetch_github():
    for repo, ref, path, desc in GITHUB_FILES:
        raw = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
        page = f"https://github.com/{repo}/blob/{ref}/{path}"
        name = "gh_" + slug(f"{repo}_{ref}_{path}").removesuffix(".md") + ".md"
        r = get(raw)
        if r.status_code != 200:
            fail(name, page, f"HTTP {r.status_code}")
            continue
        r.encoding = "utf-8"
        save(name, page, f"{repo} ({ref}) {path} — {desc}", r.text, note=f"raw: {raw}")
    # BEHAVIOR-1K 릴리스 노트 (태그 확인용)
    r = get("https://api.github.com/repos/StanfordVL/BEHAVIOR-1K/releases?per_page=10")
    tags = get("https://api.github.com/repos/StanfordVL/BEHAVIOR-1K/tags?per_page=30")
    if r.status_code == 200:
        body = ["## 태그 목록 (GitHub API)", ""]
        if tags.status_code == 200:
            body += [f"- `{t['name']}` ({t['commit']['sha'][:7]})" for t in tags.json()]
        body += ["", "## 릴리스 노트 (최근 10개)", ""]
        for rel in r.json():
            body += [
                f"### {rel['name']} — 태그 `{rel['tag_name']}`",
                "",
                f"게시: {rel['published_at']} · {rel['html_url']}",
                "",
                (rel.get("body") or "(본문 없음)").strip(),
                "",
            ]
        save(
            "gh_StanfordVL_BEHAVIOR-1K_releases.md",
            "https://github.com/StanfordVL/BEHAVIOR-1K/releases",
            "StanfordVL/BEHAVIOR-1K 태그·릴리스 노트",
            "\n".join(body),
        )
    else:
        fail("gh_StanfordVL_BEHAVIOR-1K_releases.md", "https://github.com/StanfordVL/BEHAVIOR-1K/releases", r.status_code)


# ---------------------------------------------------------------- 3. HuggingFace 데이터셋
def fetch_hf():
    for repo in ("2026-challenge-demos", "2026-challenge-rawdata", "zipped-datasets", "2025-challenge-demos"):
        card = f"https://huggingface.co/datasets/behavior-1k/{repo}"
        r = get(card + "/resolve/main/README.md")
        api = get(f"https://huggingface.co/api/datasets/behavior-1k/{repo}", params={"blobs": "true"})
        meta = api.json() if api.status_code == 200 else {}
        sib = meta.get("siblings", [])
        body = []
        if r.status_code == 200:
            r.encoding = "utf-8"
            body += ["## 데이터셋 카드 (README.md 원문)", "", r.text.strip(), ""]
        else:
            body += [f"## 데이터셋 카드", "", f"README.md 없음 (HTTP {r.status_code}).", ""]
        body += [
            "## HuggingFace API 메타데이터",
            "",
            f"- lastModified: {meta.get('lastModified')}",
            f"- gated: {meta.get('gated')} · private: {meta.get('private')}",
            f"- 파일 수: {len(sib)}",
            f"- 총 크기(API usedStorage, 바이트): {meta.get('usedStorage')}",
            "",
        ]
        # 파일 구성 요약 (숫자를 N 으로 묶어 패턴별 개수)
        pat = {}
        for s in sib:
            key = re.sub(r"\d+", "N", s["rfilename"])
            n, size = pat.get(key, (0, 0))
            pat[key] = (n + 1, size + (s.get("size") or 0))
        body += ["| 파일 패턴 | 개수 | 합계 크기(GB) |", "|---|---|---|"]
        for k, (n, size) in sorted(pat.items()):
            body.append(f"| `{k}` | {n} | {size / 1e9:.2f} |")
        if len(sib) <= 40:
            body += ["", "| 파일 | 크기(GB) |", "|---|---|"]
            body += [f"| `{s['rfilename']}` | {(s.get('size') or 0) / 1e9:.2f} |" for s in sib]
        save(f"hf_{repo}_card.md", card, f"HuggingFace behavior-1k/{repo} 데이터셋 카드·파일 구성", "\n".join(body))

    # 2026 데모의 작은 메타 파일 원문
    for path in ("meta/tasks.jsonl", "meta/info.json"):
        url = f"https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/resolve/main/{path}"
        r = get(url)
        name = "hf_2026-challenge-demos_" + slug(path) + ".md"
        if r.status_code != 200:
            fail(name, url, f"HTTP {r.status_code}")
            continue
        r.encoding = "utf-8"
        lang = "json" if path.endswith(".json") else "jsonl"
        save(name, url, f"behavior-1k/2026-challenge-demos {path} 원문", f"```{lang}\n{r.text.strip()}\n```")

    # 토론(discussions) 전부
    for kind, repo in (("datasets", "behavior-1k/2026-challenge-demos"), ("datasets", "behavior-1k/2026-challenge-rawdata"),
                       ("spaces", "behavior-1k/2026-challenge-leaderboard")):
        lst = get(f"https://huggingface.co/api/{kind}/{repo}/discussions")
        if lst.status_code != 200:
            continue
        for d in lst.json().get("discussions", []):
            url = f"https://huggingface.co/{kind}/{repo}/discussions/{d['num']}"
            det = get(f"https://huggingface.co/api/{kind}/{repo}/discussions/{d['num']}")
            if det.status_code != 200:
                fail(f"hf_disc_{slug(repo)}_{d['num']}.md", url, f"HTTP {det.status_code}")
                continue
            dj = det.json()
            body = [f"- 상태: {dj.get('status')} · PR 여부: {dj.get('isPullRequest')} · 생성: {dj.get('createdAt')}", ""]
            for ev in dj.get("events", []):
                who = (ev.get("author") or {}).get("name")
                data = ev.get("data") or {}
                text = data.get("latest", {}).get("raw") if isinstance(data.get("latest"), dict) else None
                body.append(f"### [{ev.get('type')}] {who} · {ev.get('createdAt')}")
                body.append("")
                body.append(text or json.dumps(data, ensure_ascii=False)[:2000])
                body.append("")
            save(f"hf_disc_{slug(repo)}_{d['num']}.md", url, f"HF 토론 {repo} #{d['num']}: {d['title']}", "\n".join(body))


# ---------------------------------------------------------------- 4. 리더보드 Space
def gradio_call(endpoint, data):
    r = SESSION.post(f"{LB_SPACE}/gradio_api/call/{endpoint}", json={"data": data}, timeout=60)
    eid = r.json()["event_id"]
    res = SESSION.get(f"{LB_SPACE}/gradio_api/call/{endpoint}/{eid}", timeout=120).text
    m = re.search(r"event: complete\s*\ndata: (.*)", res)
    return json.loads(m.group(1)) if m else None


def table_md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in row) + " |")
    return "\n".join(out)


def fetch_leaderboard():
    space = "https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard"
    for path in ("README.md", "data/README.md", "data/2025_results.jsonl", "data/self_reported_results.jsonl",
                 "data/results.jsonl", "data/submissions.jsonl", "data/per_task_results.jsonl", "app.py",
                 "scripts/extract_self_reported_scores.py"):
        url = f"{space}/resolve/main/{path}"
        r = get(url)
        name = "lb_space_" + slug(path).removesuffix(".md") + ".md"
        if r.status_code != 200:
            fail(name, url, f"HTTP {r.status_code}")
            continue
        r.encoding = "utf-8"
        text = r.text.strip() or "(빈 파일)"
        lang = {"py": "python", "jsonl": "jsonl", "md": ""}.get(path.rsplit(".", 1)[-1], "")
        body = text if path.endswith(".md") else f"```{lang}\n{text}\n```"
        save(name, f"{space}/blob/main/{path}", f"리더보드 Space 파일 {path}", body)
    # 실시간 표 (Gradio API)
    try:
        lb = gradio_call("display_leaderboard", ["", "All"])[0]
        lb25 = gradio_call("display_2025_leaderboard", ["", "All"])[0]
        body = ["## 2026 리더보드 (현재 표)", "", table_md(lb["headers"], lb["data"]), "",
                "## 2025 리더보드 (보관)", "", table_md(lb25["headers"], lb25["data"])]
        save("lb_live_tables.md", LB_SPACE + "/", "리더보드 실시간 표 (Gradio API 로 받은 값)", "\n".join(body),
             note="Gradio API /display_leaderboard, /display_2025_leaderboard 호출 결과. 받은 시각 기준 스냅샷")
    except Exception as e:  # noqa: BLE001
        fail("lb_live_tables.md", LB_SPACE, repr(e))


# ---------------------------------------------------------------- 5. 등록 폼 문항
def fetch_form():
    url = "https://forms.gle/Kf4ABLmDKbuK5Yhj6"
    r = get(url)
    m = re.search(r"FB_PUBLIC_LOAD_DATA_ = (.*?);</script>", r.text, re.S)
    if not m:
        fail("form_registration.md", url, "FB_PUBLIC_LOAD_DATA_ 없음")
        return
    data = json.loads(m.group(1))
    info = data[1]
    body = [f"- 최종 주소: {r.url}", f"- 폼 제목: {data[3] if len(data) > 3 else ''}", "", "## 폼 설명", "",
            (info[0] or "").strip(), "", "## 문항", ""]
    for item in info[1] or []:
        title, desc = item[1], item[2]
        q = item[4][0] if item[4] else None
        req = bool(q[2]) if q and len(q) > 2 else False
        opts = [o[0] for o in (q[1] or [])] if q and q[1] else []
        body.append(f"- **{title}**{' (필수)' if req else ''}" + (f" — {desc}" if desc else ""))
        for o in opts:
            body.append(f"  - {o or '(기타 입력)'}")
    save("form_registration.md", url, "2026 참가 등록 폼 문항 (Google Forms)", "\n".join(body),
         note="updates.html 08/24 공지의 링크. forms.gle/R7JVhpRNR7Vh7KtL6 도 같은 폼으로 연결됨")


# ---------------------------------------------------------------- 6. 논문·기사
ARXIV = [
    ("2403.09227", "BEHAVIOR-1K 벤치마크 논문"),
    ("2108.03332", "BEHAVIOR (원조, 2021)"),
    ("2503.05652", "BEHAVIOR Robot Suite (R1 로봇 선택 근거)"),
    ("2504.16054", "π0.5"),
    ("2410.24164", "π0"),
    ("2503.14734", "GR00T N1"),
    ("2512.06951", "2025 1위 기술 보고서"),
    ("2512.10071", "2025 2위 기술 보고서"),
]
ARXIV_FULLTEXT = {"2512.06951", "2512.10071"}


def fetch_papers():
    ids = ",".join(a for a, _ in ARXIV)
    r = get("http://export.arxiv.org/api/query", params={"id_list": ids, "max_results": len(ARXIV)})
    soup = BeautifulSoup(r.text, "xml")
    desc = dict(ARXIV)
    for e in soup.find_all("entry"):
        aid = re.sub(r"v\d+$", "", e.id.text.rsplit("/", 1)[-1])
        title = _collapse(e.title.text).strip()
        authors = ", ".join(a.find("name").text for a in e.find_all("author"))
        body = [f"- 제목: {title}", f"- 저자: {authors}", f"- 게시: {e.published.text[:10]} · 갱신: {e.updated.text[:10]}",
                f"- 분류: {desc.get(aid, '')}", "", "## Abstract", "", _collapse(e.summary.text).strip()]
        save(f"arxiv_{aid}_abs.md", f"https://arxiv.org/abs/{aid}", f"arXiv {aid} — {title}", "\n".join(body))
    for aid in ARXIV_FULLTEXT:
        url = f"https://arxiv.org/html/{aid}"
        r = get(url)
        if r.status_code != 200:
            fail(f"arxiv_{aid}_fulltext.md", url, f"HTTP {r.status_code}")
            continue
        r.encoding = "utf-8"
        title, md, _, _ = html_to_md(r.text, r.url, selector="article.ltx_document")
        save(f"arxiv_{aid}_fulltext.md", r.url, f"arXiv {aid} 본문(HTML판) — {title}", md,
             note="수식·그림은 변환 중 깨질 수 있음. 정확한 수치는 PDF 로 확인")
    for url, fname, sel, note in (
        ("https://hai.stanford.edu/news/behavior-challenge-charts-the-way-forward-for-domestic-robotics",
         "news_hai_behavior_challenge.md", "article", "Stanford HAI 기사 (2025 대회 회고)"),
        ("https://merlyn-labs.com/behavior-report", "report_merlin_labs_2025.md", None, "2025 11위 Merlin Labs 보고서 (리더보드 링크)"),
    ):
        r = get(url)
        if r.status_code != 200:
            fail(fname, url, f"HTTP {r.status_code}")
            continue
        r.encoding = r.apparent_encoding or "utf-8"
        title, md, _, _ = html_to_md(r.text, r.url, selector=sel)
        save(fname, r.url, f"{title}", md, note=note)


# ---------------------------------------------------------------- 목록
def write_index():
    lines = ["# raw — 원문 보관본 목록", "", f"받은 시각: {now_str()} · 생성: `_scrape.py`", "",
             "인용은 이 폴더의 파일에서 한다. 각 파일 맨 위에 원본 URL 이 있다.", "",
             "| 파일 | 원본 | 상태 | 제목/사유 |", "|---|---|---|---|"]
    for f, u, st, t in INDEX:
        lines.append(f"| [{f}]({f}) | {u} | {st} | {str(t).replace('|', '/')} |")
    (OUT / "_목록.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    steps = [crawl_site, fetch_github, fetch_hf, fetch_leaderboard, fetch_form, fetch_papers]
    for step in steps:
        try:
            step()
        except Exception as e:  # noqa: BLE001
            fail(step.__name__, "-", repr(e))
    write_index()
    print(f"done: {sum(1 for x in INDEX if x[2] == 'ok')} ok, {sum(1 for x in INDEX if x[2] != 'ok')} fail")
