#!/usr/bin/env python3
"""Static site generator for augsep.github.io.

Reads Markdown from content/ and writes HTML to the repo root, which is
what GitHub Pages serves. Run with:  ./build.py   (or  python3 build.py)
"""

import html
import re
import shutil
from datetime import date, datetime
from pathlib import Path

import markdown

# ── Config ───────────────────────────────────────────────────────────────
SITE_TITLE = "augsep"
FOOTER = f"© {date.today().year} augsep"

ROOT = Path(__file__).parent
CONTENT = ROOT / "content"
POSTS_SRC = CONTENT / "posts"
POSTS_OUT = ROOT / "posts"

MD = markdown.Markdown(
    extensions=["fenced_code", "tables", "toc", "pymdownx.arithmatex"],
    extension_configs={"pymdownx.arithmatex": {"generic": True}},
)

# KaTeX assets, injected only into pages that actually contain math.
KATEX_HEAD = (
    '  <link rel="stylesheet" '
    'href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" />\n'
)
KATEX_BODY = """  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {delimiters: [{left: '\\\\[', right: '\\\\]', display: true}, {left: '\\\\(', right: '\\\\)', display: false}]});"></script>
"""

# ── Templates ────────────────────────────────────────────────────────────
BASE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <link rel="stylesheet" href="/style.css" />
{katex_head}</head>
<body>
  <header>
    <a href="/" class="site-title">{site_title}</a>
    <nav>
      <a href="/about.html">about</a>
    </nav>
  </header>

  <main>
{main}
  </main>

  <footer>
    <p>{footer}</p>
  </footer>
{katex_body}</body>
</html>
"""

POST = """    <article>
      <header>
        <h1>{title}</h1>
        <p class="meta"><time datetime="{iso}">{human}</time></p>
      </header>

      <div class="body">
{body}
      </div>

      <a href="/" class="back">← all posts</a>
    </article>"""


# ── Frontmatter parsing ──────────────────────────────────────────────────
def parse(path: Path):
    """Return (metadata dict, markdown body) for a content file."""
    text = path.read_text(encoding="utf-8")
    meta, body = {}, text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = m.group(2)
    return meta, body


def render_md(body: str) -> str:
    MD.reset()
    return MD.convert(body)


def page(title, main):
    has_math = "arithmatex" in main
    return BASE.format(
        title=html.escape(title),
        site_title=html.escape(SITE_TITLE),
        main=main,
        footer=html.escape(FOOTER),
        katex_head=KATEX_HEAD if has_math else "",
        katex_body=KATEX_BODY if has_math else "",
    )


# ── Build ────────────────────────────────────────────────────────────────
def build():
    # Clean & recreate output posts dir so deleted drafts disappear.
    if POSTS_OUT.exists():
        shutil.rmtree(POSTS_OUT)
    POSTS_OUT.mkdir(parents=True)

    posts = []
    for src in sorted(POSTS_SRC.glob("*.md")):
        meta, body = parse(src)
        if meta.get("draft", "").lower() in ("true", "yes", "1"):
            continue
        slug = src.stem
        title = meta.get("title", slug)
        iso = meta.get("date", date.today().isoformat())
        d = datetime.strptime(iso, "%Y-%m-%d").date()
        human = d.strftime("%b %-d, %Y")

        main = POST.format(
            title=html.escape(title), iso=iso, human=human, body=render_md(body)
        )
        out = POSTS_OUT / f"{slug}.html"
        out.write_text(page(f"{title} — {SITE_TITLE}", main), encoding="utf-8")
        posts.append({"slug": slug, "title": title, "iso": iso, "human": human, "date": d})

    # Homepage: newest first.
    posts.sort(key=lambda p: p["date"], reverse=True)
    items = "\n".join(
        f'''      <li>
        <time datetime="{p["iso"]}">{p["human"]}</time>
        <a href="/posts/{p["slug"]}.html">{html.escape(p["title"])}</a>
      </li>'''
        for p in posts
    )
    index_main = f'    <ul class="post-list">\n{items}\n    </ul>'
    (ROOT / "index.html").write_text(page(SITE_TITLE, index_main), encoding="utf-8")

    # About page.
    about_meta, about_body = parse(CONTENT / "about.md")
    about_main = f'    <div class="about">\n{render_md(about_body)}\n    </div>'
    (ROOT / "about.html").write_text(
        page(f"About — {SITE_TITLE}", about_main), encoding="utf-8"
    )

    print(f"✓ built {len(posts)} post(s) + index + about")


if __name__ == "__main__":
    build()
