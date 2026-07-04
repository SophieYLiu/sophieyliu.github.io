# sophieyliu.github.io

Personal blog. Write in Markdown, publish with one command.

**Live at:** https://sophieyliu.github.io/

## Write a new post

```bash
./new "My post title"     # creates content/posts/my-post-title.md
```

Open that file, write your post in Markdown below the `---` header, then:

```bash
./publish                 # builds the site, commits, and pushes it live
```

The site rebuilds and is live in about a minute. That's the whole workflow.

## How it works

- **`content/`** — your writing (Markdown). This is the only place you edit.
  - `content/posts/*.md` — one file per post, with `title:` and `date:` at the top.
  - `content/about.md` — the about page.
- **`build.py`** — renders Markdown into HTML at the repo root (`index.html`,
  `about.html`, `posts/*.html`). Run it directly to preview locally, or let
  `./publish` run it for you.
- **`style.css`** — all the styling. Edit this to change how the site looks.
- Generated `.html` files are committed so GitHub Pages (deploy-from-`main`)
  serves them directly. `.nojekyll` keeps Pages from reprocessing them.

## Preview locally

```bash
./build.py && python3 -m http.server -d . 8000
# then open http://localhost:8000
```

## Post format

```markdown
---
title: Hello, world
date: 2026-06-25
---

Your **Markdown** goes here. Headings, links, `code`, code fences,
lists, blockquotes, and tables all work.
```

Set `draft: true` in the header to keep a post out of the build.
