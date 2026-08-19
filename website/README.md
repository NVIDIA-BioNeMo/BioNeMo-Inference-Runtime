# BioIR documentation website

Source for the BioNeMo Inference Runtime (BioIR) documentation site, built
with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and
[mkdocstrings](https://mkdocstrings.github.io/).

## Preview locally

From the repository root:

```bash
pip install -r website/requirements.txt
mkdocs serve -f website/mkdocs.yml
```

Then open <http://127.0.0.1:8000> — it redirects to
`/BioNeMo-Inference-Runtime/` because `site_url` carries the GitHub Pages
project path. To produce the static site instead:

```bash
mkdocs build -f website/mkdocs.yml
```

The output lands in `website/site/` (git-ignored). The API reference is
parsed statically by griffe, so building the site needs no GPU, torch, or
Ray.

## What lives where

- `docs/` — hand-written pages (landing page, guides). This is the only
  tracked content.
- `scripts/gen_ref_pages.py` — generates one API-reference page per
  `bionemo_ir` module at build time (virtual files, nothing on disk).
- `scripts/import_docs.py` — imports the repo's top-level `docs/` tree into
  the "Developer Docs" section at build time, rewriting links that point
  outside `docs/` to absolute GitHub URLs. `docs/nv/` (internal process
  docs) is excluded.

Deployment (GitHub Pages via CI) is intentionally not wired up yet — see
[plan.md](plan.md) for the next steps.
