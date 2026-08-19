# Website plan

What is done and what comes next for the BioNeMo Inference Runtime (BioIR)
documentation site. Build and preview instructions: [README.md](README.md).

## Done

- MkDocs Material + mkdocstrings scaffold under `website/`.
- API reference generated from docstrings (static griffe parse — no GPU
  stack needed to build).
- `docs/` imported at build time into "Developer Docs" with link rewriting;
  `docs/` itself untouched. `docs/nv/` excluded (internal process docs).
- Landing page, quickstart guide, Mermaid diagrams.
- Lint-clean under the repo's ruff / rumdl rules.

## Next steps

### CI/CD (GitHub Pages)

- [ ] Add `.github/workflows/docs.yml` (the repo's first workflow):
  - PRs touching `website/**`, `docs/**`, `bionemo_ir/**`: build-only check
    with `mkdocs build -f website/mkdocs.yml`.
  - Pushes to `main`: build, then deploy via `actions/upload-pages-artifact`
    plus `actions/deploy-pages`.
- [ ] Enable Pages in the GitHub repo settings (source: GitHub Actions).
- [ ] The workflow lands in internal GitLab first and mirrors out via
  Copybara; the deploy job only runs on the GitHub side.

### Versioning

- [ ] Add `mike` to `website/requirements.txt` and deploy per release
  (`mike deploy <version>`) once the first public release is cut.

### Content

- [ ] More guides under `website/docs/guides/`: Ray multi-GPU inference,
  custom-module onboarding (`.agents/skills/module-onboard/`), pairwise
  memory optimizations (`.agents/skills/scan-mem-opt-patterns/`).
- [ ] Landing-page branding: logo, `overrides/` theme partials.

### API reference

- [ ] Add `__init__.py` to `bionemo_ir/_torch/modules/openfold3/utils/` so
  its four modules get pages (skipped today — griffe cannot traverse
  namespace dirs). Decide whether `dsl_kernels/cute/` stays excluded.
- [ ] Fix the docstring nits griffe warns about (mismatched parameter names
  in `openfold2/structure.py`, `openfold3/embedders.py`, and friends).
- [ ] If the reference nav feels too deep, curate which `_torch` internals
  get pages.

### Nice to have

- [ ] Edit-this-page links: needs per-source paths (`edit_uri` for
  hand-written pages, `mkdocs_gen_files.set_edit_path` for generated ones).
- [ ] C++/CUDA kernel docs via Doxygen, if `cpp/` ever needs a public API
  reference.
