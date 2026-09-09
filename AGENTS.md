---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# AGENTS.md

This repo is **BioIR** (BioNeMo Inference Runtime).

## Project checks

- **Read and follow [`docs/coding.md`][coding]**
- **Run the hooks before committing.** Install once with `prek install`; they
  then run on `git commit`. Or run ad hoc with `prek run`. CI enforces the same
  set.
- **License headers.** Every source file carries the NVIDIA SPDX Apache-2.0
  header; `insert-license` adds it where missing.
- **Sign every commit** (`git commit -s`). CI blocks unsigned work; see
  [`docs/contributing.md`][contributing].

## Development and tests

Daily loop is in [`docs/dev.md`][dev]. See also

- `docker/dev.sh`,
- `scripts/fetch_weights.sh`
- `scripts/run_tests.sh`.

## Git history and MRs

- **MR titles start with a tracker reference**, then a Conventional-Commit
  summary, e.g. `[PROJ-123] feat: ...`.
- **PR and commit titles use Conventional Commits.** Tracker references are
  optional for public PRs and not required for commits. Types: `feat`, `fix`,
  `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`
  (scope optional).
- **Write MR/PR descriptions as [squash commit bodies][commit-message].**
  Summarize what changed and why: the previous problem, the new behavior, and
  the reason for the chosen approach. Let the diff explain how. Match the final
  diff and omit `Validate` and `Validation` sections.
- **Format MR/PR descriptions as commit bodies.** Do not repeat the title. Wrap
  prose at 72 characters and separate paragraphs with blank lines. Prefer
  paragraphs for context and rationale. Use bullets only for distinct changes,
  constraints, or follow-ups; do not inventory files.
- **Post validation results, follow up or additional notes as MR/PR comments.**
  Link the exact revision and CI job or artifact. State what each result proves
  and any unverified limits.

## Writing and documentation

- **Write terse, direct, active prose.** Lead with facts. Keep shared text
  self-contained; omit narration, recaps, and local-only context.
- **Prefer a list to a table** in Markdown. Use table only when the cells line
  up for numerical data or comparison
- **Documentation assets belong in `docs/assets/`.** Put every image, diagram,
  or other media a public Markdown page references there and link it with a
  relative path. That directory is the only one copied verbatim into the
  generated Fern site; a file anywhere else is rewritten to a GitHub blob URL
  and renders as a broken image, so `docs/fern/src/check_doc_links.py` rejects
  it. Do not add a per-page `img/` directory.

## Agent skills

Task-specific playbooks live in `.agents/skills/`

- `make-data-pipeline` — porting an open-source data pipeline into the BioIR
  pipeline architecture.
- `module-onboard` — moving a source model's module onto BioIR layers, with
  weight conversion and validation.
- `scan-mem-opt-patterns` — cutting activation memory or diagnosing large-`N`
  OOM in a pairwise-representation model.
- `bench-perf-oss` — benchmark BioIR vs OSS **folding** `model.forward()`
  latency and GPU use on a bench set its own `rebuild_dataset.py` builds from
  RCSB and the MSA Search NIM (serial one-sample;
  OpenStructure lDDT plus DockQ on supported protein interfaces; MSAs and
  templates attached when present, on both sides or neither). Not for affinity
  or other non-folding heads until the skill is extended.

[coding]: docs/coding.md
[commit-message]: https://cbea.ms/git-commit/
[contributing]: docs/contributing.md
[dev]: docs/dev.md
