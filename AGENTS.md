# AGENTS.md

## CRITICAL (YOU MUST)

- **Read and follow [`docs/coding.md`][coding]**
- **Run the hooks before committing.** Install once with
  `prek install -t pre-commit -t commit-msg`; they then run on `git commit`. Or
  run ad hoc with `prek run`. CI enforces the same set via `validate:styles`.
- **MR/PR titles start with a ticket key** — a GitHub, JIRA, or NVBugs
  reference — then a Conventional-Commit summary, e.g.
  `[BNMTRT-xxx] feat: ...`. Types: `feat`, `fix`, `docs`, `style`,
  `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert` (scope optional).
- **Commit titles** should follow the same shape without required ticket key.
- **License headers.** Every source file carries the NVIDIA SPDX Apache-2.0
  header; `insert-license` adds it where missing.

## Agent skills

Task-specific playbooks live in `.agents/skills/`

| Skill                   | Read it when                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `make-data-pipeline`    | porting an open-source data pipeline into the BioIR pipeline architecture                |
| `module-onboard`        | moving a source model's module onto BioIR layers, with weight conversion and validation  |
| `scan-mem-opt-patterns` | cutting activation memory or diagnosing large-`N` OOM in a pairwise-representation model |

[coding]: docs/coding.md
