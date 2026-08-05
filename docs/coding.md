# Coding Guidelines

> **Don't fight the tooling, follow it** unless you have a real good reason.

These guidelines exist so that the code is consistent, reviewable, and safe to
enforce automatically, whether it's authorized by developers or AI agents.

## How style is enforced

[`prek.toml`][prekcfg] defines the hooks. [`prek`][prek] runs them both locally
and in CI, touch only new or changed files.

- **Local:** `pip install prek`, then `prek install -t pre-commit -t commit-msg`
  (the `commit-msg` shim runs the commit-message linter). Hooks then run on
  `git commit`.
- **CI:** the `validate:styles` job runs the same hooks on the MR diff.

We are using the below tools, all of which are installable via `pip`.

| Concern              | Tool                                                             |
| -------------------- | ---------------------------------------------------------------- |
| Python format + lint | [`ruff`][ruff] (line length **120**, config in `pyproject.toml`) |
| C/C++/CUDA format    | [`clang-format`][clang-format] (`.clang-format`, in the gate)    |
| Markdown             | [`rumdl`][rumdl]                                                 |
| Shell lint + format  | [`shellcheck`][shellcheck] + [`shfmt`][shfmt]                    |
| License headers      | `insert-license` ([Lucas-C/pre-commit-hooks][license-hook])      |
| Line endings         | `.gitattributes` (`eol=lf`, git-native)                          |
| Commit messages      | Conventional Commits — [`commitizen`][cz] + `validate:mr-title`  |

## Python

Follow [PEP 8][pep8] unless noted. Target Python 3.12+.

### Naming

- Files: `snake_case.py`. Classes: `PascalCase`. `Functions/methods/variables`:
  `snake_case`. Constants: `UPPER_SNAKE_CASE`.
- Prefix non-public module/class members with a single underscore.
- For host/device tensors whose location is ambiguous, suffix `_host` /
  `_device` (or `_cuda`), especially when copies exist in both places.

### Imports

- No wildcard imports. Let `ruff` (isort rules) order imports.
- Keep `__all__` current to document the public interface.

### Typing

- Annotate every function argument and return type. Use `-> None` explicitly
  when nothing is returned.
- Prefer builtin generics and unions: `list[int]`, `dict[str, int]`,
  `int | None` — not `typing.List` / `typing.Optional`.
- Avoid `typing.Any` and `# type: ignore`. Use `Literal[...]` for a fixed set of
  string values; `Protocol` for duck-typed interfaces.

### Error handling

- Catch the narrowest exception set possible; keep the `try` body minimal and
  put logic in `else`. Prefer builtin exception types. Raise `ValueError`, don't
  `assert`, for invalid input.
- Avoid reflection when a direct expression works.

### Docstrings

- [Google style][google-style], parsable by Sphinx. Public functions and class
  initializers get docstrings; document their arguments.
- For tensor-like arguments, document expected dimensions (e.g.
  `[batch, seq_len, hidden]`) and the allowed dtype(s) when constrained.
- Reserve inline comments for non-obvious logic; don't restate the code.

### Pydantic (user-facing config)

For any user-facing configuration class, use Pydantic, not dataclasses:

- Inherit from a strict base (`extra="forbid"`) to reject unknown fields.
- No `__init__`; use `@field_validator` / `@model_validator` for validation and
  `model_post_init()` for post-validation setup.
- Every field gets `Field(description=...)`. Use `default_factory` for mutable
  defaults, `Literal[...]` for enumerations, and constrained types
  (`PositiveInt`, `Field(ge=0)`, ...) over custom validators.
- Prefer `model_dump()` / direct construction over `to_dict()` / `from_dict()`.

## C/C++/CUDA

The `cpp/` tree follows the [TensorRT-LLM C++ guidelines][trtllm-cg] (Allman
braces, east-const, 120-col, `k`/`m` naming). Don't fight `.clang-format`.

- **`clang-format`** runs in the style gate (fast, no build). It's pinned to a
  single version in `prek.toml`; use that exact version locally so output
  doesn't ping-pong.
- `.clangd` drives editor LSP for C++/CUDA.

## License header

All source files carry the NVIDIA SPDX Apache-2.0 header (see
[`.license-header.txt`][hdr]) — Python, shell, and CMake with `#` comments, and
C/C++/CUDA in a `/* ... */` block. `insert-license` adds it where missing and
leaves existing headers — including year ranges — untouched.

## Commits

Commit messages follow [Conventional Commits][conventional] so history can drive
changelog generation and other automation.

The project lives in two repos kept in sync by [Copybara][copybara]: GitLab
(internal, source of truth) and GitHub (open source). Accepted GitLab MRs on
non-proprietary paths mirror to GitHub; accepted GitHub PRs import back to
GitLab. A change is a **merge request** (GitLab) or **pull request** (GitHub) —
**MR/PR** below.

Merges are fast-forward (linear history). Squashing is the default and
recommended, so an MR/PR usually lands as one commit whose subject is the
**MR/PR title**. Write that title as the commit you want in history — it is the
enforced unit:

- **Title — ticket key first, then Conventional Commits.** A GitHub, JIRA, or
  NVBugs reference in brackets, then a conventional summary:
  `[BNMTRT-382] fix: bump deps`. On GitLab, `validate:mr-title` enforces it — a
  bracketed key, then `cz check` on the summary. Copybara scrubs the leading key
  when mirroring to GitHub, leaving a clean `fix: ...`.
- Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
  `build`, `ci`, `chore`, `revert`. A scope is allowed but not required — we
  don't enforce components.
- **Per-commit (local aid).** The `commitizen` `commit-msg` hook checks each
  commit title is Conventional Commits — the ticket key is **not** required on
  commits, so you can commit freely while experimenting. Squash discards these
  commits, so the MR/PR title is the real gate; the same tool generates the
  changelog later.

Only the **title** is enforced. Body conventions are recommended, not gated:

- **Body wrap** at ~72–80 cols for readability.
- **Breaking changes:** `type!:` and/or a `BREAKING CHANGE: <desc>` footer —
  drives a major version bump in the changelog.
- **Footer trailers** (git-trailer `Token: value`): `Refs: BNMTRT-123`,
  `Signed-off-by:` (DCO), `Co-authored-by:`. Copybara preserves trailers across
  the sync.

> To carry an MR/PR description into the squashed commit body (so it reaches the
> changelog), set the platform's squash commit template to include the
> description. Changelog generation itself (`commitizen` or `git-cliff` → the
> Keep-a-Changelog sections) is a later step.

[conventional]: https://www.conventionalcommits.org/
[copybara]: https://github.com/google/copybara
[cz]: https://github.com/commitizen-tools/commitizen
[trtllm-cg]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md
[prekcfg]: ../prek.toml
[hdr]: ../.license-header.txt
[prek]: https://github.com/j178/prek
[ruff]: https://github.com/astral-sh/ruff
[rumdl]: https://github.com/rvben/rumdl
[shellcheck]: https://github.com/shellcheck-py/shellcheck-py
[shfmt]: https://github.com/MaxWinterstein/shfmt-py
[clang-format]: https://github.com/pre-commit/mirrors-clang-format
[license-hook]: https://github.com/Lucas-C/pre-commit-hooks
[pep8]: https://peps.python.org/pep-0008/
[google-style]: https://google.github.io/styleguide/pyguide.html
