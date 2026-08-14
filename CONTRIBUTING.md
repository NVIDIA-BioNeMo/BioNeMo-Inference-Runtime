<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to BioNeMo Inference Runtime

Thanks for your interest in improving BioIR. This document explains how to
propose changes, the sign-off we require, and the local checks your change must
pass before review.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you are expected to uphold it.

## Ways to contribute

1. **Report a bug or request a feature** using the
   [issue forms](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/issues/new/choose).
2. **Discuss usage questions** in
   [Discussions](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/discussions).
3. **Open a pull request** for a fix or feature (see the workflow below).

## When an issue is required

BioIR uses a **hybrid** contribution model:

- **New features and API-breaking changes** must begin with an **issue**
  (feature request or RFC) that a maintainer reviews and approves **before** you
  open a PR. This avoids wasted work on changes that do not fit the roadmap.
- **Small bug fixes, docs, and typo corrections** may go straight to a PR; an
  issue is welcome but not required. Reference any related issue with
  `closes #NNNN`.

## Developer Certificate of Origin (DCO) — sign your work

All commits **must** be signed off. Sign-off certifies that you wrote the
contribution or otherwise have the right to submit it under the project license
(the [Developer Certificate of Origin](https://developercertificate.org/)). PRs
containing unsigned commits are blocked by the DCO check.

Sign off by adding `-s` / `--signoff` when you commit:

```bash
git commit -s -m "fix: correct attention mask for padded batches"
```

This appends a trailer to your commit message:

```text
Signed-off-by: Your Name <your@email.com>
```

Set your `user.name` and `user.email` so the trailer is accurate; the name must
match a real identity. To sign off a series of existing commits, use
`git rebase --signoff <base>`.

<details>
<summary>Full text of the Developer Certificate of Origin, Version 1.1</summary>

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license
document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right
    to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my
    knowledge, is covered under an appropriate open source license and I have
    the right under that license to submit that work with modifications, whether
    created in whole or in part by me, under the same open source license
    (unless I am permitted to submit under a different license), as indicated in
    the file; or

(c) The contribution was provided directly to me by some other person who
    certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and
    that a record of the contribution (including all personal information I
    submit with it, including my sign-off) is maintained indefinitely and may be
    redistributed consistent with this project or the open source license(s)
    involved.
```

</details>

## Development workflow

1. **Fork** the repository and clone your fork.
2. Create a topic branch from `main`.
3. Set up the environment, build, and run the tests by following the
   [Docs](docs/README.md).
4. Make your change. Add unit tests for fixes and features; add benchmarks for
   performance-sensitive kernels.
5. Open a PR against `main`. Draft PRs run CI without requesting review.
6. Address CI failures and reviewer feedback. A maintainer merges once required
   approvals and status checks pass.

## Pull request expectations

- Keep PRs focused on a single concern; split unrelated changes into separate
  PRs and note dependencies.
- Fill in the pull request template, including the DCO checkbox.
- NVIDIA developers: include the JIRA key or NVBug ID in the PR title where
  applicable.

## Review and merge

- Reviewers are routed automatically by
  [`.github/CODEOWNERS`](.github/CODEOWNERS).
- A PR is merged by someone other than its author (no self-approval), after the
  required Code Owner approvals.
- C++/CUDA changes may require additional Code Owner review at maintainer
  discretion; all changes require at least one approval.
- Security and critical-regression fixes are fast-tracked — label the PR
  `release blocker` and request expedited Code Owner review. Report
  vulnerabilities via [`SECURITY.md`](SECURITY.md), never a public issue.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
