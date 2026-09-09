<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

PR title must follow Conventional Commits (feat:, fix:, docs:, perf:, refactor:,
chore:). Mark API-breaking changes with "!" after the type (e.g. "feat!:") and
describe them in a "BREAKING CHANGE:" footer. The title becomes the changelog
line. NVIDIA developers: include the JIRA key / NVBug ID.
-->

<!-- rumdl-disable-next-line MD041 -->
## Description

<!-- What does this PR change and why? Reference issues with "closes #NNNN".
     New features / breaking changes require a maintainer-approved issue first
     (see docs/contributing.md). -->

### Usage

<!-- Optional: how a user interacts with the changed code. -->

```python
# example
```

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (API-breaking — `type!:` in title, `BREAKING CHANGE:`
      footer)
- [ ] Refactor
- [ ] Documentation
- [ ] Build / CI

## Checklist

- [ ] My commits are signed off (DCO): `git commit -s` (see
      [contributing.md](../docs/contributing.md))
- [ ] I have read the [Contributing Guidelines](../docs/contributing.md)
- [ ] For a new feature or breaking change, an issue was filed and approved
      first
- [ ] I added or updated tests, and they pass locally
- [ ] I updated documentation as needed
