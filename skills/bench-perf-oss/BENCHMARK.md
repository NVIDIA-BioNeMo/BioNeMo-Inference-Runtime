# Skill Benchmark: bench-perf-oss

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `bench-perf-oss`
- Validation status: `passed`
- Evaluation date: 2026-11-17
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 11 evaluation tasks (9 positive, 2 negative)
- Dataset digest: `sha256:84f4a60929fe1591ebfdb376f2313a0123288921678895e37f156e774ec52c2b` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| bench-perf-oss | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| bench-perf-oss | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| bench-perf-oss | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| bench-perf-oss | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| bench-perf-oss | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `bench-perf-oss`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 90.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 100.0% → 100.0% (±0.0 points) |
| Correctness | Not available | 89.1% → 89.1% (±0.0 points) |
| Discoverability | Not available | 82.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 76.3% → 91.2% (+14.9 points) |
| Efficiency | Not available | 88.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.1% → 88.1% (±0.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,976,419 | 1,512,733 | +1,463,686 | +96.76% | skill 11/11; base 11/11 |
| claude-code | bench-perf-oss-001-manifest | 966,429 | 283,124 | +683,305 | +241.34% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-002-input-parity | 128,675 | 109,653 | +19,022 | +17.35% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-003-protenix-lock | 313,073 | 71,072 | +242,001 | +340.50% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-004-boltz-quality | 224,440 | 0 | +224,440 | N/A | skill 1/1; base 1/1; Baseline total tokens are zero; percentage change is undefined. |
| claude-code | bench-perf-oss-005-artifact-audit | 293,871 | 198,580 | +95,291 | +47.99% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-006-speedup-aggregation | 232,700 | 195,447 | +37,253 | +19.06% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-007-checkpoint-gate | 231,370 | 109,732 | +121,638 | +110.85% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-008-affinity-negative | 107,278 | 106,707 | +571 | +0.54% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-009-throughput-negative | 106,271 | 105,961 | +310 | +0.29% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-010-of2-role-lock | 195,399 | 110,651 | +84,748 | +76.59% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-011-approximate-control | 176,913 | 221,806 | -44,893 | -20.24% | skill 1/1; base 1/1 |
| codex | All cases | 1,921,852 | 1,352,062 | +569,790 | +42.14% | skill 11/11; base 11/11 |
| codex | bench-perf-oss-001-manifest | 331,419 | 242,502 | +88,917 | +36.67% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-002-input-parity | 149,292 | 75,199 | +74,093 | +98.53% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-003-protenix-lock | 158,680 | 172,092 | -13,412 | -7.79% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-004-boltz-quality | 132,216 | 88,141 | +44,075 | +50.01% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-005-artifact-audit | 341,681 | 117,863 | +223,818 | +189.90% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-006-speedup-aggregation | 170,606 | 158,719 | +11,887 | +7.49% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-007-checkpoint-gate | 173,157 | 111,383 | +61,774 | +55.46% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-008-affinity-negative | 84,750 | 71,457 | +13,293 | +18.60% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-009-throughput-negative | 70,656 | 55,846 | +14,810 | +26.52% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-010-of2-role-lock | 124,462 | 118,206 | +6,256 | +5.29% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-011-approximate-control | 184,933 | 140,654 | +44,279 | +31.48% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,898,271 | 2,864,795 | +2,033,476 | +70.98% | skill 22/22; base 22/22 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 46 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 6 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 11 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across measurement.md and workflow.md:
  "## GPU inventory" in measurement.md (lines 730-775)
  vs "### GPU inventory" in workflow.md (lines 121-149) (`measurement.md:730`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across measurement.md and models/of2.md and no-pipeline.md and workflow.md:
  "## OSS `torch.compile`" in measurement.md (lines 234-276)
  vs "### Fast synthetic compile probe" in measurement.md (lines 303-308)
  vs "### Fast synthetic compile probe" in measurement.md (lines 332-340)
  vs "### Fast synthetic compile probe" in measurement.md (lines 365-366)
  vs "### Track recompiles on warmup" in measurement.md (lines 538-591)
  vs "### How to read the deltas" in measurement.md (lines 592-637)
  vs "### Compile retry ladder" in measurement.md (lines 650-674)
  vs "### Compile retry ladder" in measurement.md (lines 676-680)
  vs "### Compile retry ladder" in measurement.md (lines 689-691)
  vs "## Warmup and repeats" in measurement.md (lines 701-729)
  vs "## Result JSON" in measurement.md (lines 919-925)
  vs "## Compile — Evoformer only" in models/of2.md (lines 501-555)
  vs "# per sample. Do not torch.compile(oss_compiled) as a whole." in no-pipeline.md (lines 98-105)
  vs "### Locked inference config" in workflow.md (lines 347-353)
  vs "### Locked inference config" in workflow.md (lines 354-371)
  vs "### Processor path, OSS side — `$WORKDIR/bench/run_oss.py`" in workflow.md (lines 526-569) (`measurement.md:234`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across measurement.md and workflow.md:
  "## BioIR pretrained config" in measurement.md (lines 112-141)
  vs "### Locked inference config" in workflow.md (lines 321-334) (`measurement.md:112`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across deepspeed-evoformer.md and environment.md and models/of2.md and models/of3.md:
  "# DeepSpeed `evoformer_attn` — source build" in deepspeed-evoformer.md (lines 19-29)
  vs "## Build" in deepspeed-evoformer.md (lines 63-93)
  vs "## Verify before any forward" in deepspeed-evoformer.md (lines 158-220)
  vs "# Environment setup — BioIR vs OSS" in environment.md (lines 19-29)
  vs "## Step 3 — Diff and classify conflicts" in environment.md (lines 378-426)
  vs "# Keep the current BioIR interpreter untouched." in environment.md (lines 693-732)
  vs "## Step 5 — Shared environment" in environment.md (lines 733-751)
  vs "## OpenFold2 / OpenFold3 — DeepSpeed Evoformer kernel" in environment.md (lines 752-769)
  vs "# nothing — see deepspeed-evoformer.md." in environment.md (lines 784-834)
  vs "## Environment and kernels" in models/of2.md (lines 204-246)
  vs "## OSS runtime configuration" in models/of3.md (lines 77-127) (`deepspeed-evoformer.md:19`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across samples.md and workflow.md:
  "## Manifest fields" in samples.md (lines 235-267)
  vs "### Sample manifest" in workflow.md (lines 203-248) (`samples.md:235`)
- 47 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
