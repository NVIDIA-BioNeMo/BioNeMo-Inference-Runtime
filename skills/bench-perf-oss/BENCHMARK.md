# Skill Benchmark: bench-perf-oss

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `bench-perf-oss`
- Validation status: `passed`
- Evaluation date: 2026-11-16
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 10 evaluation tasks (8 positive, 2 negative)
- Dataset digest: `sha256:61cbe9b5ceeb1e34926036e804f9964f4d53c8422ab88df3c0b50b5ecb2c0f85` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 83.1% — baseline ran, but no comparable score was available; uplift unavailable | 91.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 90.0% (-10.0 points) | 90.0% → 100.0% (+10.0 points) |
| Correctness | 92.0% → 94.0% (+2.0 points) | 92.0% → 100.0% (+8.0 points) |
| Discoverability | 59.4% — baseline ran, but no comparable score was available; uplift unavailable | 81.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 82.0% → 82.7% (+0.7 points) | 75.4% → 84.8% (+9.4 points) |
| Efficiency | 89.1% — baseline ran, but no comparable score was available; uplift unavailable | 89.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.9% → 93.6% (-3.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.1% → 88.3% (+0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,378,113 | 1,513,189 | +864,924 | +57.16% | skill 10/10; base 10/10 |
| claude-code | bench-perf-oss-001-manifest | 673,270 | 238,868 | +434,402 | +181.86% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-002-input-parity | 110,094 | 109,088 | +1,006 | +0.92% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-003-protenix-lock | 330,976 | 229,991 | +100,985 | +43.91% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-004-boltz-quality | 109,623 | 109,351 | +272 | +0.25% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-005-artifact-audit | 369,632 | 197,906 | +171,726 | +86.77% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-006-speedup-aggregation | 258,299 | 194,393 | +63,906 | +32.87% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-007-checkpoint-gate | 111,321 | 109,004 | +2,317 | +2.13% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-008-affinity-negative | 107,488 | 106,694 | +794 | +0.74% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-009-throughput-negative | 106,791 | 106,333 | +458 | +0.43% | skill 1/1; base 1/1 |
| claude-code | bench-perf-oss-010-of2-role-lock | 200,619 | 111,561 | +89,058 | +79.83% | skill 1/1; base 1/1 |
| codex | All cases | 1,786,818 | 1,493,403 | +293,415 | +19.65% | skill 10/10; base 10/10 |
| codex | bench-perf-oss-001-manifest | 309,679 | 251,282 | +58,397 | +23.24% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-002-input-parity | 164,767 | 74,572 | +90,195 | +120.95% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-003-protenix-lock | 214,421 | 398,203 | -183,782 | -46.15% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-004-boltz-quality | 110,236 | 87,281 | +22,955 | +26.30% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-005-artifact-audit | 248,287 | 104,888 | +143,399 | +136.72% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-006-speedup-aggregation | 166,319 | 153,590 | +12,729 | +8.29% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-007-checkpoint-gate | 186,132 | 174,368 | +11,764 | +6.75% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-008-affinity-negative | 71,016 | 70,886 | +130 | +0.18% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-009-throughput-negative | 55,634 | 55,798 | -164 | -0.29% | skill 1/1; base 1/1 |
| codex | bench-perf-oss-010-of2-role-lock | 260,327 | 122,535 | +137,792 | +112.45% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,164,931 | 3,006,592 | +1,158,339 | +38.53% | skill 20/20; base 20/20 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 44 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 6 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 10 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across measurement.md and workflow.md:
  "## GPU inventory" in measurement.md (lines 724-769)
  vs "### GPU inventory" in workflow.md (lines 121-149) (`measurement.md:724`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across measurement.md and models/of2.md and no-pipeline.md and workflow.md:
  "## OSS `torch.compile`" in measurement.md (lines 228-270)
  vs "### Fast synthetic compile probe" in measurement.md (lines 297-302)
  vs "### Fast synthetic compile probe" in measurement.md (lines 326-334)
  vs "### Fast synthetic compile probe" in measurement.md (lines 359-360)
  vs "### Track recompiles on warmup" in measurement.md (lines 532-585)
  vs "### How to read the deltas" in measurement.md (lines 586-631)
  vs "### Compile retry ladder" in measurement.md (lines 644-668)
  vs "### Compile retry ladder" in measurement.md (lines 670-674)
  vs "### Compile retry ladder" in measurement.md (lines 683-685)
  vs "## Warmup and repeats" in measurement.md (lines 695-723)
  vs "## Result JSON" in measurement.md (lines 913-919)
  vs "## Compile — Evoformer only" in models/of2.md (lines 501-555)
  vs "# per sample. Do not torch.compile(oss_compiled) as a whole." in no-pipeline.md (lines 98-105)
  vs "### Locked inference config" in workflow.md (lines 346-352)
  vs "### Locked inference config" in workflow.md (lines 353-370)
  vs "### Processor path, OSS side — `$WORKDIR/bench/run_oss.py`" in workflow.md (lines 525-568) (`measurement.md:228`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across measurement.md and workflow.md:
  "## BioIR pretrained config" in measurement.md (lines 112-135)
  vs "### Locked inference config" in workflow.md (lines 321-333) (`measurement.md:112`)
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
- 45 additional finding(s) are available in the full evaluation artifacts.

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
