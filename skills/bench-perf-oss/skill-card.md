## Description: <br>
Benchmark BioIR against an OSS model's e2e path for folding models, measuring model.forward() latency and GPU use on a bench set the shipped script rebuilds from RCSB and the MSA NIM, and scoring written CIFs with OpenStructure lDDT and DockQ on every supported protein interface. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and performance engineers use this skill to compare BioIR against upstream OSS implementations of structure-folding models (Boltz-1/2, OpenFold2 / AlphaFold2, OpenFold3, Protenix-v2), producing GPU-synced model.forward() latency, GPU memory and telemetry, lDDT and DockQ quality scores, and a comparison report. It is not for affinity or other non-folding heads. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [workflow.md — phase procedures](workflow.md) <br>
- [measurement.md — timing protocol and quality metrics](measurement.md) <br>
- [environment.md — BioIR and OSS environment setup](environment.md) <br>
- [samples.md — dataset and sample catalog](samples.md) <br>
- [msa.md — MSA mapping](msa.md) <br>
- [templates.md — template mapping](templates.md) <br>
- [no-pipeline.md — OSS feature path](no-pipeline.md) <br>
- [deepspeed-evoformer.md — DeepSpeed Evoformer source build](deepspeed-evoformer.md) <br>
- [bundle.md — portable bioir-perf package](bundle.md) <br>
- [Boltz-2 model profile](models/boltz2.md) <br>
- [OpenFold2 / AlphaFold2 model profile](models/of2.md) <br>
- [OpenFold3 model profile](models/of3.md) <br>
- [Protenix-v2 model profile](models/protenix.md) <br>
- [aqlaboratory/openfold v2.2.0](https://github.com/aqlaboratory/openfold/tree/v2.2.0) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Files, Analysis] <br>
**Output Format:** [Python harness scripts, JSON result artifacts, PNG latency charts, and a Markdown comparison report] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Latency reported in seconds to three decimals; every number must cite an executed command, exit code, and artifact path.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
10 evaluation tasks (8 positive, 2 negative), 1 attempt per task, each attempt in its own isolated sandbox pod (`k8s-sandbox`); evaluator version `1.6.9`, evaluation date 2026-11-16. Baseline is the same task attempted without the target skill. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

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

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.9% → 93.6% (-3.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.1% → 88.3% (+0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 44 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 6 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 10 task(s) |

## Skill Version(s): <br>
b5afa0e7 (source: git SHA, committed 2026-11-16) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
