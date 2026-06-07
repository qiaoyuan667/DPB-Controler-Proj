# Prior-Art Survey: Privacy-Budgeted Runtime for LLM Agents

Date: 2026-06-07

This memo checks whether the proposal idea has already been published. The
proposal under review is:

> Privacy-budgeted runtime for LLM agents via local policy induction,
> tokenizer-aware exact masking, semantic leakage verification,
> counterfactual privacy cost, per-attribute budget accounting, and
> utility-preserving abstraction/rewriting.

## Bottom Line

I did not find an exact prior work that combines all of the proposal's main
components into the same runtime:

- local induction of structured protected facts and disclosure rules,
- tokenizer-aware exact string masking,
- semantic leakage verification,
- counterfactual likelihood-based privacy cost,
- per-protected-attribute privacy budgets over multi-turn agent actions,
- abstraction/rewriting as the fallback action.

However, several papers are close enough that the proposal must position itself
carefully. The highest-risk overlap is with:

1. **PlanTwin**: local gatekeeper, abstraction, and cumulative disclosure
   budgets for cloud-assisted agents.
2. **PRISM**: generation-time detection and mitigation of secret leakage in
   multi-agent LLM pipelines.
3. **PrivacyChecker / Privacy in Action**: model-agnostic inference-time
   contextual-integrity checks for LLM-powered agents.
4. **Constrained Decoding for Privacy-Preserving LLM Inference**: regex-aware
   logit masking for PII during generation.
5. **AirGapAgent**: task-scoped access minimization for privacy-conscious
   conversational agents.

The safest novelty claim is therefore not "first privacy runtime" or "first
token-level privacy defense." A better claim is:

> A unified privacy-budgeted runtime for LLM agents that combines hard
> tokenizer-aware exact-value guarantees with semantic and counterfactual
> leakage scoring, then performs per-attribute sequential budget accounting and
> abstraction-based repair across agent communication boundaries.

## Closest Work

### 1. PlanTwin

Link: https://arxiv.org/abs/2603.18377

Why it is close:

- It is explicitly a privacy-preserving architecture for cloud-assisted LLM
  agents.
- It keeps raw state local and exposes a sanitized planning abstraction.
- It includes a local gatekeeper.
- It uses cumulative disclosure budgets and multi-turn disclosure control.
- It frames privacy-utility as an abstraction granularity problem.

Key difference from our proposal:

- PlanTwin is primarily a planning-time abstraction architecture: the cloud
  planner sees a schema-constrained digital twin instead of raw local context.
- Our proposal is primarily an inference-time action/runtime controller for
  generated agent communications.
- Our exact masking is tokenizer-aware and value-specific at decoding time.
- Our semantic/counterfactual cost is candidate-response based rather than
  primarily capability/twin-disclosure based.

Risk level: **High**.

Recommended positioning:

> PlanTwin controls what the remote planner observes by projecting the local
> environment into an abstract digital twin. Our runtime instead controls what
> the agent emits at each communication boundary, combining tokenizer-level
> exact prevention with semantic and counterfactual privacy-cost accounting.

### 2. PRISM

Link: https://arxiv.org/abs/2605.10614

Why it is close:

- It is a generation-time defense for secret leakage in multi-agent LLM
  pipelines.
- It treats leakage as sequential risk accumulation.
- It intervenes before secrets are fully emitted.
- It uses token-level generation signals and risk zones.

Key difference from our proposal:

- PRISM focuses on credential/secret leakage and generation dynamics such as
  entropy collapse and logit concentration.
- Our proposal focuses on protected user attributes and policy-governed
  disclosure, including semantic abstractions and task utility.
- PRISM's central score is a calibrated per-token leakage-risk classifier.
- Our central mechanism is per-attribute privacy budget accounting over exact,
  semantic, and counterfactual costs.

Risk level: **High for the generation-time defense framing**, lower for the
full privacy-budgeted policy runtime.

Recommended positioning:

> PRISM shows that generation-time monitoring can prevent secret propagation in
> multi-agent pipelines. Our work targets broader policy-governed attribute
> disclosure and uses hard exact-value masking plus semantic/counterfactual
> privacy accounting rather than generation-dynamics-only risk scoring.

### 3. PrivacyChecker / Privacy in Action

Links:

- https://arxiv.org/abs/2509.17488
- https://www.microsoft.com/en-us/research/blog/reducing-privacy-leaks-in-ai-two-approaches-to-contextual-integrity/

Why it is close:

- It is a model-agnostic inference-time mitigation method for LLM-powered
  agents.
- It uses contextual integrity.
- It extracts information flows and classifies allow/withhold decisions.
- It integrates with agent systems as a system prompt, embedded tool, or MCP
  gate.

Key difference from our proposal:

- PrivacyChecker is primarily a contextual-integrity reasoning/checking module.
- It does not appear to provide tokenizer-aware hard decoding guarantees.
- It does not center per-attribute privacy budgets.
- It does not use counterfactual likelihood as a privacy-cost signal.

Risk level: **Medium-high**.

Recommended positioning:

> PrivacyChecker operationalizes contextual integrity through explicit
> information-flow extraction and allow/withhold judgments. Our runtime adds
> lower-level enforcement and sequential accounting: exact decoding constraints,
> semantic/counterfactual costs, and per-attribute budget updates.

### 4. Constrained Decoding for Privacy-Preserving LLM Inference

Link: https://openreview.net/forum?id=riu8VN6Do8

Why it is close:

- It proposes inference-time prevention of PII leakage through constrained
  decoding.
- It uses regex-aware logit masking.
- It argues against post-hoc filtering because PII has already been generated.

Key difference from our proposal:

- It mainly targets structured PII patterns such as emails, SSNs, IP addresses,
  and credit cards.
- Our masking compiles known protected surface forms and variants, not only PII
  regexes.
- Our runtime adds semantic leakage verification, counterfactual scoring,
  budget accounting, and rewriting.

Risk level: **High for the exact masking component**, low for the full runtime.

Recommended positioning:

> Constrained decoding provides hard prevention for structured PII patterns. We
> adopt the same inference-time prevention philosophy but extend it to
> policy-induced protected values and combine it with semantic leakage scoring
> and privacy-budgeted candidate selection.

### 5. AirGapAgent

Link: https://arxiv.org/abs/2405.05175

Why it is close:

- It is grounded in contextual integrity.
- It protects privacy-conscious conversational agents from context hijacking.
- It restricts access to only task-necessary data.

Key difference from our proposal:

- AirGapAgent focuses on restricting what data the agent can access for a task.
- Our runtime focuses on what information the agent is allowed to emit after it
  has access to private context.
- Our method includes token-level exact masking, counterfactual privacy cost,
  and per-attribute budget accounting.

Risk level: **Medium**.

Recommended positioning:

> AirGapAgent minimizes access to private data before generation. Our runtime
> controls disclosure during generation and action emission, which is
> complementary when task execution still requires private context.

## Relevant Benchmarks and Evaluation Work

### POLAR-Bench

Link: https://www.researchgate.net/publication/405044711_POLAR-Bench_A_Diagnostic_Benchmark_for_Privacy-Utility_Trade-offs_in_LLM_Agents

POLAR-Bench is directly aligned with the proposal's privacy-utility framing:
private document, allowed/task-relevant attributes, protected attributes, and
adversarial probing. It should remain a primary evaluation target.

### AgentLeak

Link: https://arxiv.org/abs/2602.11510

AgentLeak argues that privacy leakage in multi-agent systems occurs through
internal channels such as inter-agent messages, shared memory, and tool
arguments, not only final outputs. This strongly supports the proposal's
runtime-boundary framing.

### AgentDAM

Link: https://arxiv.org/abs/2503.09780

AgentDAM evaluates autonomous web agents under the principle of data
minimization. It is useful as a related benchmark and as support for the
privacy-utility trade-off framing.

### CI-Work

Link: https://arxiv.org/abs/2604.21308

CI-Work evaluates contextual integrity in enterprise LLM agents and reports
that higher task utility can correlate with more privacy violations. This
supports the need for explicit privacy-utility frontier evaluation.

### ConfAIde

Link: https://arxiv.org/abs/2310.17884

ConfAIde is an early contextual-integrity benchmark showing that LLMs leak
private information in context-sensitive situations even with privacy prompts or
chain-of-thought reasoning. It motivates inference-time privacy mechanisms.

## Component-Level Novelty Assessment

### Local policy induction

Status: partly covered by contextual-integrity and privacy-checker systems.

Many works extract or reason over information flows, roles, recipients, and
attributes. The proposal should avoid claiming that structured privacy reasoning
from context is wholly new. The differentiator is compiling induced policies
into both decoding constraints and budgeted runtime decisions.

### Tokenizer-aware exact masking

Status: not new as a general idea.

Constrained decoding, grammar-constrained decoding, and regex-aware logit
masking already exist. The proposal's defensible contribution is using this
inside a broader privacy-budgeted agent runtime for protected attribute values
and variants, rather than presenting token masking itself as the core novelty.

### Semantic leakage verifier

Status: related to contextual-integrity checkers and LLM-as-judge privacy
filters.

The proposal needs to be clear about what makes its verifier different: per
protected attribute scoring, integration into budget accounting, and use with
abstraction/rewriting.

### Counterfactual privacy cost

Status: promising but adjacent prior ideas exist.

Privacy loss, posterior leakage, counterfactual influence, and counterfactual
privacy audits exist in privacy literature. I did not find a close agent-runtime
paper using exactly the proposed conditional likelihood difference between
private and protected-attribute-free documents as a candidate-response privacy
cost. This may be one of the strongest novelty points.

### Per-attribute privacy-budgeted runtime

Status: partially overlapped.

PlanTwin uses disclosure budgets; DP literature has privacy budgets; surveys
discuss sequential composition in agents. The proposal's specific per-attribute
budget over candidate agent emissions, with costs defined by exact/semantic/
counterfactual leakage signals, still appears distinctive.

### Abstraction and rewriting

Status: common as a mitigation pattern.

Many privacy systems use redaction, anonymization, abstraction, or rewriting.
The differentiator is making rewriting the fallback of a budgeted runtime and
protecting the rewriter with exact masks.

## What Might Be "Scooped"

The following claims are unsafe:

- "First inference-time privacy runtime for LLM agents."
- "First token-level privacy-preserving decoding method."
- "First privacy-budgeted LLM agent architecture."
- "First generation-time defense against secret leakage in multi-agent systems."
- "First contextual-integrity privacy guard for LLM agents."

The following claims are more defensible:

- "A unified runtime that combines hard exact-value decoding guarantees with
  semantic/counterfactual leakage scoring and per-attribute privacy budgets."
- "A candidate-level counterfactual privacy-cost estimator for policy-governed
  LLM agent disclosure."
- "A privacy-budgeted action selection loop that chooses the highest-utility
  candidate satisfying per-attribute constraints, with abstraction-based repair."
- "An evaluation of exact masking, semantic verification, and counterfactual
  budget accounting on POLAR-style privacy-utility delegation tasks."

## Suggested Related Work Structure

1. Contextual privacy and data minimization in LLM agents:
   ConfAIde, AirGapAgent, AgentDAM, PrivacyChecker, CI-Work.
2. Multi-agent and tool-mediated leakage benchmarks:
   AgentLeak, AgentSecBench, AgentSocialBench, POLAR-Bench.
3. Inference-time and generation-time defenses:
   constrained decoding for privacy, PRISM, guardrails, output redaction.
4. Privacy accounting and posterior/counterfactual leakage:
   DP privacy loss, posterior leakage, metric-normalized posterior leakage.
5. Planning abstraction and local gatekeepers:
   PlanTwin and related local-first/cloud-assisted agent architectures.

## Recommended Rewrite of Contribution Claims

Original-style claim to avoid:

> We propose the first privacy-budgeted runtime for LLM agents.

Safer claim:

> We propose a privacy-budgeted runtime for LLM agents that unifies three
> complementary enforcement signals: tokenizer-aware exact masking for protected
> surface forms, semantic leakage verification for paraphrased disclosures, and
> counterfactual likelihood-based privacy cost for hidden dependence on protected
> attributes. The runtime performs per-attribute budget accounting over
> multi-turn agent actions and uses abstraction-based rewriting when the
> highest-utility candidate exceeds the remaining budget.

## References Checked

- PlanTwin: https://arxiv.org/abs/2603.18377
- PRISM: https://arxiv.org/abs/2605.10614
- Privacy in Action / PrivacyChecker: https://arxiv.org/abs/2509.17488
- Microsoft Research PrivacyChecker blog:
  https://www.microsoft.com/en-us/research/blog/reducing-privacy-leaks-in-ai-two-approaches-to-contextual-integrity/
- Constrained Decoding for Privacy-Preserving LLM Inference:
  https://openreview.net/forum?id=riu8VN6Do8
- AirGapAgent: https://arxiv.org/abs/2405.05175
- AgentLeak: https://arxiv.org/abs/2602.11510
- AgentDAM: https://arxiv.org/abs/2503.09780
- CI-Work: https://arxiv.org/abs/2604.21308
- ConfAIde: https://arxiv.org/abs/2310.17884
- Metric-Normalized Posterior Leakage:
  https://arxiv.org/abs/2605.01137

