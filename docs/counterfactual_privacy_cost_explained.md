# Counterfactual Privacy Cost Explained

This note explains the core idea of **counterfactual privacy cost** and why it
can be positioned as a central innovation of the privacy-budgeted runtime
proposal.

## Core Intuition

Counterfactual privacy cost does not ask:

> Does the response contain a sensitive string?

It asks a deeper question:

> Is this response more likely because the model saw protected information?

This matters because privacy leakage is often not verbatim. A model may avoid
the exact protected value but still reveal it through paraphrase, implication,
or context-specific dependence.

For example, suppose the private document contains:

```text
The acquisition budget is $2M.
The target market is healthcare.
```

The protected attribute is:

```text
budget = $2M
```

The task-relevant allowed attribute is:

```text
target market = healthcare
```

Now consider this candidate response:

```text
We should target healthcare organizations with a low-seven-figure acquisition plan.
```

This response does not directly say `$2M`, so exact string masking may not catch
it. However, the phrase `low-seven-figure acquisition plan` likely depends on
knowing the protected budget. This is a semantic or inferential leak.

Counterfactual privacy cost tries to detect that dependence.

## Counterfactual Documents

The runtime constructs one or more counterfactual versions of the private
document. These counterfactual documents remove, replace, or abstract protected
attributes while preserving task-relevant allowed information.

Original private document:

```text
The acquisition budget is $2M.
The target market is healthcare.
```

Counterfactual document:

```text
The acquisition budget is [PRIVATE_BUDGET].
The target market is healthcare.
```

The important point is that the counterfactual document does not erase the
whole private context. It removes protected information while retaining the
allowed information needed for the task.

Therefore, the score estimates dependence on protected information, not
dependence on the entire document.

## Likelihood Comparison

Given a candidate response `z`, the runtime compares two conditional
likelihoods:

```text
P(z | original private document)
```

and:

```text
P(z | counterfactual document without protected values)
```

If the response is much more likely under the original private document than
under the counterfactual document, then the response probably depends on
protected information.

Formally, for instance `i`, protected fact `a`, and intervention `j`, define:

```latex
L_{i,a}^{(j)}(z)
=
\log P_\theta(z \mid D_i, P_i, T_{\mathrm{instr},i}, H_{i,<k})
-
\log P_\theta(z \mid \widetilde{D}_{i,a}^{(j)}, P_i, T_{\mathrm{instr},i}, H_{i,<k})
```

where:

- `D_i` is the original private document,
- `\widetilde{D}_{i,a}^{(j)}` changes only protected fact `a`,
- `P_i` is the privacy policy,
- `T_{\mathrm{instr},i}` is the task instruction,
- `H_{i,<k}` is the dialogue history before step `k`,
- `z` is the candidate response,
- `P_\theta` is the language model likelihood.

For a tokenized candidate response:

```latex
z = (z_1, z_2, \ldots, z_T),
```

the conditional likelihood can be written as:

```latex
\log P_\theta(z \mid D, P_i, T_{\mathrm{instr},i}, H_{i,<k})
=
\sum_{t=1}^{T}
\log p_\theta
\left(
z_t
\mid
D, P_i, T_{\mathrm{instr},i}, H_{i,<k}, z_{<t}
\right).
```

To avoid relying on a single redaction strategy, the runtime can construct
multiple counterfactuals and use the worst-case loss:

```latex
L_{i,a}^{\max}(z)
=
\max_{j \in \{1,\ldots,J_a\}}
L_{i,a}^{(j)}(z),
\qquad
C_{i,a}(z)=\max\{0,L_{i,a}^{\max}(z)\}.
```

If:

```latex
C_{i,a}(z) > b_a^{(k)},
```

then the candidate is treated as privacy-risky and can be rejected, penalized,
or rewritten.

## Examples

### Example 1: Direct Leakage

Private document:

```text
The patient has HIV.
```

Candidate:

```text
The patient has HIV.
```

This is direct surface-form leakage. Tokenizer-aware exact masking should block
it before the full protected value is generated. Counterfactual privacy cost is
not the main defense for this case.

### Example 2: Semantic Leakage

Private document:

```text
The patient has HIV.
```

Candidate:

```text
The patient is receiving antiretroviral therapy.
```

The candidate does not directly say `HIV`, but it strongly suggests the same
protected medical condition. Under the original document, this response may be
highly likely. Under a counterfactual document where the diagnosis is removed
or abstracted, the response may be much less likely. The resulting
counterfactual privacy cost would be high.

### Example 3: Safe Abstraction

Private document:

```text
The patient has HIV.
```

Candidate:

```text
The patient has a sensitive medical condition.
```

This response is more abstract. It may remain plausible under both the original
document and the counterfactual document. Therefore, the likelihood gap may be
small, and the counterfactual privacy cost may remain low.

### Example 4: Task-Relevant Allowed Information

Private document:

```text
The client works in healthcare.
The client's budget is $2M.
```

Protected attribute:

```text
budget = $2M
```

Allowed task attribute:

```text
industry = healthcare
```

Candidate:

```text
The client works in healthcare.
```

Because the counterfactual document preserves task-relevant allowed
information, `healthcare` remains available. The candidate should have similar
likelihood under both the private and counterfactual contexts. Thus, its
counterfactual privacy cost should be low.

This is important for preserving utility.

## Why This Is Different From Existing Defenses

### Exact Masking

Exact masking asks:

> Would the next token complete a protected surface form?

It provides a hard guarantee for known protected strings. But it cannot catch
paraphrases, implications, or approximate descriptions unless those variants
are explicitly included.

### Semantic Verifier

A semantic verifier asks:

> Does the candidate response entail or imply a protected fact?

This can catch paraphrases, but it depends heavily on verifier calibration and
may be brittle across domains.

### Counterfactual Privacy Cost

Counterfactual privacy cost asks:

> Would this response still be likely if the protected information had not been
> present?

This targets dependence rather than surface form. It can flag responses that
are risky because they are grounded in protected information, even when the
protected value is not directly stated.

The three defenses are complementary:

```text
Exact masking:
  You cannot say the protected string.

Semantic verification:
  You cannot say the same protected fact in different words.

Counterfactual privacy cost:
  If your response only makes sense because you saw the protected fact, it is
  risky even if no explicit string appears.
```

## Why It Helps the Privacy-Utility Trade-Off

The counterfactual document preserves task-relevant allowed attributes while
removing protected attributes. This lets the runtime distinguish:

- responses that depend on allowed task information,
- responses that depend on protected information.

This distinction is crucial. A privacy system that simply blocks anything
related to the private document will over-refuse and destroy utility. A useful
runtime should allow task-relevant information while suppressing protected
information.

Counterfactual privacy cost supports this goal by measuring whether the
candidate's likelihood changes specifically when protected attributes are
removed.

## Role in the Privacy-Budgeted Runtime

The runtime combines counterfactual privacy cost with exact and semantic risk
signals:

```latex
\rho_a(z)
=
\max
\left\{
\rho^{\mathrm{exact}}_a(z),
\lambda_{\mathrm{sem}} q_a(z),
\lambda_{\mathrm{cf}} C_{i,a}(z)
\right\}.
```

Here:

- `\rho^{\mathrm{exact}}_a(z)` is infinite if `z` directly contains a protected
  surface form for attribute `a`,
- `q_a(z)` is the semantic leakage score,
- `C_{i,a}(z)` is the positive worst-case counterfactual cost for fact `a`,
- `\lambda_{\mathrm{sem}}` and `\lambda_{\mathrm{cf}}` scale the semantic and
  counterfactual signals.

The `C_{i,a}(z)` term means that a candidate only consumes
counterfactual privacy budget when it is more likely under the private document
than under the counterfactual document.

The runtime then chooses the highest-utility candidate that satisfies all
per-attribute budget constraints.

## Limitations

Counterfactual privacy cost should be presented as a practical privacy-risk
estimator, not as a formal differential privacy guarantee.

Main limitations:

1. It depends on the quality of the counterfactual documents.
2. It depends on the calibration of the likelihood model.
3. It may over-penalize responses that are correlated with protected facts but
   still task-appropriate.
4. It may under-detect leakage when the counterfactual still preserves too much
   information.
5. It is more computationally expensive than exact masking.
6. It does not by itself solve multi-turn composition; it needs budget
   accounting.

## Recommended Positioning

The proposal should not claim that counterfactual privacy cost is a complete
formal privacy mechanism.

A safer and stronger formulation is:

> Counterfactual privacy cost is a candidate-level privacy dependence estimator.
> It measures whether a response becomes substantially more likely under the
> original private context than under protected-attribute-free counterfactual
> contexts. This allows the runtime to detect paraphrased, implied, or
> context-dependent leakage that exact surface-form filters miss, while
> preserving utility for task-relevant attributes retained in the
> counterfactual documents.

## Short Presentation Version

One-sentence version:

> Counterfactual privacy cost asks whether the model would still have generated
> the same response if the protected information had been removed.

Three-sentence version:

> Exact masking catches direct strings. Semantic verification catches obvious
> paraphrases. Counterfactual privacy cost catches dependence: it compares the
> likelihood of a candidate response under the original private document and
> under protected-information-free counterfactual documents, flagging responses
> that only become likely when the protected fact is present.
