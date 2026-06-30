# Section 5: Counterfactual Privacy Cost

## Definition

For protected fact `a`, candidate response `z`, and valid interventions
`D_tilde[a, j]`, the implementation computes:

```text
L[a, j](z) = log P(z | D_private) - log P(z | D_tilde[a, j])
C[a](z)    = max(0, max_j L[a, j](z))
```

The result is per fact, not per benchmark instance. Only fact `a` is changed in
`D_tilde[a, j]`; all other protected and task-relevant facts are preserved. The
default aggregation is `max`. `mixture` and `mean` are exposed for ablations.

Unnormalized `costs` are measured in nats and are passed to
`PrivacyAccountant`. `normalized_costs` divide by candidate token count and are
diagnostics only.

## Policy Schema

```json
{
  "facts": [
    {
      "fact_id": "age",
      "fact": "18 years old",
      "surface_forms": ["age 18"],
      "fact_type": "age",
      "privacy_scope": "value",
      "allowed_abstractions": ["an adult"],
      "counterfactual_values": ["35 years old", "60 years old"],
      "default_budget": 2.0
    }
  ]
}
```

The extractor should provide type-preserving `counterfactual_values`. Raw
protected values remain controller-side metadata and must not be serialized
into a counterfactual model prompt.

## Runtime API

```python
estimator = CounterfactualPrivacyCostEstimator(
    scorer=HFCausalLMLikelihoodScorer(model, tokenizer),
    aggregation="max",
)

result = estimator.estimate_policy(
    private_document,
    policy,
    candidate,
    prompt_builder=render_model_a_prompt,
)

if accountant.can_spend(result.costs, "final"):
    accountant.spend(result.costs, "final")
```

`prompt_builder` changes only the private document. The task, non-secret policy
text, and released dialogue history stay fixed. By default the estimator fails
if a counterfactual prompt still contains the target protected value. In a
multi-turn run, values already present in public history must be declared with
`allowed_prompt_occurrences`.

## RunPod Smoke Run

Install dependencies inside the persistent environment:

```bash
pip install -r requirements-hf.txt
```

Run from the repository root:

```bash
python examples/counterfactual_cost_hf.py \
  --model swiss-ai/Apertus-8B-Instruct-2509 \
  --document inputs/counterfactual_document.example.txt \
  --policy inputs/counterfactual_policy.example.json \
  --candidate-file inputs/counterfactual_candidate.example.txt \
  --aggregation max \
  --counterfactual-modes placeholder abstraction substitution \
  --dtype bfloat16 \
  --device-map auto \
  --allow-download \
  --output counterfactual_results/apertus_smoke.json
```

After the model is cached under `/workspace`, omit `--allow-download` for an
offline run.

## Audit Output

The JSON result records private likelihood, per-fact costs, every intervention,
per-intervention likelihoods, per-token signed losses, positive token mass, and
invalid counterfactual validation errors.

## Claim Boundary

This implementation estimates policy-conditioned counterfactual dependence. It
does not by itself establish differential privacy: it evaluates one candidate
against a finite intervention set using model likelihoods. A formal DP claim
would additionally require bounding the full defended mechanism over all
adjacent contexts and possible outputs.
