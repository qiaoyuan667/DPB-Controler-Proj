# POLAR Induction Minimal Target

This note records the first-stage dataset decision for training the local policy
inducer on POLAR-Bench.

## Decision

For the first version, the policy inducer predicts the POLAR-aligned
`scoring_targets` object only:

```json
{
  "scoring_targets": {
    "allowed_values": [],
    "do_not_disclose_values": []
  }
}
```

This is intentionally minimal. POLAR-Bench ultimately scores privacy and utility
using these two lists, so this target directly aligns the inducer with the
benchmark objective.

## Dataset Source

Use the repaired rendered dataset:

```text
polar_bench_upstream/data/privacy_benchmark_rendered_repaired.json
```

Do not use `privacy_benchmark_rendered.jsonl` as the primary source for this
experiment, because the repaired rendered file is the benchmark-correct version.

## Initial Filter

For the simplest P1 experiment:

```text
metadata.privacy_level == 1
metadata.privacy_type == "explicit_field_constraints"
```

In the local repaired dataset this gives 1571 examples.

## Model Input

Each training example should include only the fields available to a real local
inducer:

```text
Source document:
{generated_texts.source_document_text}

Privacy policy:
{generated_texts.privacy_policy_text}

Task instruction:
{generated_texts.task_instruction_text}
```

The model must not see `hidden_target`, `scoring_targets`, or generation
metadata in the input.

## Model Target

The target is copied from:

```text
scoring_targets.allowed_values
scoring_targets.do_not_disclose_values
```

Target format:

```json
{
  "scoring_targets": {
    "allowed_values": [
      "lower back pain",
      "6 months",
      "morning"
    ],
    "do_not_disclose_values": [
      "Noah Baumann",
      "noah.baumann@alpineai.example.com",
      "+41 76 807 8673",
      "1968-11-23",
      "H-133326",
      "AlpineAI"
    ]
  }
}
```

## Why Keep the `scoring_targets` Wrapper?

Keeping the wrapper makes the inducer output isomorphic to the POLAR scoring
fields. This reduces ambiguity later when the same predicted object is consumed
by:

- induction evaluation,
- runtime policy construction,
- POLAR privacy/utility scoring,
- RunPod training and inference scripts.

The wrapper also makes it clear that this first-stage model is predicting the
minimal benchmark-aligned target, not the full future policy schema.

## What This Target Supports

The minimal predicted target is enough to support:

- candidate-level exact screening,
- protected-value replacement in counterfactual document construction,
- simple runtime policy construction,
- POLAR-aligned privacy and utility evaluation,
- zero-shot, few-shot, and SFT/LoRA/QLoRA baselines.

Mapping:

```text
do_not_disclose_values -> protected values
allowed_values         -> task-relevant allowed values
```

## What This Target Does Not Yet Include

The minimal target does not include:

- protected field names,
- allowed field names,
- partial disclosure rules,
- safe abstractions,
- semantic hints,
- per-field budgets,
- channel budgets,
- tokenizer-normalized surface variants.

These can be added later from `hidden_target` and
`source_document_inputs.safe_abstractions_available`, but they should not be part
of the first SFT target.

## Post-Processing for Runtime Use

After prediction, a deterministic post-processor can convert the minimal target
into a runtime policy:

```text
for value in scoring_targets.do_not_disclose_values:
    create ProtectedFact(
        fact=value,
        surface_forms=surface_variants(value),
        default_budget=0.0 or configured_budget
    )
```

Allowed values can be stored separately for utility checks and for preserving
task-relevant content during counterfactual construction.

## Evaluation Metrics for the Inducer

Primary:

```text
do_not_disclose_values recall
```

This matters most because a missed protected value cannot be recovered by
downstream runtime checks.

Secondary:

```text
JSON parse rate
schema validity
do_not_disclose_values precision / F1
allowed_values precision / recall / F1
exact set match
normalized set match
downstream POLAR privacy score
downstream POLAR utility score
```

## Planned Baselines

First-stage baselines:

```text
1. heuristic inducer
2. Qwen3-1.7B-Instruct zero-shot
3. Qwen3-1.7B-Instruct few-shot
4. Qwen3-1.7B-Instruct SFT / LoRA / QLoRA
5. oracle scoring_targets from POLAR
```

The oracle target is not a deployable inducer, but it provides an upper bound
for the runtime when protected and allowed values are perfectly known.

