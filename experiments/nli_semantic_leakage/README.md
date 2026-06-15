# NLI Semantic Leakage Probe

This is a small standalone experiment for checking whether a post-hard-mask
answer still leaks sensitive attributes through semantic implications.

The core idea is:

```text
semantic_leakage = entailment_confidence(answer => predicate) * information_gain_bits(predicate)
```

Examples:

- "She is a minor" does not reveal `age = 14`, but it can entail `age < 18`.
- "She lives in Guangdong" does not reveal `city = Shenzhen`, but it narrows the
  candidate city set.

## Run

From the repo root:

```powershell
python experiments\nli_semantic_leakage\nli_leakage_demo.py
```

The default model is:

```text
MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
```

It is multilingual and works better for Chinese than English-only MNLI models.
The first run may download the model from Hugging Face.

You can choose another model:

```powershell
python experiments\nli_semantic_leakage\nli_leakage_demo.py --model joeddav/xlm-roberta-large-xnli
```

If you cannot download a model yet, run the local scoring pipeline with a tiny
hand-written baseline:

```powershell
python experiments\nli_semantic_leakage\nli_leakage_demo.py --backend heuristic
```

This does not validate NLI model quality. It only checks that the leakage score
calculation and examples behave as expected.

## Reading the Output

For each answer/predicate pair, the script prints:

- `entailment`: model confidence that the answer implies the predicate.
- `info_gain_bits`: how much the predicate narrows the protected variable.
- `weighted_score`: `entailment * info_gain_bits`.

Use `entailment` as the semantic-leak confidence, and `weighted_score` as a
rough severity score.

## Caveats

NLI models are not perfect judges. They often do well on direct paraphrase and
range implications, but geography such as "Shenzhen implies Guangdong" may
depend on world knowledge. For robust experiments, combine NLI with a small
ontology or predicate lattice for attributes like age, city, province, school,
employer, and address.
