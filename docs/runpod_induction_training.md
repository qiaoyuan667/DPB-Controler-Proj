# RunPod: Protected-Only Policy Induction Training

This workflow trains a local Qwen3-1.7B policy inducer to extract only the
values that the privacy policy explicitly protects.

Current target schema:

```json
{
  "policy_targets": {
    "protected_values": [
      {"key": "email", "value": "name@example.com"}
    ]
  }
}
```

The old `allowed_values` target is intentionally not used in the current runtime
training flow.

## 1. Install Dependencies

From the repository root:

```bash
git submodule update --init --recursive
apt-get update && apt-get install -y git-lfs
git lfs install
git -C polar_bench_upstream lfs pull
pip install -r requirements-train.txt
```

If model loading fails with `RuntimeError: operator torchvision::nms does not
exist`, the RunPod image has an incompatible `torchvision` package. This project
uses text-only Qwen models, so the simplest fix is:

```bash
pip uninstall -y torchvision
python - <<'PY'
import torch
print("torch", torch.__version__)
PY
```

Confirm the repaired POLAR file is real JSON, not a Git LFS pointer:

```bash
ls -lh polar_bench_upstream/data/privacy_benchmark_rendered_repaired.json
head -n 2 polar_bench_upstream/data/privacy_benchmark_rendered_repaired.json
```

The file should be hundreds of MB and start with `[` followed by JSON objects.

## 2. Build Current Datasets

Build POLAR P1 with the protected-only key-value target:

```bash
python scripts/build_polar_induction_dataset.py \
  --input polar_bench_upstream/data/privacy_benchmark_rendered_repaired.json \
  --output-dir data/induction/p1_protected_key_value \
  --target-schema protected_key_value \
  --seed 42
```

Build the synthetic counterfactual dataset:

```bash
python scripts/build_synthetic_counterfactual_dataset.py \
  --output-dir data/induction/synthetic_counterfactual_v3_protected_key_value \
  --target-schema protected_key_value \
  --synthetic-mode protected_only \
  --num-base-docs 250 \
  --policies-per-doc 4 \
  --seed 42
```

Build the mixed training split used for the current adapter:

```bash
python scripts/build_synthetic_counterfactual_dataset.py \
  --output-dir data/induction/mixed_p1_synthetic_v3_protected_key_value \
  --polar-dir data/induction/p1_protected_key_value \
  --target-schema protected_key_value \
  --synthetic-mode protected_only \
  --num-base-docs 250 \
  --policies-per-doc 4 \
  --seed 42
```

## 3. Baselines

Zero-shot:

```bash
python scripts/run_induction_inference.py \
  --input data/induction/mixed_p1_synthetic_v3_protected_key_value/test.jsonl \
  --output runs/induction/qwen3_1p7b_zero_shot_protected_kv.jsonl \
  --model Qwen/Qwen3-1.7B \
  --target-schema protected_key_value \
  --torch-dtype bfloat16

python scripts/eval_induction_predictions.py \
  --predictions runs/induction/qwen3_1p7b_zero_shot_protected_kv.jsonl
```

Few-shot:

```bash
python scripts/run_induction_inference.py \
  --input data/induction/mixed_p1_synthetic_v3_protected_key_value/test.jsonl \
  --output runs/induction/qwen3_1p7b_few_shot_3_protected_kv.jsonl \
  --model Qwen/Qwen3-1.7B \
  --few-shot-file data/induction/mixed_p1_synthetic_v3_protected_key_value/train.jsonl \
  --num-shots 3 \
  --target-schema protected_key_value \
  --torch-dtype bfloat16

python scripts/eval_induction_predictions.py \
  --predictions runs/induction/qwen3_1p7b_few_shot_3_protected_kv.jsonl
```

## 4. LoRA Training

Train 1, 2, and 3 epoch adapters:

```bash
for EPOCHS in 1 2 3; do
  python scripts/train_inducer_qlora.py \
    --train-file data/induction/mixed_p1_synthetic_v3_protected_key_value/train.jsonl \
    --val-file data/induction/mixed_p1_synthetic_v3_protected_key_value/val.jsonl \
    --output-dir runs/induction/qwen3_1p7b_lora_mixed_v3_protected_kv_${EPOCHS}epoch \
    --model Qwen/Qwen3-1.7B \
    --no-4bit \
    --bf16 \
    --epochs ${EPOCHS} \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-4 \
    --max-length 2048
done
```

If bf16 is unavailable, replace `--bf16` with `--fp16`. If CUDA memory is tight,
reduce `--max-length` to `1536`.

## 5. Adapter Inference and Evaluation

```bash
for EPOCHS in 1 2 3; do
  python scripts/run_induction_inference.py \
    --input data/induction/mixed_p1_synthetic_v3_protected_key_value/test.jsonl \
    --output runs/induction/qwen3_1p7b_lora_mixed_v3_protected_kv_${EPOCHS}epoch_test.jsonl \
    --model Qwen/Qwen3-1.7B \
    --adapter runs/induction/qwen3_1p7b_lora_mixed_v3_protected_kv_${EPOCHS}epoch \
    --target-schema protected_key_value \
    --torch-dtype bfloat16

  python scripts/eval_induction_predictions.py \
    --predictions runs/induction/qwen3_1p7b_lora_mixed_v3_protected_kv_${EPOCHS}epoch_test.jsonl
done
```

Compare run summaries:

```bash
python scripts/compare_induction_runs.py \
  runs/induction/qwen3_1p7b_zero_shot_protected_kv.jsonl.summary.json \
  runs/induction/qwen3_1p7b_few_shot_3_protected_kv.jsonl.summary.json \
  runs/induction/qwen3_1p7b_lora_mixed_v3_protected_kv_1epoch_test.jsonl.summary.json \
  runs/induction/qwen3_1p7b_lora_mixed_v3_protected_kv_2epoch_test.jsonl.summary.json \
  runs/induction/qwen3_1p7b_lora_mixed_v3_protected_kv_3epoch_test.jsonl.summary.json \
  --output runs/induction/induction_comparison_protected_kv.csv
```

## 6. Custom Policy Inference

For a custom CV or document JSONL:

```bash
python scripts/run_induction_inference.py \
  --input data/custom/cv_policy_input.jsonl \
  --output runs/induction/cv_001_mixed_v3_protected_kv_2epoch.jsonl \
  --model Qwen/Qwen3-1.7B \
  --adapter runs/induction/qwen3_1p7b_lora_mixed_v3_protected_kv_2epoch \
  --target-schema protected_key_value \
  --torch-dtype bfloat16
```

The output JSONL contains `prediction.policy_targets.protected_values`.

## Primary Metric

Use `protected_values.recall` as the primary metric. A missed protected value
cannot be recovered by downstream runtime checks.

Secondary metrics:

- JSON parse rate
- schema validity
- `protected_values` precision / F1
- pair precision / recall / F1
- value precision / recall / F1
- key accuracy on matched values
