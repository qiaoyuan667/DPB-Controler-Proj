# RunPod: POLAR P1 Minimal Induction Training

This workflow trains a local Qwen3-1.7B policy inducer to predict the minimal
POLAR-aligned target:

```json
{
  "scoring_targets": {
    "allowed_values": [],
    "do_not_disclose_values": []
  }
}
```

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
uses text-only Qwen models, so the simplest fix is to remove `torchvision`:

```bash
pip uninstall -y torchvision
python - <<'PY'
import torch
print("torch", torch.__version__)
PY
```

Then rerun the inference or training command. If another project on the same Pod
needs vision models, install a `torchvision` wheel that exactly matches the
installed PyTorch/CUDA version instead of uninstalling it.

Confirm the repaired dataset is the real JSON file, not a Git LFS pointer:

```bash
ls -lh polar_bench_upstream/data/privacy_benchmark_rendered_repaired.json
head -n 2 polar_bench_upstream/data/privacy_benchmark_rendered_repaired.json
```

The file should be hundreds of MB and start with `[` followed by JSON objects. If
it starts with `version https://git-lfs.github.com/spec/v1`, run the Git LFS
commands above again inside the repository.

## 2. Build P1 Dataset

Use the repaired rendered POLAR-Bench dataset:

```bash
python scripts/build_polar_induction_dataset.py \
  --input polar_bench_upstream/data/privacy_benchmark_rendered_repaired.json \
  --output-dir data/induction/p1_scoring_targets \
  --seed 42
```

Expected outputs:

```text
data/induction/p1_scoring_targets/train.jsonl
data/induction/p1_scoring_targets/val.jsonl
data/induction/p1_scoring_targets/test.jsonl
data/induction/p1_scoring_targets/metadata.json
```

## 3. Zero-Shot Baseline

```bash
python scripts/run_induction_inference.py \
  --input data/induction/p1_scoring_targets/test.jsonl \
  --output runs/induction/qwen3_1p7b_zero_shot.jsonl \
  --model Qwen/Qwen3-1.7B \
  --torch-dtype bfloat16

python scripts/eval_induction_predictions.py \
  --predictions runs/induction/qwen3_1p7b_zero_shot.jsonl
```

## 4. Few-Shot Baseline

```bash
python scripts/run_induction_inference.py \
  --input data/induction/p1_scoring_targets/test.jsonl \
  --output runs/induction/qwen3_1p7b_few_shot_3.jsonl \
  --model Qwen/Qwen3-1.7B \
  --few-shot-file data/induction/p1_scoring_targets/train.jsonl \
  --num-shots 3 \
  --torch-dtype bfloat16

python scripts/eval_induction_predictions.py \
  --predictions runs/induction/qwen3_1p7b_few_shot_3.jsonl
```

## 5. QLoRA SFT

```bash
python scripts/train_inducer_qlora.py \
  --train-file data/induction/p1_scoring_targets/train.jsonl \
  --val-file data/induction/p1_scoring_targets/val.jsonl \
  --output-dir runs/induction/qwen3_1p7b_qlora \
  --model Qwen/Qwen3-1.7B \
  --bf16 \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --max-length 4096
```

## 6. Adapter Inference and Evaluation

```bash
python scripts/run_induction_inference.py \
  --input data/induction/p1_scoring_targets/test.jsonl \
  --output runs/induction/qwen3_1p7b_qlora_test.jsonl \
  --model Qwen/Qwen3-1.7B \
  --adapter runs/induction/qwen3_1p7b_qlora \
  --torch-dtype bfloat16

python scripts/eval_induction_predictions.py \
  --predictions runs/induction/qwen3_1p7b_qlora_test.jsonl
```

## 7. Epoch Ablation and Run Comparison

Train 1/2/3 epoch adapters in separate directories:

```bash
for EPOCHS in 1 2 3; do
  python scripts/train_inducer_qlora.py \
    --train-file data/induction/p1_scoring_targets/train.jsonl \
    --val-file data/induction/p1_scoring_targets/val.jsonl \
    --output-dir runs/induction/qwen3_1p7b_lora_no4bit_${EPOCHS}epoch_v2 \
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

Run each adapter on the test split:

```bash
for EPOCHS in 1 2 3; do
  python scripts/run_induction_inference.py \
    --input data/induction/p1_scoring_targets/test.jsonl \
    --output runs/induction/qwen3_1p7b_lora_no4bit_${EPOCHS}epoch_v2_test.jsonl \
    --model Qwen/Qwen3-1.7B \
    --adapter runs/induction/qwen3_1p7b_lora_no4bit_${EPOCHS}epoch_v2 \
    --torch-dtype bfloat16

  python scripts/eval_induction_predictions.py \
    --predictions runs/induction/qwen3_1p7b_lora_no4bit_${EPOCHS}epoch_v2_test.jsonl
done
```

Compare zero-shot, few-shot, and LoRA runs:

```bash
python scripts/compare_induction_runs.py \
  runs/induction/qwen3_1p7b_zero_shot.jsonl.summary.json \
  runs/induction/qwen3_1p7b_few_shot_3.jsonl.summary.json \
  runs/induction/qwen3_1p7b_lora_no4bit_1epoch_v2_test.jsonl.summary.json \
  runs/induction/qwen3_1p7b_lora_no4bit_2epochs_v2_test.jsonl.summary.json \
  runs/induction/qwen3_1p7b_lora_no4bit_3epochs_v2_test.jsonl.summary.json \
  --output runs/induction/induction_comparison.csv
```

If `bitsandbytes` fails with `Missing dependency: libnvJitLink.so.13`, do not
use `--load-in-4bit` for Qwen3-1.7B inference. The model is small enough to run
in bf16/fp16 on common 24GB GPUs. For training, either fix the CUDA/bitsandbytes
stack or run LoRA without 4-bit:

For 24GB GPUs, start with `--max-length 2048`. If CUDA reports fragmentation,
set:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

```bash
python scripts/train_inducer_qlora.py \
  --train-file data/induction/p1_scoring_targets/train.jsonl \
  --val-file data/induction/p1_scoring_targets/val.jsonl \
  --output-dir runs/induction/qwen3_1p7b_lora_no4bit \
  --model Qwen/Qwen3-1.7B \
  --no-4bit \
  --bf16 \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --max-length 2048
```

## Primary Metric

Use `do_not_disclose_values.recall` as the primary metric. A missed protected
value cannot be recovered by downstream runtime checks.

Secondary metrics:

- JSON parse rate
- schema validity
- `do_not_disclose_values` precision / F1
- `allowed_values` precision / recall / F1
- exact set match
