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
pip install -r requirements-train.txt
```

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
  --model Qwen/Qwen3-1.7B-Instruct \
  --load-in-4bit

python scripts/eval_induction_predictions.py \
  --predictions runs/induction/qwen3_1p7b_zero_shot.jsonl
```

## 4. Few-Shot Baseline

```bash
python scripts/run_induction_inference.py \
  --input data/induction/p1_scoring_targets/test.jsonl \
  --output runs/induction/qwen3_1p7b_few_shot_3.jsonl \
  --model Qwen/Qwen3-1.7B-Instruct \
  --few-shot-file data/induction/p1_scoring_targets/train.jsonl \
  --num-shots 3 \
  --load-in-4bit

python scripts/eval_induction_predictions.py \
  --predictions runs/induction/qwen3_1p7b_few_shot_3.jsonl
```

## 5. QLoRA SFT

```bash
python scripts/train_inducer_qlora.py \
  --train-file data/induction/p1_scoring_targets/train.jsonl \
  --val-file data/induction/p1_scoring_targets/val.jsonl \
  --output-dir runs/induction/qwen3_1p7b_qlora \
  --model Qwen/Qwen3-1.7B-Instruct \
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
  --model Qwen/Qwen3-1.7B-Instruct \
  --adapter runs/induction/qwen3_1p7b_qlora \
  --load-in-4bit

python scripts/eval_induction_predictions.py \
  --predictions runs/induction/qwen3_1p7b_qlora_test.jsonl
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
