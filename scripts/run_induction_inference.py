from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_runtime.induction_data import (  # noqa: E402
    extract_json_object,
    get_system_prompt,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot, few-shot, or LoRA-adapter induction inference."
    )
    parser.add_argument("--input", required=True, help="Input split JSONL.")
    parser.add_argument("--output", required=True, help="Predictions JSONL.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", default=None, help="Optional PEFT/LoRA adapter path.")
    parser.add_argument("--few-shot-file", default=None, help="JSONL file for few-shot examples.")
    parser.add_argument("--num-shots", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--target-schema",
        choices=["value", "key_value", "protected_key_value"],
        default="value",
        help="Expected output schema for the system prompt.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "run_induction_inference.py requires transformers and torch. "
            "Install requirements-train.txt on RunPod."
        ) from exc

    quantization_config = None
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError("4-bit loading requires bitsandbytes support") from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype_map[args.torch_dtype],
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
    )

    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Loading --adapter requires peft") from exc
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, record: dict, shots: list[dict], *, target_schema: str) -> str:
    messages = [{"role": "system", "content": get_system_prompt(target_schema)}]
    for shot in shots:
        messages.append({"role": "user", "content": str(shot["input"])})
        messages.append({"role": "assistant", "content": str(shot["target_text"])})
    messages.append({"role": "user", "content": str(record["input"])})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    blocks = []
    for message in messages:
        blocks.append(f"{message['role'].upper()}:\n{message['content']}")
    blocks.append("ASSISTANT:")
    return "\n\n".join(blocks)


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    shots: list[dict] = []
    if args.num_shots > 0:
        if not args.few_shot_file:
            raise ValueError("--few-shot-file is required when --num-shots > 0")
        shots = read_jsonl(args.few_shot_file)[: args.num_shots]

    model, tokenizer = load_model_and_tokenizer(args)
    outputs = []
    for index, record in enumerate(records, start=1):
        prompt = build_prompt(tokenizer, record, shots, target_schema=args.target_schema)
        prediction_text = generate_one(model, tokenizer, prompt, args.max_new_tokens)
        prediction, parse_error = extract_json_object(prediction_text)
        row = {
            "sample_id": record.get("sample_id"),
            "domain": record.get("domain"),
            "gold": record.get("target"),
            "prediction_text": prediction_text,
            "prediction": prediction or {},
            "parse_error": parse_error,
            "model": args.model,
            "adapter": args.adapter,
            "num_shots": args.num_shots,
            "target_schema": args.target_schema,
        }
        outputs.append(row)
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(records),
                    "sample_id": record.get("sample_id"),
                    "parsed": prediction is not None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_jsonl(args.output, outputs)


if __name__ == "__main__":
    main()
