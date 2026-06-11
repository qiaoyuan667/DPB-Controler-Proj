from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_runtime.induction_data import SYSTEM_PROMPT, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA/SFT training for the POLAR P1 minimal policy inducer."
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-4bit", action="store_true", help="Disable QLoRA 4-bit loading.")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training.")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def apply_chat_template(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    blocks = []
    for message in messages:
        blocks.append(f"{message['role'].upper()}:\n{message['content']}")
    if add_generation_prompt:
        blocks.append("ASSISTANT:")
    return "\n\n".join(blocks)


def prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(record["input"])},
    ]


def full_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        *prompt_messages(record),
        {"role": "assistant", "content": str(record["target_text"])},
    ]


def tokenize_record(tokenizer, record: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    prompt_text = apply_chat_template(
        tokenizer,
        prompt_messages(record),
        add_generation_prompt=True,
    )
    full_text = apply_chat_template(
        tokenizer,
        full_messages(record),
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    tokenized = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = list(input_ids)
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_training_arguments(TrainingArguments, args: argparse.Namespace):
    kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": 3,
        "load_best_model_at_end": False,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": "none",
        "optim": "paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
    }

    parameters = inspect.signature(TrainingArguments).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return TrainingArguments(**kwargs)


@dataclass
class CausalJsonDataCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_length)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_length)
            batch["labels"].append(feature["labels"] + [-100] * pad_length)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "train_inducer_qlora.py requires requirements-train.txt on RunPod."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
    )
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_records = read_jsonl(args.train_file)
    val_records = read_jsonl(args.val_file)
    train_dataset = Dataset.from_list(
        [tokenize_record(tokenizer, record, args.max_length) for record in train_records]
    )
    val_dataset = Dataset.from_list(
        [tokenize_record(tokenizer, record, args.max_length) for record in val_records]
    )

    training_args = build_training_arguments(TrainingArguments, args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CausalJsonDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
