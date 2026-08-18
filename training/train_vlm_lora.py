from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a VLM with LoRA/QLoRA using TRL")
    parser.add_argument("dataset", type=Path, help="Prepared dataset.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/triscan-lora"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--no-qlora", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from PIL import Image
        from transformers import BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Install training dependencies with: uv sync --extra local --extra train"
        ) from exc

    dataset_path = args.dataset.resolve()
    base = dataset_path.parent
    train_rows = []
    eval_rows = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        images = []
        for relative_path in row["images"]:
            with Image.open(base / relative_path) as source:
                images.append(source.convert("RGB").copy())
        prompt_content = [{"type": "image"} for _ in images]
        prompt_content.append({"type": "text", "text": row["prompt"]})
        sample = {
            "images": images,
            "prompt": [{"role": "user", "content": prompt_content}],
            "completion": [
                {"role": "assistant", "content": [{"type": "text", "text": row["completion"]}]}
            ],
        }
        (eval_rows if row.get("split") in {"val", "validation"} else train_rows).append(sample)

    if not train_rows:
        raise SystemExit("No training samples found")
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    quantization_config = None
    if not args.no_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )

    config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        bf16=use_bf16,
        fp16=bool(torch.cuda.is_available() and not use_bf16),
        gradient_checkpointing=True,
        max_length=None,
        report_to="none",
        remove_unused_columns=False,
    )
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
