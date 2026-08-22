from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

from tscan.evaluation.metrics import json_metrics
from tscan.prompts import PROMPT_VERSION
from tscan.training_safety import classify_training_row
from tscan.validation.json_parser import parse_json_output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _setting(cli_value: Any, config: dict[str, Any], name: str, default: Any) -> Any:
    return cli_value if cli_value is not None else config.get(name, default)


def _composite_metrics(processor: Any):
    def compute(eval_prediction: Any) -> dict[str, float]:
        predictions, labels = eval_prediction
        rows = []
        pad_id = processor.tokenizer.pad_token_id or 0
        for prediction, label in zip(predictions, labels, strict=False):
            label_values = label.tolist()
            prediction_values = prediction.tolist()
            mask = [value != -100 for value in label_values[1:]]
            reference_ids = [value if value != -100 else pad_id for value in label_values[1:]]
            predicted_ids = prediction_values[:-1]
            reference_text = processor.tokenizer.decode(
                [value for value, keep in zip(reference_ids, mask, strict=False) if keep],
                skip_special_tokens=True,
            )
            prediction_text = processor.tokenizer.decode(
                [value for value, keep in zip(predicted_ids, mask, strict=False) if keep],
                skip_special_tokens=True,
            )
            reference_json = parse_json_output(reference_text).value
            prediction_json = parse_json_output(prediction_text).value
            if reference_json is None or prediction_json is None:
                rows.append(
                    {
                        "field_value_f1": 0.0,
                        "amount_tolerance_accuracy": 0.0,
                        "date_normalized_accuracy": 0.0,
                        "schema_valid_rate": 0.0,
                        "hallucinated_field_rate": 1.0,
                    }
                )
            else:
                rows.append(json_metrics(reference_json, prediction_json))
        averaged = {
            key: fmean(row[key] for row in rows)
            for key in (
                "field_value_f1",
                "amount_tolerance_accuracy",
                "date_normalized_accuracy",
                "schema_valid_rate",
                "hallucinated_field_rate",
            )
        }
        averaged["validation_score"] = (
            averaged["field_value_f1"]
            + averaged["amount_tolerance_accuracy"]
            + averaged["date_normalized_accuracy"]
            + averaged["schema_valid_rate"]
            - averaged["hallucinated_field_rate"]
        )
        return averaged

    return compute


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a VLM with reproducible LoRA/QLoRA")
    parser.add_argument("dataset", type=Path, help="Prepared dataset.jsonl")
    parser.add_argument("--config", type=Path, default=Path("configs/train_lora.yaml"))
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--resume-from-checkpoint", nargs="?", const=True)
    parser.add_argument("--no-qlora", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
    model_name = _setting(args.model, config, "model", "Qwen/Qwen3-VL-2B-Instruct")
    model_revision = _setting(args.model_revision, config, "model_revision", "main")
    output_dir = Path(_setting(args.output_dir, config, "output_dir", "artifacts/checkpoints/tscan-lora"))
    epochs = float(_setting(args.epochs, config, "epochs", 2.0))
    learning_rate = float(_setting(args.learning_rate, config, "learning_rate", 1e-4))
    batch_size = int(_setting(args.batch_size, config, "batch_size", 1))
    gradient_accumulation = int(_setting(args.gradient_accumulation, config, "gradient_accumulation", 8))
    lora_rank = int(_setting(args.lora_rank, config, "lora_rank", 16))
    seed = int(_setting(args.seed, config, "seed", 42))
    warmup_ratio = float(_setting(args.warmup_ratio, config, "warmup_ratio", 0.05))
    max_grad_norm = float(_setting(args.max_grad_norm, config, "max_grad_norm", 1.0))
    eval_steps = int(_setting(args.eval_steps, config, "eval_steps", 100))
    save_steps = int(_setting(args.save_steps, config, "save_steps", 100))
    save_total_limit = int(_setting(args.save_total_limit, config, "save_total_limit", 3))
    early_stopping_patience = int(_setting(args.early_stopping_patience, config, "early_stopping_patience", 0))
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        import torch
        from datasets import Dataset, Sequence
        from datasets import Image as DatasetImage
        from peft import LoraConfig
        from PIL import Image
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
            EarlyStoppingCallback,
            set_seed,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: uv sync --extra local --extra train") from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    dataset_path = args.dataset.resolve()
    base = dataset_path.parent
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        partition = classify_training_row(row)
        if partition == "skip":
            skipped.append({"line": line_number, "document_id": row.get("document_id"), "reason": "evaluation_only"})
            continue
        image_paths = [(base / relative_path).resolve() for relative_path in row["images"]]
        corrupt = None
        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                corrupt = f"{image_path}: {exc}"
                break
        if corrupt:
            skipped.append({"line": line_number, "document_id": row.get("document_id"), "reason": "corrupt_image", "detail": corrupt})
            continue
        prompt_content = [{"type": "image"} for _ in image_paths]
        prompt_content.append({"type": "text", "text": row["prompt"]})
        sample = {
            "images": [str(path) for path in image_paths],
            "prompt": [{"role": "user", "content": prompt_content}],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": row["completion"]}]}],
        }
        (eval_rows if partition == "validation" else train_rows).append(sample)

    (output_dir / "skipped_records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in skipped), encoding="utf-8"
    )
    if not train_rows:
        raise SystemExit("No training samples found")

    def lazy_dataset(rows: list[dict[str, Any]]) -> Any:
        dataset = Dataset.from_list(rows)
        return dataset.cast_column("images", Sequence(DatasetImage(decode=True)))

    train_dataset = lazy_dataset(train_rows)
    eval_dataset = lazy_dataset(eval_rows) if eval_rows else None
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    quantization_config = None
    if not args.no_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
    processor = AutoProcessor.from_pretrained(model_name, revision=model_revision)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        revision=model_revision,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    resolved_revision = getattr(model.config, "_commit_hash", None) or model_revision

    training_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        logging_steps=5,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="validation_score" if eval_dataset is not None else None,
        greater_is_better=True,
        save_total_limit=save_total_limit,
        bf16=use_bf16,
        fp16=bool(torch.cuda.is_available() and not use_bf16),
        gradient_checkpointing=True,
        max_length=None,
        report_to="none",
        remove_unused_columns=False,
        completion_only_loss=True,
        seed=seed,
        data_seed=seed,
    )
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    def preprocess_logits(logits: Any, _labels: Any) -> Any:
        return (logits[0] if isinstance(logits, tuple) else logits).argmax(dim=-1)

    callbacks = []
    if eval_dataset is not None and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        compute_metrics=_composite_metrics(processor) if eval_dataset is not None else None,
        preprocess_logits_for_metrics=preprocess_logits if eval_dataset is not None else None,
        callbacks=callbacks,
    )
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "train_samples": len(train_rows),
        "validation_samples": len(eval_rows),
        "skipped_samples": len(skipped),
        "model": model_name,
        "requested_model_revision": model_revision,
        "resolved_model_revision": resolved_revision,
        "prompt_version": config.get("prompt_version", PROMPT_VERSION),
        "schema_version": str(config.get("schema_version", "unknown")),
        "seed": seed,
        "config": training_config.to_dict(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    metadata["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    metadata["completed_at"] = datetime.now(UTC).isoformat()
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
