# TriScan fine-tuning

Training is intentionally separated from inference. First create a leakage-safe
manifest whose split is assigned by source/template family, not by individual page.

Prepare images and conversational VLM examples:

```bash
uv run python training/prepare_sft_dataset.py data/examples/manifest.example.jsonl data/prepared
```

Run QLoRA:

```bash
uv sync --extra local --extra train
uv run python training/train_vlm_lora.py data/prepared/dataset.jsonl \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --output-dir checkpoints/triscan-qwen3-vl-lora
```

Before a full run, verify that 10–30 samples can overfit, inspect assistant-token
loss masking, and record the exact model revision, package lockfile, random seed,
GPU type, VRAM peak, prompt version and dataset hash.

