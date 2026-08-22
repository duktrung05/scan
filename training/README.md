# TScan fine-tuning

Training is intentionally separated from inference. First create a leakage-safe
manifest whose split is assigned by source/template family, not by individual page.

Prepare images and conversational VLM examples:

```bash
uv run python training/prepare_sft_dataset.py data/examples/manifest.example.jsonl data/prepared
```

Japanese bundle validation and preparation:

```bash
uv run python training/prepare_japanese_sft_dataset.py data/prepared-ja --validate-only
uv run python training/prepare_japanese_sft_dataset.py data/prepared-ja
```

The Japanese command verifies source hashes, image availability, answer JSON, language and
license gates. `EVAL_ONLY` and test splits are also rejected by the training loader.

Run QLoRA:

```bash
uv sync --extra local --extra train
uv run python training/train_vlm_lora.py data/prepared/dataset.jsonl \
  --config configs/train_lora.yaml \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --model-revision <immutable-hugging-face-commit> \
  --output-dir checkpoints/tscan-qwen3-vl-lora
```

The trainer verifies images without retaining PIL objects, then uses lazy dataset image
decoding. It fixes Python/NumPy/PyTorch/DataLoader seeds, records dataset SHA-256, resolved
model revision, prompt/schema versions and GPU/CUDA runtime, and supports warmup, gradient
clipping, checkpoint retention, resume and optional early stopping. When validation data is
present, checkpoint selection uses the configured JSON composite score rather than training
loss alone. The final held-out benchmark remains the release gate because the in-training
score is computed under teacher forcing.

Before a full run, verify that 10–30 samples can overfit, inspect assistant-token
loss masking, and record the exact model revision, package lockfile, random seed,
GPU type, VRAM peak, prompt version and dataset hash.
