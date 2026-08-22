# Japanese data bundle

This directory registers the four Japanese JSONL files supplied for TScan v0.2.0.

## Current state

- 618 valid JSONL rows; every `answer` parses as JSON.
- All IDs and image paths are unique across the four files.
- `japan_ocr_mini_benchmark` is marked `EVAL_ONLY` and `external_test`.
- The JSONL files reference 618 paths under `images/`, but those image files are not bundled.
- Dataset licenses were not supplied, so every source remains `VERIFY` and is blocked from SFT.
- Source `schema` values use a shorthand DSL. The Japanese preparation script converts that DSL
  to JSON Schema Draft 2020-12 before building prompts.

The raw JSONL files are preserved byte-for-byte in `raw/`. Do not edit them. The intake
summary is stored in `validation_report.json`. Update only `registry.json` after image
availability and license evidence have been verified.

## Expected image layout

Copy the matching image directories so paths resolve exactly as recorded in the JSONL files:

```text
data/japanese/
├── images/
│   ├── yana_ft_llm_2026_ocr_dataset_images/
│   ├── japanese_docqa_it_images/
│   ├── japan_ocr_mini_benchmark/
│   └── jdocqa_dataset/
├── raw/
└── registry.json
```

## Validate and prepare

```bash
uv run python training/prepare_japanese_sft_dataset.py data/prepared-ja --validate-only
uv run python training/prepare_japanese_sft_dataset.py data/prepared-ja
```

The command intentionally fails while images are missing or licenses are not marked `OK`.
Do not change a license status merely to bypass the gate; record the actual permission first.
