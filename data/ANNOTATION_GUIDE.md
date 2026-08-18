# Annotation contract

This file is the minimum ground-truth contract. Extend it before labeling at scale.

## General rules

- Transcribe visible content; never infer missing values.
- Use `null` for absent or unreadable fields and record the reason separately in QA notes.
- Keep identifiers exactly as printed, including leading zeros.
- Store numeric amounts as JSON numbers without currency symbols or thousands separators.
- Store currency separately using the visible ISO code when present; do not guess it from locale.
- Use ISO `YYYY-MM-DD` only when the full date is unambiguous. Otherwise retain the visible string in a schema field designed for raw dates.
- Preserve one document as one sample; multi-page pages must remain together.

## Markdown ground truth

- Use `<!-- page: N -->` before each page.
- Preserve reading order, headings, lists and tables.
- Do not reproduce decorative borders or scanner noise.
- Keep meaningful line breaks; normalize repeated spaces.

## Split safety

- Assign splits by vendor, template, source family or document lineage.
- Near duplicates, crops and synthetic variants of one template belong to one split.
- Freeze the test split before the first fine-tuning run.

## Quality control

- Pilot the guide on at least two reviewers before scaling.
- Double-review critical fields: identifiers, dates, currency, subtotal, tax and total.
- Track disagreements and update this contract with adjudicated examples.

