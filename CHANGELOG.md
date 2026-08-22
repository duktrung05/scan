# Changelog

## 0.3.0 — Multilingual edition

- Unifies English (`en`), Vietnamese (`vi`), Korean (`ko`) and Japanese (`ja`).
- Adds explicit OCR preservation rules for every supported language.
- Uses CER as the primary text error rate for Korean and Japanese; WER remains primary
  for English and Vietnamese. Both CER and WER are always reported.
- Keeps the Japanese schema adapter and the evaluation-only benchmark guard from 0.2.0.
- Renames the deliverable to `multilingual` so it is not mistaken for a Japanese-only app.

## 0.2.0

- Added Japanese prompts, schema conversion, dataset preparation and safety tests.

## 0.1.0

- Initial English, Vietnamese and Korean document-understanding pipeline.
