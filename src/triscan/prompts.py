""" Multilingual Semantic Compact Image-to-JSON Prompt.
V9 follows a semantic-schema approach:

IMAGE
  -> understand semantic roles
  -> generate compact semantic extraction TEMPLATE
  -> extract exact source-grounded values into that TEMPLATE

Core design
-----------
1. `document_type` is NOT output.
2. `language` is NOT output.
3. JSON-Schema metadata is NOT output.
4. The main `title` field is the only mandatory root structural field.
5. Other keys are compact SEMANTIC KEYS chosen from document meaning.
6. Semantic keys must be stable/general, not literal document-specific content.
7. Document / section / table / chart / figure titles are VALUES, never keys.
8. Real tables/charts/figures use small reserved containers when needed.
9. Source values remain source-faithful.
10. Structural/semantic key language follows the document's primary language.

Supported primary key languages
-------------------------------
- English
- Korean
- Japanese
- Vietnamese

This is a two-stage design:
Stage 1: generate a compact semantic extraction template.
Stage 2: extract values using exactly that template.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Literal


LanguageName = Literal["English", "Korean", "Japanese", "Vietnamese"]


# =============================================================================
# 1. RESERVED STRUCTURAL KEY MAPS
# =============================================================================

# Only a small number of structural concepts are reserved.
# Other semantic keys are generated dynamically according to document meaning.
RESERVED_KEY_MAPS: dict[str, dict[str, str]] = {
    "English": {
        "title": "title",
        "tables": "tables",
        "rows": "rows",
        "charts": "charts",
        "data": "data",
        "figures": "figures",
        "caption": "caption",
        "text": "text",
    },
    "Korean": {
        "title": "제목",
        "tables": "표",
        "rows": "행",
        "charts": "차트",
        "data": "데이터",
        "figures": "그림",
        "caption": "캡션",
        "text": "텍스트",
    },
    "Japanese": {
        "title": "タイトル",
        "tables": "表",
        "rows": "行",
        "charts": "グラフ",
        "data": "データ",
        "figures": "図",
        "caption": "キャプション",
        "text": "テキスト",
    },
    "Vietnamese": {
        "title": "tiêu_đề",
        "tables": "bảng",
        "rows": "hàng",
        "charts": "biểu_đồ",
        "data": "dữ_liệu",
        "figures": "hình",
        "caption": "chú_thích",
        "text": "văn_bản",
    },
}


# =============================================================================
# 2. TYPE MARKERS
# =============================================================================

TYPE_MARKERS = {
    "integer",
    "number",
    "boolean",
    "string",
    "verbatim-string",
    "date",
    "time",
    "date-time",
    "duration",
    "currency",
    "country",
    "language",
    "language-tag",
    "script",
    "url",
    "email-address",
    "phone-number",
    "iban",
    "bic",
    "unit-code",
}


def _is_type_marker(value: str) -> bool:
    return value in TYPE_MARKERS or value.startswith("region:")


# =============================================================================
# 3. STAGE 1 — SEMANTIC COMPACT TEMPLATE GENERATION
# =============================================================================

SCHEMA_GEN_PROMPT = r"""
# TASK

Generate ONE compact semantic extraction TEMPLATE for the attached DOCUMENT image.

This is NOT JSON Schema Draft-07.

NEVER output:
- "$schema"
- "$id"
- "$ref"
- "type" as a JSON-Schema object descriptor
- "properties"
- "required"
- "additionalProperties"
- schema metadata names/titles




# ============================================================================
# 1. ROOT RULE — TITLE ONLY
# ============================================================================

The FIRST root key is always the localized semantic equivalent of `title`.

English:
"title": "verbatim-string"

Korean:
"제목": "verbatim-string"

Japanese:
"タイトル": "verbatim-string"

Vietnamese:
"tiêu_đề": "verbatim-string"

This title field means:
"extract the visible MAIN DOCUMENT TITLE."

Do NOT put the actual document title into the TEMPLATE.

Do NOT output:
- document_type
- language


# ============================================================================
# 2. KEY LANGUAGE
# ============================================================================

If RUNTIME KEY LANGUAGE is supplied:
use it exactly.

Otherwise detect the document's primary language and use that language for:
- reserved structural keys
- generated semantic keys

Supported:
English / Korean / Japanese / Vietnamese

Do not mix key languages within one template.


# ============================================================================
# 3. SEMANTIC KEYS ARE ALLOWED
# ============================================================================

After the mandatory title field, generate only the semantic fields/objects/arrays
actually useful for this document.

GOOD semantic keys:
English:
- company_information
- company_name
- registration_number
- employee_information
- payment_summary
- item_count
- total_amount
- processing_period

Korean:
- 회사_정보
- 회사명
- 등록번호
- 직원_정보
- 결제_요약
- 품목_수
- 총액

Japanese:
- 会社情報
- 会社名
- 登録番号
- 従業員情報
- 支払概要
- 商品数
- 合計金額

Vietnamese:
- thông_tin_công_ty
- tên_công_ty
- mã_đăng_ký
- thông_tin_nhân_viên
- tóm_tắt_thanh_toán
- số_mặt_hàng
- tổng_tiền

The key describes WHAT THE VALUE MEANS.


# ============================================================================
# 4. GENERAL-KEY STABILITY TEST
# ============================================================================

Before accepting every generated semantic key, silently ask:

"If another document of the same semantic type had different actual content,
would this key still be appropriate?"

YES:
keep the semantic key.

NO:
the candidate is probably literal document content.
Do NOT use it as a key.


# ============================================================================
# 5. LITERAL CONTENT MUST NEVER BECOME A KEY
# ============================================================================

The following are VALUES, not JSON keys:

- actual document title
- article/report title
- section/event title
- table title
- chart/graph title
- figure/diagram title
- person name
- company/organization name
- product name
- model name
- actual date/time
- actual ID/reference number
- literal sentence/paragraph
- arbitrary caption

WRONG:
{
  "TAX INVOICE": {...}
}

CORRECT TEMPLATE:
{
  "title": "verbatim-string",
  ...
}

WRONG:
{
  "GST Summary": [...]
}

CORRECT TEMPLATE:
{
  "tables": [
    {
      "title": "verbatim-string",
      "rows": [...]
    }
  ]
}

WRONG:
{
  "Annual Revenue Growth 2025": {...}
}

CORRECT TEMPLATE:
{
  "charts": [
    {
      "title": "verbatim-string",
      "data": {...}
    }
  ]
}


# ============================================================================
# 6. DO NOT COPY SOURCE LABELS BLINDLY
# ============================================================================

A visible source label may inspire a semantic key, but do not blindly OCR it into
the key.

Source:
"GST ID No"

Preferred semantic key:
"gst_id"

Source:
"Total Incl. GST@6%"

Preferred semantic key:
"total_including_gst"

Source:
"Employee No"

Preferred semantic key:
"employee_id"

This makes the schema compact and semantically stable.


# ============================================================================
# 7. SEMANTIC OBJECTS
# ============================================================================

Group closely related semantic fields into compact objects when the grouping is real.

Example:

{
  "company_information": {
    "company_name": "verbatim-string",
    "registration_number": "verbatim-string",
    "address": "verbatim-string",
    "gst_id": "verbatim-string"
  }
}

Do NOT create an object merely because a heading is visually above content.

Use semantic grouping, not visual-box grouping.


# ============================================================================
# 8. REPEATED SEMANTIC RECORDS
# ============================================================================

Use [{...}] for repeated semantic records.

Example receipt items:

{
  "items": [
    {
      "description": "verbatim-string",
      "quantity": "verbatim-string",
      "unit_price": "verbatim-string",
      "total": "verbatim-string"
    }
  ]
}

This is often more compact than forcing a generic table wrapper.

Use semantic repeated objects when the row meaning is obvious and stable.


# ============================================================================
# 9. RESERVED TABLE CONTAINER
# ============================================================================

Use the localized reserved `tables` container when:
- the document contains a distinct named table,
- the table is auxiliary/summary data,
- or preserving generic column structure is more appropriate than inventing a
  domain-specific semantic array key.

Semantic shape:

{
  "tables": [
    {
      "title": "verbatim-string",
      "rows": [
        {
          "<dynamic_column_label>": "verbatim-string"
        }
      ]
    }
  ]
}

Rules:
- table title is a VALUE
- table title is never a key
- column headers may be dynamic inside row objects because they are table schema,
  not document wrapper keys
- do not invent a column header

For a headerless table, rows may be arrays of repeated scalar values.


# ============================================================================
# 10. RESERVED CHART CONTAINER
# ============================================================================

Use localized `charts` when a chart/graph is visible.

Shape:

{
  "charts": [
    {
      "title": "verbatim-string",
      "data": {
        "<dynamic_chart_label>": "verbatim-string"
      }
    }
  ]
}

Rules:
- chart title is a VALUE
- chart title is never a key
- only include readable/grounded chart data
- do not infer missing values from trends


# ============================================================================
# 11. RESERVED FIGURE CONTAINER
# ============================================================================

Use localized `figures` when a figure/diagram is relevant.

Shape:

{
  "figures": [
    {
      "title": "verbatim-string",
      "caption": "verbatim-string",
      "text": ["verbatim-string"]
    }
  ]
}

Figure title/caption are VALUES only.


# ============================================================================
# 12. FREE PROSE
# ============================================================================

When relevant prose does not belong naturally to a semantic field/object, use the
localized reserved `text` key.

Example:
"text": ["verbatim-string"]

Do not create one wrapper object per paragraph.


# ============================================================================
# 13. TYPE CHOICE
# ============================================================================

Prefer source-faithful extraction.

Default preference:
"verbatim-string"

Use typed semantic markers only when normalization is clearly desired and safe.

Supported scalar markers:
"integer"
"number"
"boolean"
"string"
"verbatim-string"
"date"
"time"
"date-time"
"duration"
"currency"
"country"
"language"
"language-tag"
"script"
"url"
"email-address"
"phone-number"
"iban"
"bic"
"unit-code"
"region:XX"

For OCR/KIE datasets, prefer "verbatim-string" for:
- visible money strings
- date strings whose formatting matters
- IDs
- codes
- names
- source labels/text
- values containing symbols


# ============================================================================
# 14. ENUMS
# ============================================================================

Single enum:
["A", "B"]

Multi enum:
[["A", "B"]]

Use enums only when a genuine closed set is defined by:
- the document
- dataset/task policy

Never invent enum classes from general knowledge.


# ============================================================================
# 15. COMPACTNESS
# ============================================================================

The template should contain only meaningful semantic fields.

Do NOT add generic boilerplate such as:
- document_type
- language
- sections
- page_index
- block_id
- block_type
- order
- fields
- metadata wrappers

unless a specific semantic field with that meaning is genuinely needed.

Prefer:

{
  "title": "verbatim-string",
  "company_information": {...},
  "items": [{...}],
  "payment_summary": {...},
  "tables": [...]
}

over:

{
  "title": "verbatim-string",
  "sections": [
    {
      "fields": {...},
      "text": [...]
    }
  ]
}


# ============================================================================
# 16. FINAL TEMPLATE CHECK
# ============================================================================

Before returning silently verify:

1. First root key is localized title.
2. No document_type.
3. No language.
4. No JSON-Schema metadata.
5. Every non-reserved key is semantic/general.
6. No actual title/name/value became a key.
7. Document/table/chart/figure titles are values.
8. Semantic repeated records use compact [{...}] where appropriate.
9. Reserved tables/charts/figures are used only when useful.
10. Template is compact.
11. Output is exactly one JSON template.
12. No prose / Markdown / code fences.
"""


# =============================================================================
# 4. STAGE 2 — EXTRACTION
# =============================================================================

EXTRACTION_INSTRUCTIONS = r"""
# TASK

Extract information from the attached DOCUMENT image using the INPUT TEMPLATE.

Return exactly ONE compact JSON object matching the INPUT TEMPLATE.


# ============================================================================
# 1. NO EXTRA METADATA
# ============================================================================

Do NOT output:
- document_type
- language
- "$schema"
- JSON-Schema metadata
- extra helper metadata


# ============================================================================
# 2. TEMPLATE IS AUTHORITATIVE
# ============================================================================

The INPUT TEMPLATE defines:
- exact semantic keys
- exact nesting
- repeated objects/lists
- reserved tables/charts/figures
- value types

Do not:
- rename semantic keys
- translate semantic keys
- add extra semantic keys
- introduce wrappers such as sections/blocks
- replace semantic keys with raw OCR labels


# ============================================================================
# 3. TITLE
# ============================================================================

The first root title key stores the actual MAIN document title.

If a clear main title is visible:
- copy it source-faithfully

If no clear main title exists:
- null

Never use a table/chart/section title as the root title unless it truly functions
as the main document title.


# ============================================================================
# 4. SEMANTIC VALUES
# ============================================================================

Populate each semantic key with the source-grounded value that matches its role.

Examples:

company_name:
-> actual company name value

registration_number:
-> actual registration number

total_amount:
-> actual visible total

Do not copy raw source labels into values unless the template explicitly asks for them.


# ============================================================================
# 5. REPEATED OBJECTS
# ============================================================================

For semantic arrays such as items / employees / transactions:

- preserve every visible record
- preserve order
- do not deduplicate
- do not merge
- do not aggregate
- unreadable field -> null
- keep the record


# ============================================================================
# 6. TABLES
# ============================================================================

For reserved tables:

- table title is copied as a VALUE
- row column labels may be dynamic keys if the template uses a dynamic-column
  placeholder
- preserve exact visible header wording
- preserve every row
- preserve duplicate rows
- unreadable cell -> null
- do not invent headers


# ============================================================================
# 7. CHARTS / FIGURES
# ============================================================================

Chart title:
- VALUE only

Figure title/caption:
- VALUE only

Do not infer unreadable chart values from trends.
Do not create a chart/figure name as a wrapper key.


# ============================================================================
# 8. SOURCE FIDELITY
# ============================================================================

For verbatim-string:
- preserve source content
- no translation
- no paraphrase
- no spelling correction
- collapse only line-break/tab/repeated-whitespace artifacts when necessary

Preserve:
- Unicode
- punctuation
- decimal points
- currency signs
- percentages
- arrows
- comparison signs
- IDs
- date formatting


# ============================================================================
# 9. NUMERIC FIDELITY
# ============================================================================

1491 -> "1491"
NOT "14.91"

14.9 -> "14.9"
NOT "149"

Never repair a visible number using:
- totals
- arithmetic
- expected ranges
- domain knowledge
- neighboring values


# ============================================================================
# 10. SYMBOL FIDELITY
# ============================================================================

Preserve:
↑ ↓ → ← ↗ ↘ ↔
+ - ±
% ‰
< > ≤ ≥ = ≈
¥ $ ₩ € etc.

Never flip signs/arrows from context.


# ============================================================================
# 11. MISSING VALUES
# ============================================================================

Missing scalar:
-> null

Missing repeated scalar:
-> []

Missing repeated object collection:
-> []

For a non-repeated semantic object:
- preserve the object shape
- recursively fill missing descendants with null / []


# ============================================================================
# 12. COMPACTNESS
# ============================================================================

Do not add:
- sections
- blocks
- ids
- order metadata
- language
- document_type
- schema metadata

unless the INPUT TEMPLATE explicitly contains a semantic key with that exact role.


# ============================================================================
# 13. FINAL EXTRACTION CHECK
# ============================================================================

1. Exactly one valid JSON object.
2. Exact template keys/nesting.
3. No document_type.
4. No language.
5. No JSON-Schema metadata.
6. Root title is actual title/null.
7. No extra keys.
8. No title/table/chart/figure name became a key.
9. Repeated records preserved.
10. Source values remain faithful.
11. No numeric/symbol repair.
12. No prose / Markdown / code fences.
"""


# =============================================================================
# 5. PROMPT BUILDERS
# =============================================================================

def build_schema_prompt(
    *,
    key_language: LanguageName | None = None,
    user_instruction: str | None = None,
) -> str:
    parts = [SCHEMA_GEN_PROMPT]

    if key_language is not None:
        if key_language not in RESERVED_KEY_MAPS:
            raise ValueError(f"Unsupported key_language: {key_language}")

        parts.append(f"# RUNTIME KEY LANGUAGE\n{key_language}")
        parts.append(
            "# RESERVED STRUCTURAL KEY MAP\n"
            + json.dumps(
                RESERVED_KEY_MAPS[key_language],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        parts.append(
            "# AVAILABLE RESERVED KEY MAPS\n"
            + json.dumps(
                RESERVED_KEY_MAPS,
                ensure_ascii=False,
                indent=2,
            )
        )

    if user_instruction and user_instruction.strip():
        parts.append(
            "# OPTIONAL USER INSTRUCTION\n"
            "May refine extraction scope, but must not add document_type/language, "
            "JSON-Schema metadata, or permit literal document-specific values to "
            "become semantic keys.\n"
            + user_instruction.strip()
        )

    parts.append(
        "# CONTEXT\n"
        "The document is the attached image. "
        "Read it directly and return one compact semantic extraction TEMPLATE."
    )

    return "\n\n".join(parts)


def build_extraction_prompt(
    template: Any,
    user_instruction: str | None = None,
) -> str:
    template_str = (
        template
        if isinstance(template, str)
        else json.dumps(template, ensure_ascii=False, indent=2)
    )

    parts = [
        EXTRACTION_INSTRUCTIONS,
        "# INPUT TEMPLATE\n" + template_str,
    ]

    if user_instruction and user_instruction.strip():
        parts.append(
            "# OPTIONAL USER INSTRUCTION\n"
            "May refine scope but cannot change template keys/shape or add "
            "document_type/language/schema metadata.\n"
            + user_instruction.strip()
        )

    parts.append(
        "# CONTEXT\n"
        "The document is the attached image. "
        "Read it directly and return one compact extraction JSON object."
    )

    return "\n\n".join(parts)


# =============================================================================
# 6. JSON SYNTAX REPAIR
# =============================================================================

JSON_REPAIR_PROMPT = r"""
Repair ONLY JSON syntax.

Do NOT:
- add document_type
- add language
- add JSON-Schema metadata
- rename keys
- translate keys
- translate values
- change numbers
- change decimal points
- change signs/arrows
- reorder repeated rows/items
- create new semantic content

Return ONLY one valid JSON object.
"""


def build_json_repair_prompt(model_output: str) -> str:
    return (
        JSON_REPAIR_PROMPT
        + "\n\n# MODEL OUTPUT TO REPAIR\n"
        + model_output
    )


# =============================================================================
# 7. CLEAN EMPTY VALUES
# =============================================================================

def clean_empty_values(
    data: Any,
    *,
    keep_root_title: bool = True,
) -> Any:
    """Remove None, [], {} recursively.

    Keeps root title even when null when `keep_root_title=True`.
    Preserves 0, False and "".
    """
    title_keys = {
        mapping["title"]
        for mapping in RESERVED_KEY_MAPS.values()
    }

    def clean(node: Any, *, is_root: bool = False) -> Any:
        if isinstance(node, dict):
            result: dict[str, Any] = {}

            for key, value in node.items():
                cleaned = clean(value)

                if (
                    is_root
                    and keep_root_title
                    and key in title_keys
                ):
                    result[key] = cleaned
                    continue

                if cleaned not in (None, [], {}):
                    result[key] = cleaned

            return result

        if isinstance(node, list):
            result = []
            for item in node:
                cleaned = clean(item)
                if cleaned not in (None, [], {}):
                    result.append(cleaned)
            return result

        return node

    return clean(data, is_root=True)


# =============================================================================
# 8. TEMPLATE / ANSWER VALIDATION
# =============================================================================

FORBIDDEN_ROOT_KEYS = {
    "document_type",
    "language",
    "$schema",
    "$id",
    "$ref",
    "properties",
    "required",
    "additionalProperties",
    "문서_유형",
    "언어",
    "文書タイプ",
    "言語",
    "loại_tài_liệu",
    "ngôn_ngữ",
}


def _is_dynamic_placeholder(key: str) -> bool:
    return key.startswith("<dynamic_") and key.endswith(">")


def validate_template_answer_shape(
    template: Any,
    answer: Any,
) -> list[str]:
    """Validate Stage-2 answer against V9 template semantics.

    Supports dynamic placeholder dictionaries such as:
        {"<dynamic_column_label>": "verbatim-string"}
    which allow arbitrary output keys whose values match the placeholder type.
    """
    errors: list[str] = []

    if isinstance(answer, dict):
        for forbidden in FORBIDDEN_ROOT_KEYS & set(answer):
            errors.append(
                f"$.{forbidden}: forbidden metadata/output field"
            )

    def walk(t: Any, a: Any, path: str) -> None:
        # Scalar type marker.
        if isinstance(t, str):
            if not _is_type_marker(t):
                # Literal string is only expected inside enum declarations.
                return

            if a is None:
                return

            if t == "integer":
                if not isinstance(a, int) or isinstance(a, bool):
                    errors.append(
                        f"{path}: expected integer/null, got {type(a).__name__}"
                    )
                return

            if t == "number":
                if not isinstance(a, (int, float)) or isinstance(a, bool):
                    errors.append(
                        f"{path}: expected number/null, got {type(a).__name__}"
                    )
                return

            if t == "boolean":
                if not isinstance(a, bool):
                    errors.append(
                        f"{path}: expected boolean/null, got {type(a).__name__}"
                    )
                return

            # Standardized/string-like markers.
            if not isinstance(a, str):
                errors.append(
                    f"{path}: expected string-like/null, got {type(a).__name__}"
                )
            return

        # Object.
        if isinstance(t, dict):
            if not isinstance(a, dict):
                errors.append(
                    f"{path}: expected object, got {type(a).__name__}"
                )
                return

            dynamic_keys = [key for key in t if _is_dynamic_placeholder(key)]

            if dynamic_keys:
                # A dynamic-placeholder object should contain exactly one placeholder key.
                if len(t) != 1 or len(dynamic_keys) != 1:
                    errors.append(
                        f"{path}: ambiguous dynamic-placeholder template"
                    )
                    return

                placeholder = dynamic_keys[0]
                value_template = t[placeholder]

                for actual_key, actual_value in a.items():
                    walk(
                        value_template,
                        actual_value,
                        f"{path}[{actual_key!r}]",
                    )
                return

            expected = set(t)
            actual = set(a)

            for key in expected - actual:
                errors.append(f"{path}.{key}: missing_in_answer")

            for key in actual - expected:
                errors.append(f"{path}.{key}: extra_in_answer")

            for key in expected & actual:
                walk(t[key], a[key], f"{path}.{key}")

            return

        # List / enum / repeated record.
        if isinstance(t, list):
            if not t:
                errors.append(f"{path}: empty template list is ambiguous")
                return

            # Repeated object.
            if len(t) == 1 and isinstance(t[0], dict):
                if not isinstance(a, list):
                    errors.append(
                        f"{path}: expected array, got {type(a).__name__}"
                    )
                    return

                for i, item in enumerate(a):
                    walk(t[0], item, f"{path}[{i}]")
                return

            # Repeated scalar.
            if (
                len(t) == 1
                and isinstance(t[0], str)
                and _is_type_marker(t[0])
            ):
                if not isinstance(a, list):
                    errors.append(
                        f"{path}: expected array, got {type(a).__name__}"
                    )
                    return

                for i, item in enumerate(a):
                    walk(t[0], item, f"{path}[{i}]")
                return

            # Multi enum.
            if (
                len(t) == 1
                and isinstance(t[0], list)
                and len(t[0]) >= 2
                and all(isinstance(x, str) for x in t[0])
            ):
                if not isinstance(a, list):
                    errors.append(
                        f"{path}: expected array, got {type(a).__name__}"
                    )
                    return

                allowed = set(t[0])
                for i, item in enumerate(a):
                    if item not in allowed:
                        errors.append(
                            f"{path}[{i}]: value {item!r} not in enum"
                        )
                return

            # Single enum.
            if len(t) >= 2 and all(isinstance(x, str) for x in t):
                if a is None:
                    return

                if isinstance(a, list):
                    errors.append(
                        f"{path}: expected scalar enum/null, got array"
                    )
                    return

                if a not in t:
                    errors.append(
                        f"{path}: value {a!r} not in enum"
                    )
                return

            errors.append(
                f"{path}: unsupported/ambiguous template list"
            )
            return

        errors.append(
            f"{path}: unsupported template node {type(t).__name__}"
        )

    walk(template, answer, "$")
    return errors


# =============================================================================
# 9. EXAMPLE V9 TEMPLATE / OUTPUT
# =============================================================================

def example_english_receipt_template() -> dict[str, Any]:
    """Example of the compact semantic style V9 is targeting."""
    return {
        "title": "verbatim-string",
        "company_information": {
            "company_name": "verbatim-string",
            "registration_number": "verbatim-string",
            "address": "verbatim-string",
            "gst_id": "verbatim-string",
        },
        "items": [
            {
                "description": "verbatim-string",
                "quantity": "verbatim-string",
                "unit_price": "verbatim-string",
                "total": "verbatim-string",
            }
        ],
        "payment_summary": {
            "item_count": "verbatim-string",
            "quantity_count": "verbatim-string",
            "total_including_gst": "verbatim-string",
            "cash": "verbatim-string",
            "change": "verbatim-string",
        },
        "tables": [
            {
                "title": "verbatim-string",
                "rows": [
                    {
                        "<dynamic_column_label>": "verbatim-string"
                    }
                ],
            }
        ],
        "transaction_information": {
            "date_time": "verbatim-string",
            "operator": "verbatim-string",
        },
        "text": ["verbatim-string"],
    }


def example_english_receipt_output() -> dict[str, Any]:
    return {
        "title": "TAX INVOICE",
        "company_information": {
            "company_name": "MR. D.I.Y. (KUCHAI) SDN BHD",
            "registration_number": "750441-W",
            "address": "LOT 1851-A & 1851-B, JALAN KPB 6, KAWASAN PERINDUSTRIAN BALAKONG, 43300 SERI KEMBANGAN, SELANGOR",
            "gst_id": "000473792512",
        },
        "items": [
            {
                "description": "MAGNETIC WHITEBOARD LD001# 90*60CM *S",
                "quantity": "1",
                "unit_price": "35.00",
                "total": "35.00",
            }
        ],
        "payment_summary": {
            "item_count": "7",
            "quantity_count": "8",
            "total_including_gst": "RM 119.70",
            "cash": "RM 120.00",
            "change": "RM 0.30",
        },
        "tables": [
            {
                "title": "GST Summary",
                "rows": [
                    {
                        "GST Summary": "GST S@6%",
                        "Amt(RM)": "112.92",
                        "Tax(RM)": "6.78",
                    }
                ],
            }
        ],
        "transaction_information": {
            "date_time": "15-03-18 15:41",
            "operator": "TRAINEE CASHIER",
        },
        "text": [
            "EXCHANGE ARE ALLOWED WITHIN 7 DAY WITH RECEIPT. STRICTLY NO CASH REFUND."
        ],
    }