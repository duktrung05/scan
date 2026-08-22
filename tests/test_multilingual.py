from __future__ import annotations

import json
from pathlib import Path

import pytest

from tscan import __version__
from tscan.evaluation.metrics import text_metrics
from tscan.prompts import build_prompt
from tscan.types import DocumentLanguage, ExtractionMode

LANGUAGE_EXPECTATIONS = {
    DocumentLanguage.EN: ("Language: English", "Preserve capitalization"),
    DocumentLanguage.VI: ("Language: Vietnamese", "Vietnamese diacritic"),
    DocumentLanguage.KO: ("Language: Korean", "Never romanize Korean"),
    DocumentLanguage.JA: ("Language: Japanese", "Never romanize Japanese"),
}


@pytest.mark.parametrize(("language", "expected"), LANGUAGE_EXPECTATIONS.items())
def test_prompt_has_a_language_specific_profile(
    language: DocumentLanguage,
    expected: tuple[str, str],
) -> None:
    prompt = build_prompt(
        mode=ExtractionMode.JSON,
        language=language,
        document_type="document",
        schema={"type": "object"},
    )
    assert expected[0] in prompt
    assert expected[1] in prompt


def test_registry_and_enum_cover_the_same_four_languages() -> None:
    registry_path = Path(__file__).parents[1] / "data" / "language_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registered = {item["code"] for item in registry["languages"]}
    supported = {item.value for item in DocumentLanguage if item != DocumentLanguage.AUTO}
    assert registered == supported == {"en", "vi", "ko", "ja"}


def test_language_aware_primary_text_metrics() -> None:
    assert text_metrics("領収書", "領収証", language="ja")["primary_error_rate"] == text_metrics(
        "領収書", "領収証", language="ja"
    )["cer"]
    assert text_metrics("영수증", "영수정", language="ko")["primary_error_rate"] == text_metrics(
        "영수증", "영수정", language="ko"
    )["cer"]
    assert text_metrics("hóa đơn", "hóa don", language="vi")["primary_error_rate"] == text_metrics(
        "hóa đơn", "hóa don", language="vi"
    )["wer"]


def test_release_version() -> None:
    assert __version__ == "0.3.0"
