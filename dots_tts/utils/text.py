from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

try:
    from langcodes import Language as LangcodesLanguage
except Exception:
    LangcodesLanguage = None

try:
    from lingua import Language, LanguageDetectorBuilder
except Exception:
    Language = None
    LanguageDetectorBuilder = None

try:
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
except Exception:
    ZhNormalizer = None
    EnNormalizer = None

TextLanguage = Literal["zh", "en", "unknown"]

_WHITESPACE_PATTERN = re.compile(r"\s+")


@lru_cache(maxsize=1)
def get_chinese_text_normalizer() -> ZhNormalizer:
    if ZhNormalizer is None:
        raise ImportError("WeTextProcessing is required for Chinese text normalization.")
    return ZhNormalizer()


@lru_cache(maxsize=1)
def get_english_text_normalizer() -> EnNormalizer:
    if EnNormalizer is None:
        raise ImportError("WeTextProcessing is required for English text normalization.")
    return EnNormalizer()


@lru_cache(maxsize=1)
def get_language_detector():
    if Language is None or LanguageDetectorBuilder is None:
        raise ImportError("lingua-language-detector is required for automatic language detection.")
    supported_languages = tuple(
        sorted(Language.all(), key=lambda language: language.name)
    )
    return LanguageDetectorBuilder.from_languages(*supported_languages).build()


def _lingua_language_to_code(language: Language | None) -> str | None:
    if language is None:
        return None
    iso_code_639_1 = getattr(language.iso_code_639_1, "name", None)
    if iso_code_639_1:
        return iso_code_639_1.lower()
    iso_code_639_3 = getattr(language.iso_code_639_3, "name", None)
    if iso_code_639_3:
        return iso_code_639_3.lower()
    return language.name.lower()


def detect(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    if Language is None or LanguageDetectorBuilder is None:
        cjk_count = sum(1 for char in stripped if "\u4e00" <= char <= "\u9fff")
        alpha_count = sum(1 for char in stripped if char.isalpha())
        ascii_alpha_count = sum(1 for char in stripped if ("a" <= char.lower() <= "z"))
        if cjk_count > 0 and cjk_count >= max(1, alpha_count // 3):
            return "zh"
        if ascii_alpha_count > 0 and ascii_alpha_count >= max(1, alpha_count // 2):
            return "en"
        return None
    language = get_language_detector().detect_language_of(stripped)
    return _lingua_language_to_code(language)


def normalize_language_code(language: str | None) -> str | None:
    if language is None:
        return None

    stripped = language.strip()
    if not stripped or stripped.lower() in {"none", "unknown"}:
        return None
    if stripped.startswith("口音:"):
        return stripped

    fallback_names = {
        "auto": None,
        "auto_detect": None,
        "english": "EN",
        "chinese": "ZH",
        "mandarin": "ZH",
        "cantonese": "口音:粤语",
        "japanese": "JA",
        "korean": "KO",
        "spanish": "ES",
        "french": "FR",
        "german": "DE",
        "arabic": "AR",
        "hindi": "HI",
        "portuguese": "PT",
        "russian": "RU",
        "italian": "IT",
        "turkish": "TR",
        "vietnamese": "VI",
    }
    lowered = stripped.lower()
    if lowered in fallback_names:
        return fallback_names[lowered]
    if stripped.isalpha() and 2 <= len(stripped) <= 3:
        return stripped.upper()

    if LangcodesLanguage is not None:
        for resolver in (LangcodesLanguage.get, LangcodesLanguage.find):
            try:
                normalized_language = resolver(stripped).prefer_macrolanguage()
            except Exception:
                continue

            language_code = (normalized_language.language or "").strip().upper()
            if language_code and language_code != "UND":
                return language_code
    return None


def attach_language_tag(text: str, language: str | None) -> str:
    if not text:
        return text

    language_code = normalize_language_code(language)
    if language_code is None:
        return text

    if language_code == "YUE":
        language_code = "口音:粤语"

    language_tag = f"[{language_code}]"
    if text.startswith(language_tag):
        return text
    return f"{language_tag}{text}"


def detect_text_language(text: str) -> TextLanguage:
    language_code = detect(text)
    if language_code == "zh":
        return "zh"
    if language_code == "en":
        return "en"
    return "unknown"


def _normalize_with(normalizer, text: str) -> str:
    normalized = normalizer.normalize(text)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalize_chinese_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if ZhNormalizer is None:
        return stripped
    return _normalize_with(get_chinese_text_normalizer(), stripped)


def normalize_english_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if EnNormalizer is None:
        return stripped
    return _normalize_with(get_english_text_normalizer(), stripped)


def normalize_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    language = detect_text_language(stripped)
    if language == "zh" and ZhNormalizer is not None:
        return _normalize_with(get_chinese_text_normalizer(), stripped)
    if language == "en" and EnNormalizer is not None:
        return _normalize_with(get_english_text_normalizer(), stripped)
    return stripped
