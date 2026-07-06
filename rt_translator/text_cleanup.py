"""ASR/翻訳出力の繰り返し幻覚を抑制する。"""
import re

TOKEN_REPETITION_MIN = 8
SUBSTRING_REPETITION_MIN = 24
DEFAULT_ASR_SUPPRESSED_PHRASES = [
    "栗子",
    "株式 関値 位置ループ",
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
]
DEFAULT_TRANSLATION_SUPPRESSED_PHRASES = [
    "栗子",
    "株式 関値 位置ループ",
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
]

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_PHRASE_PUNCT_RE = re.compile(r"[\s　、。,.!?！？]+")
_ENGLISH_SOURCE_LANGS = {"en", "eng", "english", "en_us", "en_gb", "eng_latn"}


def _is_mostly_repetition(text_len: int, removed: int, min_len: int) -> bool:
    return text_len >= min_len and removed > int(text_len * 0.6)


def _collapse_token_repeats(text: str) -> str:
    tokens = text.split()
    if len(tokens) < 3:
        return text
    out = []
    removed = 0
    i = 0
    while i < len(tokens):
        best_n = 0
        best_k = 0
        for n in (4, 3, 2, 1):
            k = 1
            while (i + (k + 1) * n <= len(tokens)
                   and tokens[i:i + n] == tokens[i + k * n:i + (k + 1) * n]):
                k += 1
            if k >= 3:
                best_n = n
                best_k = k
                break
        if best_k:
            out.extend(tokens[i:i + best_n])
            removed += (best_k - 1) * best_n
            i += best_k * best_n
        else:
            out.append(tokens[i])
            i += 1
    if _is_mostly_repetition(len(tokens), removed, TOKEN_REPETITION_MIN):
        return ""
    return " ".join(out) if removed else text


def _collapse_substring_repeats(text: str) -> str:
    if len(text) < 6:
        return text
    out = []
    removed = 0
    i = 0
    while i < len(text):
        best_unit = ""
        best_k = 0
        best_removed = 0
        max_unit = min(16, (len(text) - i) // 3)
        for unit_len in range(1, max_unit + 1):
            unit = text[i:i + unit_len]
            if unit.isspace():
                continue
            k = 1
            while text[i + k * unit_len:i + (k + 1) * unit_len] == unit:
                k += 1
            removed_chars = (k - 1) * unit_len
            if k >= 3 and removed_chars > best_removed:
                best_unit = unit
                best_k = k
                best_removed = removed_chars
        if best_k:
            out.append(best_unit)
            removed += best_removed
            i += best_k * len(best_unit)
        else:
            out.append(text[i])
            i += 1
    if _is_mostly_repetition(len(text), removed, SUBSTRING_REPETITION_MIN):
        return ""
    return "".join(out) if removed else text


def collapse_repeats(text: str) -> str:
    """同じ語句や文字列の過剰な連続を1回に畳み、ほぼ全部なら破棄する。"""
    return _collapse_substring_repeats(_collapse_token_repeats(text))


def _is_english_source_language(language: str) -> bool:
    return language.lower().replace("-", "_") in _ENGLISH_SOURCE_LANGS


def _normalize_phrase(text: str) -> str:
    return _PHRASE_PUNCT_RE.sub("", text).lower()


def _compact_len(text: str) -> int:
    return len(_normalize_phrase(text))


def cleanup_asr_text(text: str, source_language: str,
                     suppressed_phrases: list[str] | tuple[str, ...] | None = None,
                     reject_cjk_for_english: bool = True) -> str:
    """ASR専用の後処理。翻訳結果には適用しない。"""
    text = collapse_repeats(text).strip()
    if not text:
        return ""

    phrases = suppressed_phrases
    if phrases is None:
        phrases = DEFAULT_ASR_SUPPRESSED_PHRASES
    normalized = _normalize_phrase(text)
    if normalized and normalized in {_normalize_phrase(p) for p in phrases}:
        return ""

    if (reject_cjk_for_english
            and _is_english_source_language(source_language)
            and _CJK_RE.search(text)
            and not _LATIN_RE.search(text)):
        return ""
    return text


def cleanup_translation_text(text: str, source_text: str, source_language: str,
                             suppressed_phrases: list[str] | tuple[str, ...] | None = None,
                             reject_short_cjk: bool = True,
                             source_min_chars: int = 24,
                             target_max_chars: int = 12) -> str:
    """翻訳結果専用の後処理。長文入力から短い名詞片だけが出るMT幻覚を落とす。"""
    text = collapse_repeats(text).strip()
    if not text:
        return ""

    phrases = suppressed_phrases
    if phrases is None:
        phrases = DEFAULT_TRANSLATION_SUPPRESSED_PHRASES
    normalized = _normalize_phrase(text)
    if normalized and normalized in {_normalize_phrase(p) for p in phrases}:
        return ""

    if (reject_short_cjk
            and _is_english_source_language(source_language)
            and _compact_len(source_text) >= source_min_chars
            and _CJK_RE.search(text)
            and not _LATIN_RE.search(text)
            and not _HIRAGANA_RE.search(text)
            and _compact_len(text) <= target_max_chars):
        return ""
    return text
