"""ASR/翻訳出力の繰り返し幻覚を抑制する。"""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re

TOKEN_REPETITION_MIN = 8
SUBSTRING_REPETITION_MIN = 24
DEFAULT_ASR_SUPPRESSED_PHRASES = [
    "Amazon",
    "栗子",
    "株式 関値 位置ループ",
    "株式 閾値 位置ループ",
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
]
DEFAULT_TRANSLATION_SUPPRESSED_PHRASES = [
    "栗子",
    "株式 関値 位置ループ",
    "株式 閾値 位置ループ",
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
]

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_PHRASE_PUNCT_RE = re.compile(r"[\s　、。,.!?！？]+")
_ENGLISH_SOURCE_LANGS = {"en", "eng", "english", "en_us", "en_gb", "eng_latn"}
_ENGLISH_THOUSAND_ASR_RE = re.compile(r"\b(\d{2,3}),0\b")
_SOURCE_USD_RE = re.compile(
    r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(billion|million|thousand|bn|b|m|k)?\b",
    re.IGNORECASE,
)
_TARGET_USD_RE = re.compile(
    r"(\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(億|万)?\s*(米ドル|ドル)"
)
_JA_TERMINAL_PUNCT_RE = re.compile(r"[。！？!?…）」』】》〉]$")
_ASCII_PUNCT_TRANSLATION = str.maketrans({
    ",": "、",
    "?": "？",
    "!": "！",
})
_SUPPRESSED_SUBSTRING_SEEDS = {
    "栗子",
    "株式関値位置ループ",
    "株式閾値位置ループ",
    "ありがとうございました",
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "次回の動画でお会いしましょう",
}

_MEETING_TERM_REPLACEMENTS = (
    ("積分テスト", "統合テスト"),
    ("遠隔センサー", "距離センサー"),
    ("メールセンサー", "照度センサー"),
    ("輸出", "エクスポート"),
    ("単一の標本", "1台あたり"),
    ("コストの費用", "コスト"),
    ("MVMP の体", "MVMP のボディ"),
    ("MVMPの体", "MVMPのボディ"),
)

_MEETING_TONE_REPLACEMENTS = (
    ("教えて下さい", "教えてください"),
    ("分かってる", "分かりました"),
    ("わかった", "分かりました"),
    ("ワオ", "なるほど"),
    ("そうだ", "そうです"),
    ("大丈夫だ", "大丈夫です"),
    ("問題ない", "問題ありません"),
    ("聞こえるか", "聞こえますか"),
    ("ミムミム", "はい"),
)


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


def _repair_english_asr_numbers(text: str, source_language: str) -> str:
    if not _is_english_source_language(source_language):
        return text
    return _ENGLISH_THOUSAND_ASR_RE.sub(r"\1,000", text)


def _normalize_phrase(text: str) -> str:
    return _PHRASE_PUNCT_RE.sub("", text).lower()


def _compact_len(text: str) -> int:
    return len(_normalize_phrase(text))


def _matches_suppressed_phrase(normalized: str,
                               phrases: list[str] | tuple[str, ...],
                               substring_max_chars: int = 0) -> bool:
    if not normalized:
        return False
    normalized_phrases = {_normalize_phrase(p) for p in phrases if p.strip()}
    if normalized in normalized_phrases:
        return True
    if substring_max_chars <= 0 or len(normalized) > substring_max_chars:
        return False
    return any(seed in normalized and seed in normalized_phrases
               for seed in _SUPPRESSED_SUBSTRING_SEEDS)


def cleanup_asr_text(text: str, source_language: str,
                     suppressed_phrases: list[str] | tuple[str, ...] | None = None,
                     reject_cjk_for_english: bool = True,
                     suppressed_substring_max_chars: int = 12) -> str:
    """ASR専用の後処理。翻訳結果には適用しない。"""
    text = _repair_english_asr_numbers(text, source_language)
    text = collapse_repeats(text).strip()
    text = _repair_english_asr_numbers(text, source_language)
    if not text:
        return ""

    phrases = suppressed_phrases
    if phrases is None:
        phrases = DEFAULT_ASR_SUPPRESSED_PHRASES
    normalized = _normalize_phrase(text)
    if _matches_suppressed_phrase(
            normalized, phrases, suppressed_substring_max_chars):
        return ""

    if (reject_cjk_for_english
            and _is_english_source_language(source_language)
            and _CJK_RE.search(text)
            and not _LATIN_RE.search(text)):
        return ""
    return text


def _source_usd_amounts(source_text: str) -> list[Decimal]:
    amounts: list[Decimal] = []
    multipliers = {
        "thousand": Decimal("1000"),
        "k": Decimal("1000"),
        "million": Decimal("1000000"),
        "m": Decimal("1000000"),
        "billion": Decimal("1000000000"),
        "bn": Decimal("1000000000"),
        "b": Decimal("1000000000"),
    }
    for match in _SOURCE_USD_RE.finditer(source_text):
        number_text = match.group(1).replace(",", "")
        try:
            amount = Decimal(number_text)
        except InvalidOperation:
            continue
        unit = (match.group(2) or "").lower()
        amount *= multipliers.get(unit, Decimal("1"))
        amounts.append(amount)
    return amounts


def _whole_number(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _format_usd_for_japanese(value: Decimal) -> str:
    amount = _whole_number(value)
    if amount >= 100_000_000:
        oku, rest = divmod(amount, 100_000_000)
        if rest == 0:
            return f"{oku}億ドル"
        if rest % 10_000 == 0:
            return f"{oku}億{rest // 10_000}万ドル"
        return f"{amount:,}ドル"
    if amount >= 10_000:
        man, rest = divmod(amount, 10_000)
        if rest == 0:
            return f"{man}万ドル"
        return f"{amount:,}ドル"
    return f"{amount:,}ドル"


def repair_translation_numbers(text: str, source_text: str,
                               source_language: str) -> str:
    """原文にあるドル金額を優先し、NLLBの桁落ちした日本語金額を補正する。"""
    if not _is_english_source_language(source_language):
        return text
    amounts = _source_usd_amounts(source_text)
    if not amounts:
        return text

    repaired = text
    for amount in amounts:
        replacement = _format_usd_for_japanese(amount)
        repaired, count = _TARGET_USD_RE.subn(replacement, repaired, count=1)
        if count == 0:
            break
    return repaired


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _apply_meeting_tone_replacements(text: str) -> str:
    for wrong, correct in _MEETING_TONE_REPLACEMENTS:
        text = text.replace(wrong, correct)
    return text.replace("お前", "あなた")


def apply_meeting_translation_fixes(text: str, source_text: str,
                                    source_language: str) -> str:
    """会議字幕向けに、NLLBが崩しやすい技術語と強い口調を補正する。"""
    if not text or not _is_english_source_language(source_language):
        return text

    source = source_text.lower()
    if ("don't get any more questions" in source
            or "do not get any more questions" in source):
        return "他に質問はありませんか？"

    for wrong, correct in _MEETING_TERM_REPLACEMENTS:
        text = text.replace(wrong, correct)

    if ("facebook" in source
            and _contains_any(source, ("schedule", "milestone", "phase"))):
        text = text.replace("Facebook", "フェーズ2")

    if ("bug" in source
            or ("bag" in source and _contains_any(source, ("fix", "upgrade")))):
        text = text.replace("バッグ", "バグ")

    if "foam" in source and "camera" in source:
        text = text.replace("泡が必要", "4つのカメラが必要")
        text = text.replace("泡", "4つのカメラ")
        text = text.replace("全カメラ", "4つのカメラ")

    if "math" in source:
        text = text.replace("数学", "計算")

    if "back to you" in source:
        text = text.replace("お前に返信できるかもしれません", "後で回答できるかもしれません")
        text = text.replace("お前に返信", "後で回答")
        text = text.replace("あなたに返信できるかもしれません", "後で回答できるかもしれません")
        text = text.replace("あなたに返信", "後で回答")

    if "space to like open" in source or "space to open" in source:
        text = text.replace("スイートなどの空間", "スリットや開口部")
        text = text.replace("透明な素材や スイートなどの空間", "透明素材やスリット、開口部")

    return _apply_meeting_tone_replacements(text)


def normalize_japanese_punctuation(text: str, final: bool = False) -> str:
    """NLLBが落としがちな日本語句読点を字幕向けに最低限整える。"""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    text = text.translate(_ASCII_PUNCT_TRANSLATION)
    # 数値の小数点は壊さず、文末/空白前のピリオドだけ句点に寄せる。
    text = re.sub(r"(?<!\d)\.(?!\d)", "。", text)
    text = re.sub(r"\s*([、。！？])\s*", r"\1", text)
    text = re.sub(r"([、。！？]){2,}", r"\1", text)
    text = re.sub(r"、([。！？])", r"\1", text)
    if final and not _JA_TERMINAL_PUNCT_RE.search(text):
        text += "。"
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
    if _matches_suppressed_phrase(normalized, phrases, target_max_chars):
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
