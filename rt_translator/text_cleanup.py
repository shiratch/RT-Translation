"""ASR/翻訳出力の繰り返し幻覚を抑制する。"""


def _is_mostly_repetition(text_len: int, removed: int) -> bool:
    return text_len >= 24 and removed > int(text_len * 0.6)


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
    if _is_mostly_repetition(len(tokens), removed):
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
    if _is_mostly_repetition(len(text), removed):
        return ""
    return "".join(out) if removed else text


def collapse_repeats(text: str) -> str:
    """同じ語句や文字列の過剰な連続を1回に畳み、ほぼ全部なら破棄する。"""
    return _collapse_substring_repeats(_collapse_token_repeats(text))
