"""ユーザー辞書。VoiceText プロジェクトの user_dictionary.txt と同じ書式を読む。

書き方(1行1エントリ):
  間違い=正しい   … 出力テキストに対する確実な置換
  用語           … Whisper の hotwords として認識をその語に寄せる(バイアス)
行頭が # の行と空行は無視。

ファイルの更新時刻を見て自動リロードするので、編集すれば次の認識から
反映される(再起動不要)。ファイルが無ければ何もしない。
"""
import re
from pathlib import Path

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_ENGLISH_SOURCE_LANGS = {"en", "eng", "english", "en_us", "en_gb", "eng_latn"}
_JAPANESE_SOURCE_LANGS = {"ja", "jp", "japanese", "jpn", "jpn_jpan", "ja_jp"}


def _normalize_language(language: str) -> str:
    return language.lower().replace("-", "_")


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _matches_asr_language(text: str, source_language: str | None) -> bool:
    if not source_language:
        return True
    normalized = _normalize_language(source_language)
    if normalized in _JAPANESE_SOURCE_LANGS:
        return True
    if normalized in _ENGLISH_SOURCE_LANGS:
        return not _contains_cjk(text)
    return True


class UserDictionary:
    def __init__(self, path: str):
        self.path = Path(path)
        self._mtime = None
        self._replacements: dict[str, str] = {}
        self._hotwords: list[str] = []
        if self.path.is_file():
            self._refresh()
            print(f"[dict] ユーザー辞書を使用: {self.path} "
                  f"(置換 {len(self._replacements)} 件 / 用語 {len(self._hotwords)} 語)")

    def _refresh(self):
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._mtime = None
            self._replacements, self._hotwords = {}, []
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        replacements: dict[str, str] = {}
        hotwords: list[str] = []
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                wrong, _, correct = line.partition("=")
                wrong, correct = wrong.strip(), correct.strip()
                if wrong and correct:
                    replacements[wrong] = correct
                    hotwords.append(correct)
            else:
                hotwords.append(line)
        self._replacements = replacements
        self._hotwords = hotwords

    def hotwords(self, source_language: str | None = None) -> str | None:
        """Whisper の hotwords 引数に渡す文字列(なければ None)。"""
        self._refresh()
        words = [
            word for word in dict.fromkeys(self._hotwords)
            if _matches_asr_language(word, source_language)
        ]
        return " ".join(words) or None

    def apply(self, text: str, source_language: str | None = None) -> str:
        """置換ルールを適用する。認識後の英文にも翻訳後の和文にも使える。"""
        self._refresh()
        for wrong, correct in self._replacements.items():
            if (not _matches_asr_language(wrong, source_language)
                    or not _matches_asr_language(correct, source_language)):
                continue
            text = text.replace(wrong, correct)
        return text
