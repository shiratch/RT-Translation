"""ユーザー辞書。VoiceText プロジェクトの user_dictionary.txt と同じ書式を読む。

書き方(1行1エントリ):
  間違い=正しい   … 出力テキストに対する確実な置換
  用語           … Whisper の hotwords として認識をその語に寄せる(バイアス)
行頭が # の行と空行は無視。

ファイルの更新時刻を見て自動リロードするので、編集すれば次の認識から
反映される(再起動不要)。ファイルが無ければ何もしない。
"""
from pathlib import Path


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

    def hotwords(self) -> str | None:
        """Whisper の hotwords 引数に渡す文字列(なければ None)。"""
        self._refresh()
        return " ".join(dict.fromkeys(self._hotwords)) or None

    def apply(self, text: str) -> str:
        """置換ルールを適用する。認識後の英文にも翻訳後の和文にも使える。"""
        self._refresh()
        for wrong, correct in self._replacements.items():
            text = text.replace(wrong, correct)
        return text
