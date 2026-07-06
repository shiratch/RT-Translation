"""翻訳エンジン。既定は NLLB-200 (CTranslate2 変換済み) のローカル GPU 実行。

Translator インターフェースを実装すれば DeepL / Claude API 等に差し替え可能。
翻訳ワーカーは ASR からの queue を読み、古い暫定結果は捨てて最新だけ翻訳する
(確定結果は必ず順番どおり全部翻訳する)。
"""
import queue
import re
import threading
import time

from .config import Config
from .text_cleanup import collapse_repeats

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_JAPANESE_SOURCE_LANGS = {"ja", "jp", "japanese", "jpn", "jpn_jpan", "ja_jp"}


def is_japanese_source_language(language: str) -> bool:
    return language.lower().replace("-", "_") in _JAPANESE_SOURCE_LANGS


class Translator:
    def translate(self, text: str) -> str:
        raise NotImplementedError


class NllbCT2Translator(Translator):
    def __init__(self, cfg: Config):
        import ctranslate2
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        print(f"[mt] NLLB モデルを準備中: {cfg.nllb_repo}")
        model_dir = snapshot_download(cfg.nllb_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, src_lang=cfg.source_lang_nllb)
        try:
            self.model = ctranslate2.Translator(
                model_dir, device=cfg.device, compute_type=cfg.nllb_compute_type)
            print(f"[mt] NLLB を {cfg.device} でロードしました")
        except Exception as exc:
            if cfg.device == "cuda":
                print(f"[mt] CUDA でのロードに失敗、CPU にフォールバックします: {exc}")
                self.model = ctranslate2.Translator(
                    model_dir, device="cpu", compute_type="int8")
            else:
                raise
        self.target_lang = cfg.target_lang

    def translate(self, text: str) -> str:
        # NLLB は長い複文で訳が劣化するため、文単位に分割して一括バッチ翻訳する
        sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()] or [text]
        batch = [self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(s))
                 for s in sentences]
        results = self.model.translate_batch(
            batch,
            target_prefix=[[self.target_lang]] * len(batch),
            beam_size=1,
            max_decoding_length=256,
        )
        translated = []
        for result in results:
            output = result.hypotheses[0]
            if output and output[0] == self.target_lang:
                output = output[1:]
            ids = self.tokenizer.convert_tokens_to_ids(output)
            translated.append(self.tokenizer.decode(ids, skip_special_tokens=True).strip())
        return " ".join(translated)


class TranslationWorker(threading.Thread):
    def __init__(self, cfg: Config, translator: Translator | None,
                 in_queue: queue.Queue, ui_queue: queue.Queue,
                 stop_event: threading.Event, dictionary=None, transcript_writer=None):
        super().__init__(daemon=True, name="mt")
        self.cfg = cfg
        self.translator = translator
        self.in_queue = in_queue
        self.ui_queue = ui_queue
        self.stop_event = stop_event
        self.dictionary = dictionary
        self.transcript_writer = transcript_writer
        self._passthrough_source = is_japanese_source_language(cfg.source_language)
        # 暫定字幕のチラつき防止: 完結した文の訳をセグメント内でキャッシュし、
        # 表示済みの訳が更新のたびに書き換わらないようにする
        self._sentence_cache: dict[str, str] = {}
        self._last_partial_result: dict | None = None

    def run(self):
        while not self.stop_event.is_set():
            try:
                item = self.in_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # 溜まっていたら確定は全部処理し、暫定は最新の 1 件だけ残す
            pending = [item]
            while True:
                try:
                    pending.append(self.in_queue.get_nowait())
                except queue.Empty:
                    break
            finals = [p for p in pending if p["kind"] == "final"]
            partials = [p for p in pending if p["kind"] == "partial"]
            todo = finals + (partials[-1:] if not finals else [])
            for entry in todo:
                self._translate_one(entry)

    def _translate_one(self, entry: dict):
        if not entry["text"]:
            if entry["kind"] == "final" and self._emit_partial_fallback("final text empty"):
                return
            # 空テキスト(幻覚破棄によるクリア指示)はそのまま UI へ
            self._emit_result({"kind": entry["kind"], "en": "", "ja": "",
                               "speaker_change": False})
            if entry["kind"] == "final":
                self._last_partial_result = None
                self._sentence_cache.clear()
            return
        start = time.perf_counter()
        try:
            if self._passthrough_source:
                japanese = entry["text"]
            elif entry["kind"] == "partial":
                japanese = self._translate_partial(entry["text"])
            else:
                if self.translator is None:
                    raise RuntimeError("翻訳モデルが未初期化です")
                japanese = self.translator.translate(entry["text"])
                self._sentence_cache.clear()  # セグメント確定でキャッシュを捨てる
        except Exception as exc:
            print(f"[mt] 翻訳エラー: {exc}")
            if entry["kind"] == "final":
                self._emit_partial_fallback("final translate error")
            return
        if self.dictionary is not None:
            japanese = self.dictionary.apply(japanese)
        cleaned = collapse_repeats(japanese)
        if cleaned != japanese and self.cfg.log_latency:
            print(f"[mt] 繰り返し幻覚を{'除去' if not cleaned else '圧縮'}: {japanese[:60]}")
        japanese = cleaned
        if entry["kind"] == "final" and not japanese:
            if self._emit_partial_fallback("final translation empty"):
                return
        if self.cfg.log_latency:
            elapsed = (time.perf_counter() - start) * 1000
            print(f"[mt] {entry['kind']} {elapsed:.0f}ms: {japanese[:60]}")
        source_text = "" if self._passthrough_source else entry["text"]
        result = {"kind": entry["kind"], "en": source_text, "ja": japanese,
                  "speaker_change": entry.get("speaker_change", False)}
        if entry["kind"] == "partial" and japanese:
            self._last_partial_result = result
        elif entry["kind"] == "final":
            self._last_partial_result = None
        self._emit_result(result)

    def _emit_partial_fallback(self, reason: str) -> bool:
        """確定結果が空/失敗したとき、直近の暫定翻訳を確定として残す。"""
        if not self._last_partial_result:
            return False
        cached = dict(self._last_partial_result)
        cached["kind"] = "final"
        self._emit_result(cached)
        self._last_partial_result = None
        self._sentence_cache.clear()
        if self.cfg.log_latency:
            print(f"[mt] {reason}; 直近の partial を final として採用: {cached['ja'][:60]}")
        return True

    def _emit_result(self, result: dict):
        if result["kind"] == "final" and self.transcript_writer is not None:
            self.transcript_writer.write_final(
                result.get("en", ""), result.get("ja", ""),
                self.cfg.remote_transcript_label,
                self.cfg.remote_transcript_format)
        self.ui_queue.put(result)

    def _translate_partial(self, text: str) -> str:
        """完結した文はキャッシュした訳を使い回し、末尾の言いかけの文だけを
        訳し直す。グレー字幕の頭部分が更新のたびに変わるのを防ぐ。"""
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if not sentences:
            return ""
        parts = []
        for i, sentence in enumerate(sentences):
            complete = i < len(sentences) - 1 or sentence.endswith((".", "!", "?"))
            if complete:
                if sentence not in self._sentence_cache:
                    if len(self._sentence_cache) > 100:
                        self._sentence_cache.clear()
                    if self.translator is None:
                        raise RuntimeError("翻訳モデルが未初期化です")
                    self._sentence_cache[sentence] = self.translator.translate(sentence)
                parts.append(self._sentence_cache[sentence])
            else:
                if self.translator is None:
                    raise RuntimeError("翻訳モデルが未初期化です")
                parts.append(self.translator.translate(sentence))
        return " ".join(p for p in parts if p)
