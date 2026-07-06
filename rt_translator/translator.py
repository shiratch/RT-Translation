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
from .text_cleanup import (cleanup_translation_text, normalize_japanese_punctuation,
                           repair_translation_numbers)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SENTENCE_END = re.compile(r'[.!?。！？]["\')\]]*\s*$')
_FORMULA_FRAGMENT_RE = re.compile(
    r"\b(formula|study|practice|plus|times|time|equals?|fluency)\b",
    re.IGNORECASE,
)
_JAPANESE_SOURCE_LANGS = {"ja", "jp", "japanese", "jpn", "jpn_jpan", "ja_jp"}
MAX_SENTENCE_CACHE = 100


def is_japanese_source_language(language: str) -> bool:
    return language.lower().replace("-", "_") in _JAPANESE_SOURCE_LANGS


class Translator:
    def translate(self, text: str, beam_size: int = 1,
                  length_penalty: float = 1.0,
                  repetition_penalty: float = 1.0,
                  no_repeat_ngram_size: int = 0,
                  max_decoding_length: int = 256,
                  max_unit_chars: int = 180,
                  split_sentences: bool = True) -> str:
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

    def translate(self, text: str, beam_size: int = 1,
                  length_penalty: float = 1.0,
                  repetition_penalty: float = 1.0,
                  no_repeat_ngram_size: int = 0,
                  max_decoding_length: int = 256,
                  max_unit_chars: int = 180,
                  split_sentences: bool = True) -> str:
        # 長文は文単位に分割するが、確定字幕の短〜中程度の段落は文脈を
        # 保ったまま一括入力する方が、断片訳になりにくい。
        units = self._translation_units(text, split_sentences, max_unit_chars)
        batch = [self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(unit))
                 for unit in units]
        translate_kwargs = {
            "target_prefix": [[self.target_lang]] * len(batch),
            "beam_size": beam_size,
            "max_decoding_length": max_decoding_length,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
        }
        if no_repeat_ngram_size > 0:
            translate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        try:
            results = self.model.translate_batch(batch, **translate_kwargs)
        except TypeError:
            # 古い CTranslate2 でも最低限 beam/length は使えるようにする
            translate_kwargs.pop("repetition_penalty", None)
            translate_kwargs.pop("no_repeat_ngram_size", None)
            results = self.model.translate_batch(batch, **translate_kwargs)
        translated = []
        for result in results:
            output = result.hypotheses[0]
            if output and output[0] == self.target_lang:
                output = output[1:]
            ids = self.tokenizer.convert_tokens_to_ids(output)
            translated.append(self.tokenizer.decode(ids, skip_special_tokens=True).strip())
        return " ".join(translated)

    def _translation_units(self, text: str, split_sentences: bool,
                           max_unit_chars: int) -> list[str]:
        if not split_sentences:
            return [text.strip()] if text.strip() else [text]
        units = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()] or [text]
        split_units = []
        for unit in units:
            split_units.extend(self._split_long_unit(unit, max_unit_chars))
        return split_units

    def _split_long_unit(self, text: str, max_chars: int) -> list[str]:
        if max_chars <= 0 or len(text) <= max_chars:
            return [text]
        separators = (
            ", ", "; ", ": ", " and ", " but ", " which ", " because ",
            " so ", " at the moment ", " to ", " from ",
        )
        chunks = []
        rest = text.strip()
        min_cut = max(40, max_chars // 3)
        while len(rest) > max_chars:
            cut = 0
            for sep in separators:
                pos = rest.rfind(sep, min_cut, max_chars + 1)
                if pos > cut:
                    cut = pos + len(sep)
            if cut <= 0:
                cut = rest.rfind(" ", min_cut, max_chars + 1)
            if cut <= 0:
                cut = max_chars
            chunks.append(rest[:cut].strip())
            rest = rest[cut:].strip()
        if rest:
            chunks.append(rest)
        return chunks


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
        self._final_buffer: list[dict] = []
        self._final_buffer_updated_at = 0.0

    def run(self):
        idle_since = 0.0
        while True:
            if self.stop_event.is_set() and self.in_queue.empty():
                if not idle_since:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since >= self.cfg.shutdown_drain_seconds:
                    break
            else:
                idle_since = 0.0
            try:
                item = self.in_queue.get(timeout=0.1)
            except queue.Empty:
                self._flush_stale_final_buffer()
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
            for entry in finals:
                self._handle_final(entry)
            self._flush_stale_final_buffer()
            if partials:
                self._translate_one(partials[-1])
        self._flush_final_buffer("stop")

    def _handle_final(self, entry: dict, allow_split: bool = True):
        if (self._passthrough_source
                or not self.cfg.translation_buffer_final_fragments):
            self._flush_final_buffer("immediate final")
            self._translate_one(entry)
            return

        if not entry["text"]:
            self._flush_final_buffer("empty final")
            self._translate_one(entry)
            return

        if entry.get("speaker_change", False) and self._final_buffer:
            self._flush_final_buffer("speaker change")

        if allow_split:
            split = self._split_deferred_fragment(entry["text"])
            if split is not None:
                head, tail = split
                head_entry = dict(entry)
                head_entry["text"] = head
                self._handle_final(head_entry, allow_split=False)

                tail_entry = dict(entry)
                tail_entry["text"] = tail
                tail_entry["defer_flush"] = True
                self._handle_final(tail_entry, allow_split=False)
                return

        self._final_buffer.append(entry)
        self._final_buffer_updated_at = time.monotonic()
        if self._should_flush_final_buffer():
            self._flush_final_buffer("complete final")

    def _should_flush_final_buffer(self) -> bool:
        if not self._final_buffer:
            return False
        text = self._final_buffer_text()
        latest_text = self._final_buffer[-1]["text"].strip()
        latest_deferred = self._final_buffer[-1].get("defer_flush", False)
        if bool(_SENTENCE_END.search(latest_text)) and not latest_deferred:
            return True
        compact_len = len(re.sub(r"\s+", "", text))
        return (
            compact_len >= self.cfg.translation_fragment_max_chars
            or len(self._final_buffer) >= self.cfg.translation_fragment_max_segments
        )

    def _flush_stale_final_buffer(self):
        if not self._final_buffer:
            return
        latest_text = self._final_buffer[-1]["text"].strip()
        latest_complete = bool(_SENTENCE_END.search(latest_text))
        timeout = (
            self.cfg.translation_fragment_flush_seconds
            if latest_complete
            else self.cfg.translation_incomplete_fragment_flush_seconds
        )
        if any(entry.get("defer_flush", False) for entry in self._final_buffer):
            timeout = self.cfg.translation_deferred_fragment_flush_seconds
        if time.monotonic() - self._final_buffer_updated_at >= timeout:
            self._flush_final_buffer("timeout")

    def _flush_final_buffer(self, reason: str):
        if not self._final_buffer:
            return
        first = self._final_buffer[0]
        combined = dict(first)
        combined["text"] = self._final_buffer_text()
        combined["speaker_change"] = first.get("speaker_change", False)
        count = len(self._final_buffer)
        self._final_buffer = []
        self._final_buffer_updated_at = 0.0
        if self.cfg.log_latency and count > 1:
            print(f"[mt] final断片を結合({reason}, {count}件): {combined['text'][:80]}")
        self._translate_one(combined)

    def _final_buffer_text(self) -> str:
        return " ".join(
            entry["text"].strip() for entry in self._final_buffer
            if entry["text"].strip()
        )

    def _split_deferred_fragment(self, text: str) -> tuple[str, str] | None:
        if not self.cfg.translation_defer_formula_fragments:
            return None
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if len(sentences) < 2:
            return None
        tail = sentences[-1]
        if len(tail) > self.cfg.translation_formula_fragment_max_chars:
            return None
        if not _FORMULA_FRAGMENT_RE.search(tail):
            return None
        head = " ".join(sentences[:-1]).strip()
        previous = sentences[-2]
        if "formula" not in head.lower() and not _FORMULA_FRAGMENT_RE.search(previous):
            return None
        return head, tail

    def _translate_one(self, entry: dict):
        if not entry["text"]:
            # 空テキスト(幻覚破棄によるクリア指示)はそのまま UI へ
            self._emit_result({"kind": entry["kind"], "en": "", "ja": "",
                               "speaker_change": False})
            if entry["kind"] == "final":
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
                japanese = self._translate_final(entry["text"])
                self._sentence_cache.clear()  # セグメント確定でキャッシュを捨てる
        except Exception as exc:
            print(f"[mt] 翻訳エラー: {exc}")
            if entry["kind"] == "final":
                self._emit_empty_final(entry, "final translate error")
            return
        if self.dictionary is not None and not self._passthrough_source:
            japanese = self.dictionary.apply(japanese)
        if not self._passthrough_source:
            repaired = repair_translation_numbers(
                japanese, entry["text"], self.cfg.source_language)
            if repaired != japanese and self.cfg.log_latency:
                print(f"[mt] 原文金額に合わせて補正: {japanese[:60]} -> {repaired[:60]}")
            japanese = repaired
        raw_japanese = japanese
        cleaned = cleanup_translation_text(
            japanese,
            entry["text"],
            self.cfg.source_language,
            self.cfg.translation_suppressed_phrases,
            self.cfg.translation_reject_short_cjk and entry["kind"] == "final",
            self.cfg.translation_suspicious_source_min_chars,
            self.cfg.translation_suspicious_target_max_chars,
        )
        if cleaned != japanese and self.cfg.log_latency:
            print(f"[mt] 翻訳幻覚候補を{'除去' if not cleaned else '補正'}: {japanese[:60]}")
        japanese = cleaned
        if (japanese
                and not self._passthrough_source
                and self.cfg.translation_normalize_punctuation):
            japanese = normalize_japanese_punctuation(
                japanese, final=entry["kind"] == "final")
        if entry["kind"] == "final" and not japanese:
            if raw_japanese:
                self._emit_empty_final(
                    entry,
                    "final translation rejected",
                    transcript_ja=raw_japanese,
                    transcript_ja_label=self.cfg.transcript_rejected_translation_label,
                )
                return
            self._emit_empty_final(
                entry,
                "final translation empty",
                transcript_ja="(empty)",
                transcript_ja_label=self.cfg.transcript_rejected_translation_label,
            )
            return
        if self.cfg.log_latency:
            elapsed = (time.perf_counter() - start) * 1000
            source_preview = entry["text"][:60]
            print(f"[mt] {entry['kind']} {elapsed:.0f}ms: {source_preview} -> {japanese[:60]}")
        source_text = "" if self._passthrough_source else entry["text"]
        result = {"kind": entry["kind"], "en": source_text, "ja": japanese,
                  "speaker_change": entry.get("speaker_change", False)}
        self._emit_result(result)

    def _emit_empty_final(self, entry: dict, reason: str,
                          transcript_ja: str = "", transcript_ja_label: str = ""):
        source_text = "" if self._passthrough_source else entry["text"]
        self._emit_result({
            "kind": "final",
            "en": source_text,
            "ja": "",
            "transcript_ja": transcript_ja,
            "transcript_ja_label": transcript_ja_label,
            "speaker_change": entry.get("speaker_change", False),
        })
        self._sentence_cache.clear()
        if self.cfg.log_latency:
            print(f"[mt] {reason}; 空の final としてクリア")

    def _emit_result(self, result: dict):
        if result["kind"] == "final" and self.transcript_writer is not None:
            self.transcript_writer.write_final(
                result.get("en", ""),
                result.get("transcript_ja", result.get("ja", "")),
                self.cfg.remote_transcript_label,
                self.cfg.remote_transcript_format,
                result.get("transcript_ja_label", ""))
        self.ui_queue.put(result)

    def _translate_final(self, text: str) -> str:
        if self.translator is None:
            raise RuntimeError("翻訳モデルが未初期化です")
        return self.translator.translate(
            text,
            beam_size=self.cfg.translation_final_beam_size,
            length_penalty=self.cfg.translation_length_penalty,
            repetition_penalty=self.cfg.translation_repetition_penalty,
            no_repeat_ngram_size=self.cfg.translation_no_repeat_ngram_size,
            max_decoding_length=self.cfg.translation_max_decoding_length,
            max_unit_chars=self.cfg.translation_unit_max_chars,
            split_sentences=not self._should_translate_as_single_pass(text),
        )

    def _translate_fast(self, text: str) -> str:
        if self.translator is None:
            raise RuntimeError("翻訳モデルが未初期化です")
        return self.translator.translate(
            text,
            beam_size=self.cfg.translation_partial_beam_size,
            max_decoding_length=self.cfg.translation_max_decoding_length,
            max_unit_chars=self.cfg.translation_unit_max_chars,
            split_sentences=True,
        )

    def _should_translate_as_single_pass(self, text: str) -> bool:
        if not self.cfg.translation_final_single_pass:
            return False
        compact_len = len(re.sub(r"\s+", "", text))
        return compact_len <= self.cfg.translation_single_pass_max_chars

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
                    if len(self._sentence_cache) >= MAX_SENTENCE_CACHE:
                        self._drop_old_sentence_cache_entries()
                    self._sentence_cache[sentence] = self._translate_fast(sentence)
                parts.append(self._sentence_cache[sentence])
            else:
                parts.append(self._translate_fast(sentence))
        return " ".join(p for p in parts if p)

    def _drop_old_sentence_cache_entries(self):
        drop_count = max(1, len(self._sentence_cache) // 2)
        for key in list(self._sentence_cache)[:drop_count]:
            self._sentence_cache.pop(key, None)
