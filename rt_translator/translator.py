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

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


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
        return "".join(translated)


class TranslationWorker(threading.Thread):
    def __init__(self, cfg: Config, translator: Translator,
                 in_queue: queue.Queue, ui_queue: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(daemon=True, name="mt")
        self.cfg = cfg
        self.translator = translator
        self.in_queue = in_queue
        self.ui_queue = ui_queue
        self.stop_event = stop_event

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
        start = time.perf_counter()
        try:
            japanese = self.translator.translate(entry["text"])
        except Exception as exc:
            print(f"[mt] 翻訳エラー: {exc}")
            return
        if self.cfg.log_latency:
            elapsed = (time.perf_counter() - start) * 1000
            print(f"[mt] {entry['kind']} {elapsed:.0f}ms: {japanese[:60]}")
        self.ui_queue.put({"kind": entry["kind"], "en": entry["text"], "ja": japanese})
