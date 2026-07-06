"""確定字幕を会議ログとしてテキスト保存する。"""
from datetime import datetime
from pathlib import Path
import queue
import threading
import time

from .config import Config


class TranscriptWriter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = cfg.transcript_enabled
        self.path: Path | None = None
        self._lock = threading.Lock()
        if not self.enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            path_text = cfg.transcript_path.format(timestamp=timestamp)
        except Exception as exc:
            print(f"[transcript] transcript_path の展開に失敗、既定値を使います: {exc}")
            path_text = f"transcripts/session_{timestamp}.txt"

        self.path = Path(path_text)
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)

        started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append([
            "# RT Translator Transcript",
            f"# Started: {started}",
            "",
        ])
        print(f"[transcript] 保存先: {self.path}")

    def write_final(self, en: str, ja: str, source_label: str = "",
                    transcript_format: str = "", ja_label: str = ""):
        if not self.enabled or self.path is None:
            return

        en = en.strip()
        ja = ja.strip()
        if not en and not ja:
            return

        mode = (transcript_format or self.cfg.transcript_format).lower()
        if mode not in {"both", "ja", "en"}:
            print(f"[transcript] 未知の transcript_format を both として扱います: {mode}")
            mode = "both"

        now = datetime.now().strftime("%H:%M:%S")
        if mode == "ja":
            lines = [self._with_prefix(now, ja or en, source_label), ""]
        elif mode == "en":
            lines = [self._with_prefix(now, en or ja, source_label), ""]
        else:
            lines = []
            header = self._header(now, source_label)
            if header:
                lines.append(header)
            if en:
                lines.append(f"{self.cfg.transcript_english_label}: {en}")
            if ja:
                label = ja_label or (
                    self.cfg.transcript_translation_label if en
                    else self.cfg.transcript_japanese_label
                )
                lines.append(f"{label}: {ja}")
            lines.append("")

        self._append(lines)

    def _header(self, now: str, source_label: str) -> str:
        parts = []
        if self.cfg.transcript_timestamps:
            parts.append(f"[{now}]")
        if self.cfg.transcript_source_labels and source_label:
            parts.append(source_label)
        return " ".join(parts)

    def _with_prefix(self, now: str, text: str, source_label: str) -> str:
        header = self._header(now, source_label)
        if header:
            return f"{header} {text}"
        return text

    def _append(self, lines: list[str]):
        if self.path is None:
            return
        text = "\n".join(lines) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(text)


class TranscriptOnlyWorker(threading.Thread):
    def __init__(self, cfg: Config, in_queue: queue.Queue, stop_event: threading.Event,
                 transcript_writer: TranscriptWriter, source_language: str,
                 source_label: str):
        super().__init__(daemon=True, name=f"transcript-{source_label.lower()}")
        self.cfg = cfg
        self.in_queue = in_queue
        self.stop_event = stop_event
        self.transcript_writer = transcript_writer
        self.source_language = source_language
        self.source_label = source_label

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
                continue
            if item["kind"] != "final" or not item["text"]:
                continue
            if self._is_japanese_source():
                self.transcript_writer.write_final(
                    "", item["text"], self.source_label, self.cfg.mic_transcript_format)
            else:
                self.transcript_writer.write_final(
                    item["text"], "", self.source_label, self.cfg.mic_transcript_format)
            if self.cfg.log_latency:
                print(f"[transcript] {self.source_label}: {item['text'][:60]}")

    def _is_japanese_source(self) -> bool:
        normalized = self.source_language.lower().replace("-", "_")
        return normalized in {"ja", "jp", "japanese", "jpn", "jpn_jpan", "ja_jp"}
