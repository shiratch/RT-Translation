"""faster-whisper による低遅延ストリーミング認識。

戦略:
- 音声を発話バッファに溜め、partial_interval ごとにバッファ全体を暫定認識
- Silero VAD (faster-whisper 同梱) で末尾無音を監視し、
  silence_finalize 秒続いたらそのセグメントを確定認識してバッファをリセット
- 出力は {"kind": "partial"|"final", "text": str} を out_queue へ
"""
import queue
import threading
import time

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

from .audio_capture import TARGET_RATE
from .config import Config
from .text_cleanup import cleanup_asr_text


class StreamingTranscriber(threading.Thread):
    def __init__(self, cfg: Config, audio_queue: queue.Queue, out_queue: queue.Queue,
                 stop_event: threading.Event, speaker_detector=None, dictionary=None,
                 model=None, model_lock=None, source_language: str | None = None,
                 name: str = "asr", emit_partials: bool = True,
                 use_hotwords: bool = True,
                 min_segment_rms: float | None = None):
        super().__init__(daemon=True, name=name)
        self.cfg = cfg
        self.audio_queue = audio_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.speaker_detector = speaker_detector
        self.dictionary = dictionary
        self.model = model if model is not None else self._load_model()
        self.model_lock = model_lock if model_lock is not None else threading.Lock()
        self.source_language = source_language or cfg.source_language
        self.emit_partials = emit_partials
        self.use_hotwords = use_hotwords
        self.min_segment_rms = min_segment_rms
        # speech_pad_ms 既定値(400ms)は発話間のポーズを潰して文単位の確定を
        # 妨げるので短くする(セグメント切り出し時に自前で前パディングする)
        self._vad_options = VadOptions(min_silence_duration_ms=250, speech_pad_ms=100)

    def _load_model(self) -> WhisperModel:
        cfg = self.cfg
        try:
            model = WhisperModel(cfg.whisper_model, device=cfg.device,
                                 compute_type=cfg.whisper_compute_type)
            print(f"[asr] Whisper '{cfg.whisper_model}' を {cfg.device} でロードしました")
            return model
        except Exception as exc:
            if cfg.device == "cuda":
                print(f"[asr] CUDA でのロードに失敗、CPU にフォールバックします: {exc}")
                return WhisperModel(cfg.whisper_model, device="cpu", compute_type="int8")
            raise

    def run(self):
        cfg = self.cfg
        buffer = np.zeros(0, dtype=np.float32)
        last_partial_at = 0.0
        last_partial_len = 0
        last_audio_at = time.monotonic()
        last_speaker_check = 0.0
        silence_block = np.zeros(TARGET_RATE // 10, dtype=np.float32)

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                chunk = None
            if chunk is not None:
                last_audio_at = time.monotonic()
                buffer = np.concatenate([buffer, chunk])
                while True:  # 溜まっている分は一気に取り込む
                    try:
                        buffer = np.concatenate([buffer, self.audio_queue.get_nowait()])
                    except queue.Empty:
                        break
            elif len(buffer) > 0 and time.monotonic() - last_audio_at > 0.3:
                # WASAPI loopback は再生が止まるとデータが来なくなるので
                # 無音を合成して VAD の無音判定を進める
                buffer = np.concatenate([buffer, silence_block])

            if len(buffer) < int(0.5 * TARGET_RATE):
                continue

            speech = get_speech_timestamps(buffer, self._vad_options)
            if not speech:
                # 発話なし: 直近 1 秒だけ残して捨てる(発話頭の欠落防止)
                buffer = buffer[-TARGET_RATE:]
                last_partial_at = 0.0
                last_partial_len = 0
                continue

            silence_samples = int(cfg.silence_finalize * TARGET_RATE)
            speech_start = speech[0]["start"]

            # 確定位置を探す: 発話間のポーズ / 話者交代 / 末尾の無音 / 最大発話長の超過
            split_end = None
            check_speaker = (self.speaker_detector is not None
                             and time.monotonic() - last_speaker_check >= 0.5)
            for prev, nxt in zip(speech, speech[1:]):
                if nxt["start"] - prev["end"] >= silence_samples:
                    split_end = prev["end"]
                    break
                # ポーズが短くても声紋が変わっていれば話者交代として確定する
                # (埋め込み計算は CPU ~50ms×2 なのでサイクルを 0.5 秒に 1 回に間引く)
                if check_speaker:
                    last_speaker_check = time.monotonic()
                    before = buffer[speech_start:prev["end"]]
                    after = buffer[nxt["start"]:speech[-1]["end"]]
                    speaker_changed = self.speaker_detector.is_boundary(before, after)
                    check_speaker = False
                    if speaker_changed:
                        split_end = prev["end"]
                        break
            if split_end is None:
                trailing_silence = len(buffer) - speech[-1]["end"]
                speech_sec = (speech[-1]["end"] - speech_start) / TARGET_RATE
                if trailing_silence >= silence_samples or speech_sec >= cfg.max_utterance:
                    split_end = speech[-1]["end"]

            if split_end is not None:
                segment = buffer[max(0, speech_start - TARGET_RATE // 4):split_end]
                buffer = buffer[split_end:]
                last_partial_len = 0
                last_partial_at = 0.0
                voiced_sec = self._speech_duration(speech, speech_start, split_end)
                if not self._is_usable_segment(segment, voiced_sec, "final"):
                    self.out_queue.put({"kind": "final", "text": "",
                                        "speaker_change": False})
                    continue
                text = self._transcribe(segment, beam_size=cfg.final_beam_size, kind="final")
                speaker_change = False
                # 幻覚破棄したセグメント(音楽等)では声紋の基準を更新しない
                if text and self.speaker_detector is not None:
                    speaker_change = self.speaker_detector.is_change(segment)
                # text が空(幻覚破棄)でも流し、画面のグレー字幕をクリアさせる
                self.out_queue.put({"kind": "final", "text": text,
                                    "speaker_change": speaker_change})
            elif self.emit_partials and time.monotonic() - last_partial_at >= cfg.partial_interval:
                speech_end = speech[-1]["end"]
                voiced_sec = self._speech_duration(speech, speech_start, speech_end)
                segment = buffer[max(0, speech_start - TARGET_RATE // 4):]
                if not self._is_usable_segment(segment, voiced_sec, "partial"):
                    continue
                # 前回の暫定認識から音声が伸びていなければスキップ
                if speech_end <= last_partial_len:
                    continue
                last_partial_at = time.monotonic()
                last_partial_len = speech_end
                speaker_change = False
                if self.speaker_detector is not None:
                    speaker_change = self.speaker_detector.peek_change(segment)
                text = self._transcribe(segment, beam_size=cfg.partial_beam_size, kind="partial")
                self.out_queue.put({"kind": "partial", "text": text,
                                    "speaker_change": speaker_change})
        self._flush_remaining_buffer(buffer)

    def _transcribe(self, audio: np.ndarray, beam_size: int, kind: str) -> str:
        start = time.perf_counter()
        hotwords = (self.dictionary.hotwords(self.source_language)
                    if self.dictionary is not None and self.use_hotwords else None)
        with self.model_lock:
            segments, _ = self.model.transcribe(
                audio,
                language=self.source_language,
                beam_size=beam_size,
                # 圧縮率チェックに失敗したセグメントは温度を上げて再試行させる
                temperature=[0.0, 0.2, 0.4],
                condition_on_previous_text=False,
                without_timestamps=True,
                hotwords=hotwords,
            )
        text = "".join(seg.text for seg in segments).strip()
        cleaned = self._cleanup_text(text)
        if cleaned != text and self.cfg.log_latency:
            print(f"[asr] 幻覚候補を{'除去' if not cleaned else '補正'}: {text[:60]}")
        text = cleaned
        if self.dictionary is not None:
            before_dict = text
            text = self.dictionary.apply(text, self.source_language)
            cleaned = self._cleanup_text(text)
            if cleaned != text and self.cfg.log_latency:
                print(f"[asr] 辞書適用後の幻覚候補を{'除去' if not cleaned else '補正'}: {text[:60]}")
            text = cleaned
            if before_dict != text and not text:
                last = before_dict[:60]
                if self.cfg.log_latency and last:
                    print(f"[asr] 辞書適用後に空になりました: {last}")
        if self.cfg.log_latency:
            elapsed = (time.perf_counter() - start) * 1000
            audio_sec = len(audio) / TARGET_RATE
            print(f"[asr] {kind} {audio_sec:.1f}s -> {elapsed:.0f}ms: {text[:60]}")
        return text

    def _flush_remaining_buffer(self, buffer: np.ndarray):
        """終了時に無音確定を待っていた最後の発話を final として流す。"""
        if len(buffer) < int(0.3 * TARGET_RATE):
            return
        speech = get_speech_timestamps(buffer, self._vad_options)
        if not speech:
            return
        speech_start = speech[0]["start"]
        speech_end = speech[-1]["end"]
        segment = buffer[max(0, speech_start - TARGET_RATE // 4):speech_end]
        voiced_sec = self._speech_duration(speech, speech_start, speech_end)
        if not self._is_usable_segment(segment, voiced_sec, "final"):
            return
        text = self._transcribe(segment, beam_size=self.cfg.final_beam_size, kind="final")
        speaker_change = False
        if text and self.speaker_detector is not None:
            speaker_change = self.speaker_detector.is_change(segment)
        self.out_queue.put({"kind": "final", "text": text,
                            "speaker_change": speaker_change})

    def _speech_duration(self, speech: list[dict], start: int, end: int) -> float:
        samples = 0
        for item in speech:
            overlap_start = max(start, item["start"])
            overlap_end = min(end, item["end"])
            if overlap_end > overlap_start:
                samples += overlap_end - overlap_start
        return samples / TARGET_RATE

    def _segment_rms_threshold(self) -> float:
        if self.min_segment_rms is not None:
            return self.min_segment_rms
        return self.cfg.asr_min_segment_rms

    def _is_usable_segment(self, audio: np.ndarray, voiced_sec: float, kind: str) -> bool:
        if voiced_sec < self.cfg.min_speech:
            if self.cfg.log_latency and kind == "final":
                print(f"[asr] 短すぎる発話を破棄: voiced={voiced_sec:.2f}s")
            return False
        if audio.size == 0:
            return False

        peak_threshold = self.cfg.asr_min_segment_peak
        if peak_threshold > 0:
            peak = float(np.max(np.abs(audio)))
            if peak < peak_threshold:
                if self.cfg.log_latency and kind == "final":
                    print(f"[asr] ほぼ無音の発話を破棄: peak={peak:.4f}")
                return False

        rms_threshold = self._segment_rms_threshold()
        if rms_threshold > 0:
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
            if rms < rms_threshold:
                if self.cfg.log_latency and kind == "final":
                    print(f"[asr] 低音量の発話を破棄: rms={rms:.4f}")
                return False
        return True

    def _cleanup_text(self, text: str) -> str:
        return cleanup_asr_text(
            text,
            self.source_language,
            self.cfg.asr_suppressed_phrases,
            self.cfg.english_asr_reject_cjk,
            self.cfg.asr_suppressed_substring_max_chars,
        )
