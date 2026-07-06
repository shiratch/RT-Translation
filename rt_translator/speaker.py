"""話者交代の検出。

確定セグメントごとに話者埋め込み(声紋ベクトル)を sherpa-onnx +
NeMo TitaNet-small(CPU, ~50ms)で計算し、直前セグメントとのコサイン類似度が
閾値を下回ったら「話者が変わった」と判定する。

「誰が話しているか」の特定(話者分離)ではなく変化点の検出のみ。
短すぎるセグメントは判定が不安定なので「同一話者」として扱う。
"""
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from .config import Config

MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/nemo_en_titanet_small.onnx")
SAMPLE_RATE = 16000


def _model_path() -> Path:
    path = Path.home() / ".cache" / "rt_translator" / "nemo_en_titanet_small.onnx"
    if not path.exists():
        print("[spk] 話者埋め込みモデルをダウンロード中 (40MB)...")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
                with tmp.open("wb") as f:
                    shutil.copyfileobj(response, f)
            tmp.replace(path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
    return path


class SpeakerChangeDetector:
    def __init__(self, cfg: Config):
        import sherpa_onnx
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(_model_path()), num_threads=2)
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self._threshold = cfg.speaker_change_threshold
        self._min_samples = int(cfg.speaker_min_speech * SAMPLE_RATE)
        self._log = cfg.log_latency
        self._prev: np.ndarray | None = None
        print("[spk] 話者交代検出を有効化しました")

    def _embed(self, audio: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        emb = np.array(self._extractor.compute(stream), dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-8)

    def is_boundary(self, before: np.ndarray, after: np.ndarray) -> bool:
        """発話中の小さなポーズの前後で話者が変わったかを判定する。
        (無音を待たずに文を確定するための判定。状態は更新しない)"""
        if len(before) < self._min_samples or len(after) < self._min_samples:
            return False
        similarity = float(np.dot(self._embed(before), self._embed(after)))
        changed = similarity < self._threshold
        if changed and self._log:
            print(f"[spk] 発話中に話者交代を検出 (similarity={similarity:.2f})")
        return changed

    def peek_change(self, audio: np.ndarray) -> bool:
        """暫定(認識途中)セグメント用: 直前の確定話者と比較するだけで、
        基準となる話者ベクトルは更新しない。"""
        if self._prev is None or len(audio) < self._min_samples:
            return False
        emb = self._embed(audio)
        return float(np.dot(self._prev, emb)) < self._threshold

    def is_change(self, audio: np.ndarray) -> bool:
        """確定セグメントの音声を受け取り、直前と話者が変わったかを返す。"""
        if len(audio) < self._min_samples:
            return False  # 短い相槌等は判定不能 -> 同一話者扱い
        emb = self._embed(audio)
        prev = self._prev
        self._prev = emb
        if prev is None:
            return False
        similarity = float(np.dot(prev, emb))
        changed = similarity < self._threshold
        if self._log:
            print(f"[spk] similarity={similarity:.2f} -> "
                  f"{'話者交代' if changed else '同一話者'}")
        return changed
