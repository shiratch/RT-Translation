"""日英切り替え、encoder再利用、表示・保存の回帰テスト。"""
from dataclasses import make_dataclass
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from faster_whisper import WhisperModel

from rt_translator.asr import StreamingTranscriber
from rt_translator.bilingual_whisper import BilingualWhisperModel
from rt_translator.config import Config
from rt_translator.transcript import TranscriptWriter
from rt_translator.translator import TranslationWorker


class BilingualRecognitionTests(unittest.TestCase):
    def test_one_encoder_output_is_used_for_detection_and_recognition(self):
        options_type = make_dataclass("Options", ["multilingual", "hotwords"])
        model = BilingualWhisperModel.__new__(BilingualWhisperModel)
        model.feature_extractor = SimpleNamespace(nb_max_frames=3000)
        encoded = object()
        model.encode = Mock(return_value=encoded)
        # 対象外言語が1位でも、日英の確率を比較する。
        model.model = SimpleNamespace(detect_language=Mock(return_value=[[
            ("<|zh|>", .50), ("<|ja|>", .40), ("<|en|>", .10)]]))
        tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(token_to_id=lambda token: token))
        options = options_type(True, "English hotword")

        def generate(_model, features, selected, selected_options, progress, encoder):
            self.assertIs(encoder, encoded)
            self.assertEqual(selected.language_code, "ja")
            self.assertEqual(selected.language, "<|ja|>")
            self.assertFalse(selected_options.multilingual)
            self.assertIsNone(selected_options.hotwords)
            yield SimpleNamespace(text="確認します。")

        with patch.object(WhisperModel, "generate_segments", generate):
            result = list(model.generate_segments(np.zeros((128, 100)), tokenizer, options, False))
        self.assertEqual(result[0].language, "ja")
        model.encode.assert_called_once()
        model.model.detect_language.assert_called_once_with(encoded)
        self.assertTrue(options.multilingual)  # 呼び出し間で設定を変更しない

    def test_fixed_language_does_not_run_detection(self):
        model = BilingualWhisperModel.__new__(BilingualWhisperModel)
        model.encode = Mock(side_effect=AssertionError("extra encoder"))
        segment = SimpleNamespace(text="日本語")
        with patch.object(WhisperModel, "generate_segments", return_value=iter([segment])):
            self.assertEqual(list(model.generate_segments(
                None, None, SimpleNamespace(multilingual=False), False)), [segment])
        model.encode.assert_not_called()

    def test_detected_language_controls_cleanup_and_dictionary(self):
        model = Mock()
        model.transcribe.return_value = (iter([SimpleNamespace(
            text="予算を確認します。", language="ja")]), None)
        dictionary = Mock()
        dictionary.apply.side_effect = lambda text, language: text
        worker = StreamingTranscriber(
            Config(asr_bilingual=True, log_latency=False), queue.Queue(), queue.Queue(),
            threading.Event(), model=model, dictionary=dictionary)
        self.assertEqual(worker._transcribe(np.ones(16000), 1, "partial"), "予算を確認します。")
        self.assertEqual(worker.detected_language, "ja")
        self.assertTrue(model.transcribe.call_args.kwargs["multilingual"])
        dictionary.apply.assert_called_once_with("予算を確認します。", "ja")

    def test_japanese_microphone_stays_fixed_even_with_bilingual_profile(self):
        model = Mock()
        model.transcribe.return_value = (iter([SimpleNamespace(text="確認します。")]), None)
        worker = StreamingTranscriber(
            Config(asr_bilingual=True, log_latency=False), queue.Queue(), queue.Queue(),
            threading.Event(), model=model, source_language="ja", emit_partials=False)
        worker._transcribe(np.ones(16000), 5, "final")
        self.assertFalse(model.transcribe.call_args.kwargs["multilingual"])
        self.assertEqual(model.transcribe.call_args.kwargs["language"], "ja")
        self.assertEqual(worker.detected_language, "ja")

    def test_final_flush_preserves_detected_language_in_queue(self):
        model = Mock()
        model.transcribe.return_value = (iter([SimpleNamespace(
            text="確認します。", language="ja")]), None)
        out = queue.Queue()
        worker = StreamingTranscriber(Config(asr_bilingual=True, log_latency=False),
                                      queue.Queue(), out, threading.Event(), model=model)
        with patch("rt_translator.asr.get_speech_timestamps", return_value=[{"start": 0, "end": 16000}]):
            worker._flush_remaining_buffer(np.ones(16000))
        self.assertEqual(out.get_nowait()["language"], "ja")


class BilingualRoutingTests(unittest.TestCase):
    def worker(self, **kwargs):
        engine = Mock()
        engine.translate.side_effect = lambda text, **kw: "訳：" + text
        cfg = Config(source_language="en", log_latency=False, translation_meeting_fixes=False,
                     translation_normalize_punctuation=False, **kwargs)
        worker = TranslationWorker(cfg, engine, queue.Queue(), queue.Queue(), threading.Event())
        return worker, engine

    def test_japanese_partial_and_final_bypass_translation_without_fragment_wait(self):
        worker, engine = self.worker(translation_stream_preview=True)
        worker._translate_one({"kind": "partial", "text": "確認します", "language": "ja"})
        worker._handle_final({"kind": "final", "text": "確認します", "language": "ja"})
        results = [worker.ui_queue.get_nowait(), worker.ui_queue.get_nowait()]
        self.assertEqual([r["ja"] for r in results], ["確認します", "確認します"])
        self.assertEqual([r["en"] for r in results], ["", ""])
        engine.translate.assert_not_called()
        engine.translate_stream.assert_not_called()
        self.assertFalse(worker._final_buffer)

    def test_pending_english_flushes_before_japanese_then_english_resumes(self):
        worker, engine = self.worker()
        worker._handle_final({"kind": "final", "text": "Please check", "language": "en"})
        self.assertTrue(worker.ui_queue.empty())
        worker._translate_one({"kind": "partial", "text": "確認します", "language": "ja"})
        worker._handle_final({"kind": "final", "text": "確認します。", "language": "ja"})
        worker._handle_final({"kind": "final", "text": "Thank you.", "language": "en"})
        results = [worker.ui_queue.get_nowait() for _ in range(4)]
        self.assertEqual([r["kind"] for r in results], ["final", "partial", "final", "final"])
        self.assertEqual([r["en"] for r in results], ["Please check", "", "", "Thank you."])
        self.assertEqual([c.args[0] for c in engine.translate.call_args_list],
                         ["Please check", "Thank you."])

    def test_japanese_final_also_flushes_english_fragment(self):
        worker, engine = self.worker()
        worker._handle_final({"kind": "final", "text": "Next item", "language": "en"})
        worker._handle_final({"kind": "final", "text": "次の項目です。", "language": "ja"})
        self.assertEqual(worker.ui_queue.get_nowait()["en"], "Next item")
        self.assertEqual(worker.ui_queue.get_nowait()["ja"], "次の項目です。")
        self.assertFalse(worker._final_buffer)
        engine.translate.assert_called_once()

    def test_language_correction_clears_partial_cache_and_preview_length(self):
        worker, _ = self.worker()
        worker._sentence_cache["old"] = "old translation"
        worker._visible_partial_chars = 100
        worker._translate_one({"kind": "partial", "text": "確認", "language": "ja"})
        self.assertFalse(worker._sentence_cache)
        self.assertEqual(worker._visible_partial_chars, 2)
        worker._translate_one({"kind": "partial", "text": "Check", "language": "en"})
        self.assertEqual(worker._active_language, "en")

    def test_japanese_is_saved_as_original_not_english_or_translation(self):
        worker, _ = self.worker()
        lines = []
        with patch.object(TranscriptWriter, "_append", side_effect=lines.extend):
            worker.transcript_writer = TranscriptWriter(worker.cfg)
            worker._translate_one({"kind": "partial", "text": "未確定", "language": "ja"})
            worker._handle_final({"kind": "final", "text": "確定しました。", "language": "ja"})
        saved = "\n".join(lines)
        self.assertIn("JA: 確定しました。", saved)
        self.assertNotIn("EN_SOURCE:", saved)
        self.assertNotIn("JA_TRANSLATION:", saved)
        self.assertNotIn("未確定", saved)


if __name__ == "__main__":
    unittest.main()
