"""音声機器・モデルのダウンロードなしで実行できる回帰テスト。"""
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rt_translator.asr import StreamingTranscriber
from rt_translator.config import Config, load_config
from rt_translator.text_cleanup import (apply_translation_fixes,
                                       load_translation_fixes,
                                       normalize_japanese_punctuation,
                                       repair_translation_numbers)
from rt_translator.translator import TranslationWorker


class TextIntegrityTests(unittest.TestCase):
    def test_multiple_amounts_are_replaced_once_each(self):
        self.assertEqual(
            repair_translation_numbers("予算は130万ドル、費用は200万ドルです。",
                                       "Budget is $13 million; cost is $2 million.", "en"),
            "予算は1300万ドル、費用は200万ドルです。")

    def test_cents_are_not_rounded_away(self):
        self.assertEqual(repair_translation_numbers("価格は12ドルです。",
                                                   "It costs $12.50.", "en"),
                         "価格は12.50ドルです。")

    def test_ambiguous_amount_count_is_not_guessed(self):
        self.assertEqual(repair_translation_numbers("合計は30ドルです。",
                                                   "$10 plus $20", "en"),
                         "合計は30ドルです。")

    def test_meeting_fixes_are_disabled_by_default(self):
        cfg = Config()
        self.assertFalse(cfg.translation_meeting_fixes)
        self.assertFalse(cfg.translation_repair_numbers)

    def test_translation_fixes_do_not_rewrite_tone(self):
        for ja in ("大丈夫だった。", "そうだったんです。", "問題ないですよね。", "分かってるの？"):
            self.assertEqual(apply_translation_fixes(ja, "It was fine.", "en", []), ja)

    def test_translation_fixes_file_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixes.txt"
            path.write_text("# comment\n林檎=りんご\napple,pie | 円周率=パイ\n"
                            "any questions | *=質問はありますか？\n", encoding="utf-8")
            fixes = load_translation_fixes(str(path))
        self.assertEqual(apply_translation_fixes("林檎の円周率", "Apple Pie", "en", fixes),
                         "りんごのパイ")
        # 原文に条件語がそろわなければ条件付きルールは使わない
        self.assertEqual(apply_translation_fixes("円周率", "pie chart", "en", fixes), "円周率")
        self.assertEqual(apply_translation_fixes("何か質問は", "Any questions?", "en", fixes),
                         "質問はありますか？")
        self.assertEqual(load_translation_fixes(str(Path(tmp) / "missing.txt")), [])

    def test_punctuation_preserves_thousands_and_decimals(self):
        self.assertEqual(normalize_japanese_punctuation("価格は1,234.50ドル, 確認?", True),
                         "価格は1,234.50ドル、確認？")


class QueueOrderingTests(unittest.TestCase):
    def render(self, entries):
        incoming, outgoing = queue.Queue(), queue.Queue()
        for kind, text in entries:
            incoming.put({"kind": kind, "text": text})
        stop = threading.Event()
        stop.set()
        cfg = Config(source_language="ja", log_latency=False, shutdown_drain_seconds=0)
        worker = TranslationWorker(cfg, None, incoming, outgoing, stop)
        worker.run()
        result = []
        while not outgoing.empty():
            item = outgoing.get_nowait()
            result.append((item["kind"], item["ja"]))
        return result

    def test_partial_before_final_does_not_reappear(self):
        self.assertEqual(self.render([("partial", "古い字幕"), ("final", "確定字幕")]),
                         [("final", "確定字幕")])

    def test_only_partial_after_last_final_survives(self):
        entries = [("partial", "古い字幕"), ("final", "確定一"),
                   ("partial", "途中"), ("final", "確定二"),
                   ("partial", "次の発話"), ("partial", "次の発話の続き")]
        self.assertEqual(self.render(entries), [("final", "確定一"), ("final", "確定二"),
                                               ("partial", "次の発話の続き")])

    def test_empty_final_also_invalidates_partial(self):
        self.assertEqual(self.render([("partial", "破棄すべき字幕"), ("final", "")]),
                         [("final", "")])


class LazyInferenceTests(unittest.TestCase):
    def test_fast_partials_skip_temperature_retries_but_finals_keep_them(self):
        temperatures = []

        class Model:
            def transcribe(self, *args, **kwargs):
                temperatures.append(kwargs["temperature"])
                return iter([SimpleNamespace(text="Hello world.")]), None

        cfg = Config(log_latency=False, partial_temperature_fallback=False)
        worker = StreamingTranscriber(cfg, queue.Queue(), queue.Queue(), threading.Event(), model=Model())
        for kind in ("partial", "final"):
            worker._transcribe(np.ones(16000, dtype=np.float32), 1, kind)
        self.assertEqual(temperatures, [[0.0], [0.0, 0.2, 0.4]])

    def test_inference_slot_is_held_while_consuming_generator(self):
        lock = threading.Lock()
        observed = []

        class Model:
            def transcribe(self, *args, **kwargs):
                def segments():
                    observed.append(lock.locked())
                    yield SimpleNamespace(text="Hello world.")
                return segments(), None

        worker = StreamingTranscriber(Config(log_latency=False), queue.Queue(), queue.Queue(),
                                      threading.Event(), model=Model(), model_lock=lock)
        self.assertEqual(worker._transcribe(np.ones(16000, dtype=np.float32), 1, "final"),
                         "Hello world.")
        self.assertEqual(observed, [True])
        self.assertFalse(lock.locked())


class ProfileTests(unittest.TestCase):
    def test_profile_overrides_personal_config_but_mode_still_wins(self):
        original_read = Path.read_text
        original_exists = Path.exists
        personal = {"source_language": "ja", "font_size": 27, "final_beam_size": 1}

        def read(path, *args, **kwargs):
            return (json.dumps(personal) if path == Path("config.json")
                    else original_read(path, *args, **kwargs))

        def exists(path):
            return path == Path("config.json") or original_exists(path)

        with patch.object(Path, "read_text", read), patch.object(Path, "exists", exists), patch.dict(os.environ, {
            "RT_TRANSLATOR_MODE": "en", "VOICE_STREAM_SILENCE_THRESHOLD": "",
        }):
            cfg = load_config("rtx5090")
            self.assertEqual(cfg.source_language, "en")
            self.assertEqual(cfg.font_size, 27)
            self.assertEqual(cfg.final_beam_size, 5)
            self.assertEqual(cfg.whisper_num_workers, 2)


if __name__ == "__main__":
    unittest.main()
