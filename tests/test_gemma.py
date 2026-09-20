"""モデルをロードせず、Gemmaの接続・翻訳失敗・後処理の境界を検証する。"""
import io
import json
import os
import queue
import threading
import unittest
from unittest.mock import Mock, patch

from rt_translator.config import Config, load_config
from rt_translator.gemma import TranslateGemmaTranslator
from rt_translator.gemma_setup import ensure_model, get_model_spec
from rt_translator.translator import TranslationWorker, create_translator


def fake_engine(result):
    engine = TranslateGemmaTranslator.__new__(TranslateGemmaTranslator)
    engine._lock = threading.Lock()
    engine._process = Mock()
    engine._process.poll.return_value = None
    engine._request = Mock(return_value=result)
    engine._log = None
    engine.last_timings = {}
    return engine


class GemmaModelSelectionTests(unittest.TestCase):
    def test_launch_profiles_choose_their_model_and_keep_fast_previews(self):
        with patch.dict(os.environ, {"RT_TRANSLATOR_MODE": "en"}):
            for profile, model in (("translategemma12b", "12b-q8_0"),
                                   ("translategemma27b", "27b-q6_k")):
                cfg = load_config(profile)
                self.assertEqual(cfg.gemma_model, model)
                self.assertEqual(cfg.source_language, "en")
                self.assertTrue(cfg.translation_stream_preview)
                self.assertEqual(cfg.final_beam_size, 5)

    def test_model_selection_uses_separate_repositories_and_pinned_revisions(self):
        with patch("huggingface_hub.hf_hub_download", return_value="model.gguf") as download:
            ensure_model()
            first = download.call_args.kwargs
            ensure_model("27b-q6_k")
            second = download.call_args.kwargs
        self.assertEqual(first["repo_id"], "mradermacher/translategemma-12b-it-GGUF")
        self.assertEqual(first["filename"], "translategemma-12b-it.Q8_0.gguf")
        self.assertEqual(second["repo_id"], "mradermacher/translategemma-27b-it-GGUF")
        self.assertEqual(second["filename"], "translategemma-27b-it.Q6_K.gguf")
        for call in (first, second):
            self.assertRegex(call["revision"], r"^[a-f0-9]{40}$")
            self.assertTrue(call["local_files_only"])

    def test_unknown_model_does_not_download_a_default_model(self):
        with patch("huggingface_hub.hf_hub_download") as download:
            with self.assertRaises(ValueError):
                ensure_model("27b-typo")
            download.assert_not_called()

    def test_runtime_reports_selected_quantization(self):
        engine = fake_engine(None)
        spec = get_model_spec("27b-q6_k")
        engine.cfg = Config(gemma_model="27b-q6_k")
        engine._compute_type, engine._model_label = spec.compute_type, spec.label
        engine.model_path = spec.filename
        engine._offloaded_layers = 63
        self.assertEqual(engine.runtime_info()["compute_type"], "Q6_K")
        self.assertEqual(engine.runtime_info()["model"], "TranslateGemma 27B Q6_K")


class GemmaProtocolTests(unittest.TestCase):
    def response(self, content="これは訳文です。", reason="stop"):
        return {"choices": [{"finish_reason": reason, "message": {"content": content}}]}

    def test_uses_translation_template_and_keeps_whole_input(self):
        engine = fake_engine(self.response())
        source = 'The budget is $13 million. Keep "<source>" in the text.'
        self.assertEqual(engine.translate(source, beam_size=5, max_unit_chars=10), "これは訳文です。")
        endpoint, payload = engine._request.call_args.args
        self.assertEqual(endpoint, "/v1/chat/completions")
        self.assertEqual(payload["messages"], [{"role": "user", "content": source}])
        self.assertEqual(payload["chat_template_kwargs"],
                         {"source_lang_code": "en", "target_lang_code": "ja"})
        self.assertEqual(payload["temperature"], 0)

    def test_truncated_output_is_not_published_as_complete(self):
        engine = fake_engine(self.response("途中までの訳", "length"))
        with self.assertRaises(RuntimeError):
            engine.translate("Source")

    def test_empty_response_is_an_error(self):
        with self.assertRaises(RuntimeError):
            fake_engine(self.response("")).translate("Source")

    def test_empty_source_makes_no_request(self):
        engine = fake_engine(self.response())
        self.assertEqual(engine.translate("  "), "")
        engine._request.assert_not_called()

    def test_stopped_server_is_detected_before_sending_text(self):
        engine = fake_engine(self.response())
        engine._process.poll.return_value = 1
        with self.assertRaises(RuntimeError):
            engine.translate("Source")
        engine._request.assert_not_called()

    def test_close_is_idempotent_and_stops_owned_process(self):
        engine = fake_engine(self.response())
        child = engine._process
        engine.close()
        engine.close()
        child.terminate.assert_called_once()
        child.wait.assert_called_once()

    def test_unknown_backend_fails_instead_of_using_another_model(self):
        with self.assertRaises(ValueError):
            create_translator(Config(translation_backend="typo"))

    def stream_engine(self, reason="stop", done=True):
        chunks = [{"choices": [{"delta": {"content": part}, "finish_reason": None}]}
                  for part in ("これは", "訳文です。")]
        chunks.append({"choices": [{"delta": {}, "finish_reason": reason}],
                       "timings": {"predicted_ms": 120}})
        data = ": ping\r\n\r\n" + "".join(
            "data: " + json.dumps(chunk, ensure_ascii=False) + "\r\n\r\n"
            for chunk in chunks)
        if done:
            data += "data: [DONE]\r\n\r\n"
        response = io.BytesIO(data.encode("utf-8"))
        engine = fake_engine(None)
        engine._open_request = Mock(return_value=response)
        return engine, response

    def test_stream_emits_cumulative_utf8_before_finishing_and_closes_response(self):
        engine, response = self.stream_engine()
        parts = engine.translate_stream("Source", max_decoding_length=123)
        self.assertEqual(next(parts), "これは")
        self.assertFalse(response.closed)
        self.assertEqual(list(parts), ["これは訳文です。"])
        self.assertTrue(response.closed)
        payload = engine._open_request.call_args.args[1]
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["max_tokens"], 123)
        self.assertEqual(engine.last_timings, {"predicted_ms": 120})

    def test_stream_rejects_truncated_or_disconnected_completion(self):
        for reason, done in (("length", True), (None, True), ("stop", False)):
            with self.subTest(reason=reason, done=done):
                engine, response = self.stream_engine(reason, done)
                with self.assertRaises(RuntimeError):
                    list(engine.translate_stream("Source"))
                self.assertTrue(response.closed)

    def test_stream_error_event_is_reported_and_connection_is_closed(self):
        engine = fake_engine(None)
        response = io.BytesIO(b'data: {"error": {"message": "context exceeded"}}\n\n')
        engine._open_request = Mock(return_value=response)
        with self.assertRaisesRegex(RuntimeError, "context exceeded"):
            list(engine.translate_stream("Source"))
        self.assertTrue(response.closed)

    def test_sse_data_lines_form_one_event(self):
        response = io.BytesIO(b'data: {"choices":\ndata: []}\n\ndata: [DONE]\n\n')
        self.assertEqual(list(TranslateGemmaTranslator._stream_events(response)), [{"choices": []}])


class GemmaPipelineTests(unittest.TestCase):
    def worker(self, cfg, translator):
        return TranslationWorker(cfg, translator, queue.Queue(), queue.Queue(), threading.Event())

    def test_partial_passes_full_context_in_one_call(self):
        translator = Mock()
        translator.translate.return_value = "二文の翻訳。"
        worker = self.worker(Config(translation_partial_full_context=True), translator)
        source = "This is the first sentence. This is the second."
        self.assertEqual(worker._translate_partial(source), "二文の翻訳。")
        translator.translate.assert_called_once()
        self.assertEqual(translator.translate.call_args.args[0], source)

    def test_gemma_profile_does_not_rewrite_facebook_or_amounts(self):
        translator = Mock()
        translator.translate.return_value = "Facebookのスケジュールです。費用は12.50ドルです。"
        cfg = Config(translation_meeting_fixes=False, translation_repair_numbers=False,
                     log_latency=False)
        worker = self.worker(cfg, translator)
        worker._translate_one({"kind": "final", "text": "Facebook schedule. It costs $15."})
        result = worker.ui_queue.get_nowait()
        self.assertEqual(result["ja"], "Facebookのスケジュールです。費用は12.50ドルです。")

    def test_translation_error_preserves_source_in_transcript(self):
        translator = Mock()
        translator.translate.side_effect = TimeoutError("Timed out")
        worker = self.worker(Config(log_latency=False), translator)
        worker.transcript_writer = Mock()
        worker._translate_one({"kind": "final", "text": "Keep this source."})
        self.assertEqual(worker.transcript_writer.write_final.call_args.args[0], "Keep this source.")
        self.assertEqual(worker.ui_queue.get_nowait()["ja"], "")

    def test_preview_is_visible_before_final_and_transcript_only_saves_complete_text(self):
        translator = Mock()
        worker = self.worker(Config(translation_stream_preview=True, log_latency=False), translator)
        worker.transcript_writer = Mock()

        def generate(*args, **kwargs):
            yield "これは訳文"
            item = worker.ui_queue.get_nowait()
            self.assertEqual((item["kind"], item["ja"]), ("partial", "これは訳文"))
            worker.transcript_writer.write_final.assert_not_called()
            yield "これは訳文です。"

        translator.translate_stream.side_effect = generate
        worker._translate_one({"kind": "final", "text": "This is a translation."})
        items = list(worker.ui_queue.queue)
        self.assertEqual(items[-1]["kind"], "final")
        self.assertEqual(items[-1]["ja"], "これは訳文です。")
        worker.transcript_writer.write_final.assert_called_once()
        self.assertEqual(worker.transcript_writer.write_final.call_args.args[1], "これは訳文です。")
        translator.translate.assert_not_called()

    def test_failed_stream_clears_preview_and_preserves_source_without_incomplete_translation(self):
        translator = Mock()
        worker = self.worker(Config(translation_stream_preview=True, log_latency=False), translator)
        worker.transcript_writer = Mock()

        def generate(*args, **kwargs):
            yield "これは途中の訳"
            raise RuntimeError("stream interrupted")

        translator.translate_stream.side_effect = generate
        worker._translate_one({"kind": "final", "text": "Keep this source."})
        self.assertEqual([(i["kind"], i["ja"]) for i in worker.ui_queue.queue],
                         [("partial", "これは途中の訳"), ("final", "")])
        self.assertEqual(worker.transcript_writer.write_final.call_args.args[:2], ("Keep this source.", ""))

    def test_regenerated_preview_does_not_shrink_existing_subtitle_to_first_tokens(self):
        worker = self.worker(Config(log_latency=False), Mock())
        worker._emit_result({"kind": "partial", "ja": "すでに表示している字幕"})
        callback = worker._preview_callback({"text": "Updated source."}, 0)
        with patch("rt_translator.translator.time.perf_counter", side_effect=[1, 1.1, 1.2]):
            callback("新しい訳")
            self.assertEqual(worker.ui_queue.qsize(), 1)
            callback("すでに表示している字幕の続き")
            self.assertEqual(worker.ui_queue.qsize(), 2)
            worker._emit_result({"kind": "final", "ja": "確定しました"})
            callback("次の発話")
        self.assertEqual(worker.ui_queue.get_nowait()["ja"], "すでに表示している字幕")
        self.assertEqual(list(worker.ui_queue.queue)[-1]["ja"], "次の発話")


if __name__ == "__main__":
    unittest.main()
