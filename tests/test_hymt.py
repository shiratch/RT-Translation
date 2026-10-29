"""Hy-MT2のモデル選択・公式指示文・共有ストリーム接続を検証する。"""
import io
import json
import os
import threading
import unittest
from unittest.mock import Mock, patch

from rt_translator.config import Config, load_config
from rt_translator.gemma_setup import ensure_model
from rt_translator.gemma import TranslateGemmaTranslator
from rt_translator.hymt import HyMT2Translator
from rt_translator.translator import create_translator


class HyMTTests(unittest.TestCase):
    def test_hymt_model_cannot_accidentally_use_gemma_template(self):
        with self.assertRaises(ValueError):
            TranslateGemmaTranslator(Config(gemma_model='hymt2-30b-a3b-q4_k_m'))

    def engine(self):
        engine = HyMT2Translator.__new__(HyMT2Translator)
        engine.cfg = Config()
        engine._lock = threading.Lock()
        engine._process = Mock()
        engine._process.poll.return_value = None
        engine.last_timings = {}
        return engine

    def test_prompt_keeps_whole_source_and_uses_japanese_translation_instruction(self):
        engine = self.engine()
        source = 'Keep "Facebook". The budget is $13 million.'
        payload = engine._translation_payload(source, 256, True)
        self.assertEqual(payload['messages'], [{'role': 'user', 'content':
            'Translate the following text into Japanese. Note that you should only '
            'output the translated result without any additional explanation:\n' + source}])
        self.assertNotIn('chat_template_kwargs', payload)
        self.assertEqual(engine._server_template_args(), [])
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['max_tokens'], 256)
        engine.cfg.hymt_temperature = 0.7
        self.assertEqual(engine._translation_payload(source, 256, False)['temperature'], 0.7)

    def test_stream_uses_shared_completion_validation_and_closes_response(self):
        engine = self.engine()
        data = [
            {'choices': [{'delta': {'content': '翻訳結果です。'}, 'finish_reason': None}]},
            {'choices': [{'delta': {}, 'finish_reason': 'length'}]},
        ]
        response = io.BytesIO((''.join('data: ' + json.dumps(v) + '\n\n' for v in data)
                               + 'data: [DONE]\n\n').encode('utf-8'))
        engine._open_request = Mock(return_value=response)
        result = engine.translate_stream('Source')
        self.assertEqual(next(result), '翻訳結果です。')
        with self.assertRaises(RuntimeError):
            next(result)
        self.assertTrue(response.closed)

    def test_model_download_is_pinned_to_official_tencent_q4(self):
        with patch('huggingface_hub.hf_hub_download', return_value='model.gguf') as download:
            ensure_model('hymt2-30b-a3b-q4_k_m')
        args = download.call_args.kwargs
        self.assertEqual(args['repo_id'], 'tencent/Hy-MT2-30B-A3B-GGUF')
        self.assertEqual(args['filename'], 'Hy-MT2-30B-A3B-Q4_K_M.gguf')
        self.assertEqual(args['revision'], 'fd3dcbb6b31e9e03923ef4f9f42500f74c3851c3')

    def test_profile_and_factory_choose_hymt_with_fast_previews(self):
        with patch.dict(os.environ, {'RT_TRANSLATOR_MODE': 'en'}):
            cfg = load_config('hymt2')
        self.assertEqual(cfg.translation_backend, 'hymt2')
        self.assertTrue(cfg.translation_stream_preview)
        self.assertTrue(cfg.translation_partial_full_context)
        self.assertEqual(cfg.final_beam_size, 5)
        with patch('rt_translator.hymt.HyMT2Translator') as backend:
            self.assertIs(create_translator(cfg), backend.return_value)
            backend.assert_called_once_with(cfg)


if __name__ == '__main__':
    unittest.main()
