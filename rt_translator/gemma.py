"""TranslateGemma専用の翻訳テンプレート。"""
import json
from pathlib import Path

from .config import Config
from .llama_backend import LocalLlamaTranslator


class TranslateGemmaTranslator(LocalLlamaTranslator):
    def __init__(self, cfg: Config):
        if not cfg.gemma_model_path and cfg.gemma_model not in {"12b-q8_0", "27b-q6_k"}:
            raise ValueError("gemma_modelは12b-q8_0または27b-q6_kを指定してください。")
        super().__init__(
            cfg, model_key=cfg.gemma_model, model_path=cfg.gemma_model_path,
            server_path=cfg.gemma_server_path, context_size=cfg.gemma_context_size,
            startup_timeout=cfg.gemma_startup_timeout, request_timeout=cfg.gemma_request_timeout)

    def _server_template_args(self):
        return ["--chat-template-file",
                str(Path(__file__).parent / "templates" / "translategemma_en_ja.jinja"),
                "--chat-template-kwargs",
                json.dumps({"source_lang_code": "en", "target_lang_code": "ja"})]

    @staticmethod
    def _translation_payload(text, max_decoding_length, stream):
        return {
            "messages": [{"role": "user", "content": text.strip()}],
            "chat_template_kwargs": {"source_lang_code": "en", "target_lang_code": "ja"},
            "temperature": 0, "max_tokens": max_decoding_length,
            "stream": stream, "cache_prompt": True,
        }
