"""Hy-MT2 30B-A3Bの英日翻訳。モデル付属chat templateを使用する。"""
from .config import Config
from .llama_backend import LocalLlamaTranslator


class HyMT2Translator(LocalLlamaTranslator):
    def __init__(self, cfg: Config):
        super().__init__(
            cfg, model_key="hymt2-30b-a3b-q4_k_m", model_path=cfg.hymt_model_path,
            server_path=cfg.hymt_server_path, context_size=cfg.hymt_context_size)

    def _translation_payload(self, text, max_decoding_length, stream):
        prompt = ("Translate the following text into Japanese. Note that you should only "
                  "output the translated result without any additional explanation:\n" + text.strip())
        return {
            "messages": [{"role": "user", "content": prompt}],
            # 公式推奨の0.7に対し、字幕の安定性・比較の再現性を優先して既定は0。
            "temperature": self.cfg.hymt_temperature,
            "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repeat_penalty": 1.0,
            "max_tokens": max_decoding_length, "stream": stream, "cache_prompt": True,
        }
