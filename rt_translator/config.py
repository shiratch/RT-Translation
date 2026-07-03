"""アプリ設定。config.json (カレントディレクトリ) があれば同名キーで上書きされる。"""
import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class Config:
    # --- ASR ---
    whisper_model: str = "large-v3-turbo"
    device: str = "cuda"                # "cuda" / "cpu"(自動フォールバックあり)
    whisper_compute_type: str = "int8_float16"
    source_language: str = "en"
    partial_interval: float = 0.8       # 発話中に暫定認識を回す間隔 [秒]
    silence_finalize: float = 0.6       # この長さの無音でセグメント確定 [秒]
    max_utterance: float = 18.0         # 強制確定までの最大発話長 [秒]
    min_speech: float = 0.25            # これ未満の発話は無視 [秒]
    partial_beam_size: int = 1
    final_beam_size: int = 1

    # --- 翻訳 ---
    nllb_repo: str = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
    nllb_compute_type: str = "int8_float16"
    target_lang: str = "jpn_Jpan"
    source_lang_nllb: str = "eng_Latn"

    # --- オーバーレイ表示 ---
    font_family: str = "Yu Gothic UI"
    font_size: int = 22
    overlay_alpha: float = 0.85
    overlay_width_ratio: float = 0.6    # 画面幅に対する字幕帯の幅
    final_lines: int = 2                # 確定字幕を何行(何発話)残すか
    show_source: bool = False           # 英語原文も小さく表示するか

    # --- その他 ---
    log_latency: bool = True            # 各段の処理時間をコンソールに出す


def load_config() -> Config:
    cfg = Config()
    path = Path("config.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                print(f"[config] 未知のキーを無視: {key}")
    return cfg
