"""アプリ設定。config.json (カレントディレクトリ) があれば同名キーで上書きされる。"""
import dataclasses
import json
import os
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

    # --- 話者交代検出 ---
    speaker_change_detection: bool = True
    speaker_change_threshold: float = 0.45  # コサイン類似度がこれ未満なら話者交代
    speaker_min_speech: float = 1.0         # これ未満の発話は判定せず同一話者扱い [秒]

    # --- オーバーレイ表示 ---
    font_family: str = "Yu Gothic UI"
    font_size: int = 18
    overlay_alpha: float = 0.85
    overlay_width_ratio: float = 0.6    # 画面幅に対する字幕帯の幅
    final_lines: int = 7                # 画面に同時表示する最大行数(折り返し込み)
    show_source: bool = False           # 英語原文も小さく表示するか

    # --- 文字起こし保存 ---
    transcript_enabled: bool = True
    transcript_path: str = "transcripts/session_{timestamp}.txt"
    transcript_format: str = "both"     # "both" / "ja" / "en"
    remote_transcript_format: str = ""  # 空なら transcript_format と同じ
    mic_transcript_format: str = "ja"
    transcript_timestamps: bool = True
    transcript_source_labels: bool = True
    remote_transcript_label: str = "REMOTE"

    # --- マイク文字起こし保存 ---
    mic_transcript_enabled: bool = False
    mic_source_language: str = "ja"     # 自分のマイク音声は既定で日本語認識
    mic_device_name: str = ""           # 空なら既定の録音デバイス
    mic_transcript_label: str = "MIC"

    # --- その他 ---
    log_latency: bool = True            # 各段の処理時間をコンソールに出す
    user_dictionary: str = "user_dictionary.txt"  # ユーザー辞書(VoiceText と同書式)


def _apply_values(cfg: Config, data: dict, source: str):
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            print(f"[config] 未知のキーを無視({source}): {key}")


def _apply_launch_mode(cfg: Config):
    mode = os.environ.get("RT_TRANSLATOR_MODE", "").strip().lower()
    if not mode:
        return
    if mode in {"ja", "jp", "japanese"}:
        cfg.source_language = "ja"
        cfg.transcript_format = "ja"
        cfg.remote_transcript_format = "ja"
        cfg.mic_transcript_enabled = True
        cfg.mic_transcript_format = "ja"
        cfg.mic_source_language = "ja"
        print("[config] 起動モード: 日本語文字起こし")
    elif mode in {"en", "english"}:
        cfg.source_language = "en"
        cfg.transcript_format = "both"
        cfg.remote_transcript_format = "en"
        cfg.mic_transcript_enabled = True
        cfg.mic_transcript_format = "ja"
        cfg.mic_source_language = "ja"
        print("[config] 起動モード: 英語→日本語字幕")
    else:
        print(f"[config] 未知の RT_TRANSLATOR_MODE を無視: {mode}")


def load_config() -> Config:
    cfg = Config()
    path = Path("config.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        _apply_values(cfg, data, str(path))
    _apply_launch_mode(cfg)
    return cfg
