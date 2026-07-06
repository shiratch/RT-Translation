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
    silence_finalize: float = 0.45      # この長さの無音でセグメント確定 [秒]
    max_utterance: float = 12.0         # 強制確定までの最大発話長 [秒]
    min_speech: float = 0.30            # VAD上の実発話累計がこれ未満なら無視 [秒]
    asr_min_segment_peak: float = 1e-3  # ほぼ無音の塊をWhisperへ渡さない
    asr_min_segment_rms: float = 0.0    # 相手音声用RMSゲート(0で無効)
    partial_beam_size: int = 1
    final_beam_size: int = 1
    asr_suppressed_phrases: list[str] = dataclasses.field(default_factory=lambda: [
        "Amazon",
        "栗子",
        "株式 関値 位置ループ",
        "株式 閾値 位置ループ",
        "ご視聴ありがとうございました",
        "ご清聴ありがとうございました",
    ])
    asr_suppressed_substring_max_chars: int = 12
    english_asr_reject_cjk: bool = True  # 英語認識なのに日本語文字だけ出た場合は破棄

    # --- 翻訳 ---
    nllb_repo: str = "JustFrederik/nllb-200-distilled-1.3B-ct2-int8"
    nllb_compute_type: str = "int8_float16"
    target_lang: str = "jpn_Jpan"
    source_lang_nllb: str = "eng_Latn"
    translation_partial_beam_size: int = 1
    translation_final_beam_size: int = 4
    translation_length_penalty: float = 1.0
    translation_repetition_penalty: float = 1.1
    translation_no_repeat_ngram_size: int = 3
    translation_max_decoding_length: int = 256
    translation_unit_max_chars: int = 120
    translation_final_single_pass: bool = True
    translation_single_pass_max_chars: int = 160
    translation_buffer_final_fragments: bool = True
    translation_fragment_flush_seconds: float = 0.6
    translation_incomplete_fragment_flush_seconds: float = 3.0
    translation_deferred_fragment_flush_seconds: float = 4.0
    translation_fragment_max_chars: int = 220
    translation_fragment_max_segments: int = 4
    translation_defer_formula_fragments: bool = True
    translation_formula_fragment_max_chars: int = 80
    translation_suppressed_phrases: list[str] = dataclasses.field(default_factory=lambda: [
        "栗子",
        "株式 関値 位置ループ",
        "株式 閾値 位置ループ",
        "ご視聴ありがとうございました",
        "ご清聴ありがとうございました",
    ])
    translation_reject_short_cjk: bool = True
    translation_suspicious_source_min_chars: int = 24
    translation_suspicious_target_max_chars: int = 12
    translation_normalize_punctuation: bool = True

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
    transcript_english_label: str = "EN_SOURCE"
    transcript_translation_label: str = "JA_TRANSLATION"
    transcript_rejected_translation_label: str = "JA_TRANSLATION_REJECTED"
    transcript_japanese_label: str = "JA"

    # --- マイク文字起こし保存 ---
    mic_transcript_enabled: bool = False
    mic_source_language: str = "ja"     # 自分のマイク音声は既定で日本語認識
    mic_device_name: str = ""           # 空なら既定の録音デバイス
    mic_transcript_label: str = "MIC"
    mic_dictionary_hotwords: bool = False  # 無音時に辞書語へ寄りすぎるのを避ける
    mic_min_segment_rms: float = 0.015  # マイク無音/環境音からの幻覚を抑える

    # --- その他 ---
    log_latency: bool = True            # 各段の処理時間をコンソールに出す
    user_dictionary: str = "user_dictionary.txt"  # ユーザー辞書(VoiceText と同書式)
    hallucinations_file: str = "hallucinations.txt"  # 1行1語句の幻覚抑制リスト
    shutdown_drain_seconds: float = 5.0  # 終了時に残りのASR/翻訳キューを処理する猶予


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
        cfg.remote_transcript_format = "both"
        cfg.mic_transcript_enabled = True
        cfg.mic_transcript_format = "ja"
        cfg.mic_source_language = "ja"
        print("[config] 起動モード: 英語→日本語字幕")
    else:
        print(f"[config] 未知の RT_TRANSLATOR_MODE を無視: {mode}")


def _read_phrase_file(path: str) -> list[str]:
    if not path:
        return []
    phrase_path = Path(path)
    if not phrase_path.exists():
        return []
    phrases: list[str] = []
    try:
        lines = phrase_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"[config] 幻覚抑制リストを読めませんでした: {phrase_path} ({exc})")
        return []
    for raw_line in lines:
        line = raw_line.strip()
        if line and not line.startswith("#"):
            phrases.append(line)
    return phrases


def _extend_unique(values: list[str], additions: list[str]):
    seen = set(values)
    for value in additions:
        if value not in seen:
            values.append(value)
            seen.add(value)


def _apply_hallucinations_file(cfg: Config):
    phrases = _read_phrase_file(cfg.hallucinations_file)
    if not phrases:
        return
    _extend_unique(cfg.asr_suppressed_phrases, phrases)
    _extend_unique(cfg.translation_suppressed_phrases, phrases)
    print(f"[config] 幻覚抑制リストを読み込み: {cfg.hallucinations_file} ({len(phrases)}件)")


def _apply_env_aliases(cfg: Config):
    threshold = os.environ.get("VOICE_STREAM_SILENCE_THRESHOLD", "").strip()
    if not threshold:
        return
    try:
        cfg.mic_min_segment_rms = float(threshold)
        print("[config] VOICE_STREAM_SILENCE_THRESHOLD を "
              f"mic_min_segment_rms として適用: {cfg.mic_min_segment_rms}")
    except ValueError:
        print(f"[config] VOICE_STREAM_SILENCE_THRESHOLD を無視: {threshold}")


def load_config() -> Config:
    cfg = Config()
    path = Path("config.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        _apply_values(cfg, data, str(path))
    _apply_launch_mode(cfg)
    _apply_env_aliases(cfg)
    _apply_hallucinations_file(cfg)
    return cfg
