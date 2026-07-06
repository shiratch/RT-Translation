"""エントリポイント: python -m rt_translator.main"""
import os
import queue
import sys
import threading
from pathlib import Path


def _add_nvidia_dll_dirs():
    """pip の nvidia-cublas/cudnn wheel の DLL を CTranslate2 から見えるようにする。

    これによりシステムへの CUDA Toolkit / cuDNN のインストールが不要になる。
    """
    if sys.platform != "win32":
        return
    # Windows は開発者モード無効だと symlink を作れず HF ダウンロードが失敗する
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    for base in map(Path, sys.path):
        nvidia = base / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def main():
    _add_nvidia_dll_dirs()

    from .asr import StreamingTranscriber
    from .audio_capture import LoopbackCapture, MicrophoneCapture
    from .config import load_config
    from .overlay import SubtitleOverlay
    from .transcript import TranscriptOnlyWorker, TranscriptWriter
    from .translator import (NllbCT2Translator, TranslationWorker,
                             is_japanese_source_language)

    cfg = load_config()
    stop_event = threading.Event()
    audio_queue = queue.Queue()
    text_queue = queue.Queue()
    ui_queue = queue.Queue()
    mic_capture = None

    print("=== RT Translator: モデルをロードしています(初回はダウンロードに数分かかります) ===")
    from .dictionary import UserDictionary
    dictionary = UserDictionary(cfg.user_dictionary)
    transcript_writer = TranscriptWriter(cfg)
    if is_japanese_source_language(cfg.source_language):
        print("[mt] 日本語音声の文字起こしモード: 翻訳モデルはロードしません")
        translator = None
    else:
        translator = NllbCT2Translator(cfg)
    speaker_detector = None
    if cfg.speaker_change_detection:
        from .speaker import SpeakerChangeDetector
        speaker_detector = SpeakerChangeDetector(cfg)
    asr = StreamingTranscriber(cfg, audio_queue, text_queue, stop_event,
                               speaker_detector=speaker_detector, dictionary=dictionary,
                               name="asr-remote")
    mt = TranslationWorker(cfg, translator, text_queue, ui_queue, stop_event,
                           dictionary=dictionary, transcript_writer=transcript_writer)

    capture = LoopbackCapture(audio_queue)
    capture.start()
    asr.start()
    mt.start()

    if cfg.mic_transcript_enabled:
        mic_source_language = cfg.mic_source_language or cfg.source_language
        mic_audio_queue = queue.Queue()
        mic_text_queue = queue.Queue()
        try:
            mic_capture = MicrophoneCapture(mic_audio_queue, device_name=cfg.mic_device_name)
            mic_capture.start()
            mic_asr = StreamingTranscriber(
                cfg, mic_audio_queue, mic_text_queue, stop_event,
                speaker_detector=None, dictionary=dictionary,
                model=asr.model, model_lock=asr.model_lock,
                source_language=mic_source_language, name="asr-mic")
            mic_writer = TranscriptOnlyWorker(
                cfg, mic_text_queue, stop_event, transcript_writer,
                mic_source_language, cfg.mic_transcript_label)
            mic_asr.start()
            mic_writer.start()
            print("[mic] マイク音声は文字起こし保存のみ行います(字幕には表示しません)")
        except Exception as exc:
            print(f"[mic] マイク文字起こしを無効化しました: {exc}")
            if mic_capture is not None:
                mic_capture.stop()
                mic_capture = None

    language_label = "日本語音声" if is_japanese_source_language(cfg.source_language) else "英語音声"
    print(f"=== 準備完了。{language_label}を再生すると字幕が表示されます(字幕右クリックで終了) ===")
    overlay = SubtitleOverlay(cfg, ui_queue, stop_event)
    try:
        overlay.run()
    finally:
        stop_event.set()
        capture.stop()
        if mic_capture is not None:
            mic_capture.stop()


if __name__ == "__main__":
    main()
