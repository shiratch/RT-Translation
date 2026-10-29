"""エントリポイント: python -m rt_translator.main"""
import os
import queue
import sys
import threading
import traceback
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


def _source_language_label(language: str, is_japanese_source_language) -> str:
    normalized = language.lower().replace("-", "_")
    if is_japanese_source_language(language):
        return "日本語音声"
    if normalized in {"en", "eng", "english", "en_us", "en_gb", "eng_latn"}:
        return "英語音声"
    return f"{language}音声"


def main():
    _add_nvidia_dll_dirs()

    from .config import load_config
    from .dictionary import UserDictionary
    from .transcript import TranscriptWriter
    from .translator import create_translator, is_japanese_source_language

    cfg = load_config()
    print("=== RT Translator: モデルをロードしています(初回はダウンロードに数分かかります) ===")
    dictionary = UserDictionary(cfg.user_dictionary)
    transcript_writer = TranscriptWriter(cfg)
    if is_japanese_source_language(cfg.source_language):
        print("[mt] 日本語音声の文字起こしモード: 翻訳モデルはロードしません")
        translator = None
    else:
        translator = create_translator(cfg)
    # 以降(Whisper のロード等)で失敗しても、翻訳サーバーは必ず止める。
    try:
        _run_session(cfg, translator, dictionary, transcript_writer)
    finally:
        if translator is not None:
            translator.close()


def _run_session(cfg, translator, dictionary, transcript_writer):
    from .asr import StreamingTranscriber
    from .audio_capture import LoopbackCapture, MicrophoneCapture
    from .overlay import SubtitleOverlay
    from .transcript import TranscriptOnlyWorker
    from .translator import TranslationWorker, is_japanese_source_language

    stop_event = threading.Event()
    audio_queue = queue.Queue()
    text_queue = queue.Queue()
    ui_queue = queue.Queue()
    mic_capture = None
    mic_asr = None
    mic_writer = None

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
                source_language=mic_source_language, name="asr-mic",
                emit_partials=False,
                use_hotwords=cfg.mic_dictionary_hotwords,
                min_segment_rms=cfg.mic_min_segment_rms,
                suppressed_phrases=cfg.mic_asr_suppressed_phrases,
                suppressed_substring_max_chars=cfg.mic_asr_suppressed_substring_max_chars)
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

    language_label = _source_language_label(cfg.source_language, is_japanese_source_language)
    if cfg.asr_bilingual and cfg.source_language == "en":
        language_label = "英語・日本語音声（英語は翻訳、日本語は文字起こし）"
    print(f"=== 準備完了。{language_label}を再生すると字幕が表示されます(字幕右クリックで終了) ===")
    overlay = SubtitleOverlay(cfg, ui_queue, stop_event)
    try:
        overlay.run()
    finally:
        stop_event.set()
        print("=== 終了処理中: 残りの発話を保存しています ===", flush=True)
        for audio in (capture, mic_capture):
            if audio is not None:
                try:
                    audio.stop()
                except Exception as exc:
                    print(f"[audio] キャプチャ停止に失敗しました: {exc}")
        asr.join(timeout=cfg.shutdown_drain_seconds + 1.0)
        if mic_asr is not None:
            mic_asr.join(timeout=cfg.shutdown_drain_seconds + 1.0)
        mt.join(timeout=cfg.shutdown_drain_seconds + 5.0)
        if mic_writer is not None:
            mic_writer.join(timeout=cfg.shutdown_drain_seconds + 1.0)


def _run():
    from .win_process import disable_console_quick_edit
    restore_console = disable_console_quick_edit()
    exit_code = 1
    try:
        main()
        exit_code = 0
    except SystemExit as exc:
        if exc.code is None or isinstance(exc.code, int):
            exit_code = exc.code or 0
        else:
            print(exc.code, file=sys.stderr)
    except KeyboardInterrupt:
        exit_code = 130
    except BaseException:
        traceback.print_exc()
    finally:
        # 後片付けは main() 内で済んでいる。ここで通常終了に任せると、
        # 推論中のデーモンスレッドや CUDA モデルの破棄でインタプリタの終了処理が
        # 止まり、プロセスごと VRAM を掴み続けることがある。OS に即時回収させる。
        restore_console()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    _run()
