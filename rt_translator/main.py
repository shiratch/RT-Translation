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
    from .audio_capture import LoopbackCapture
    from .config import load_config
    from .overlay import SubtitleOverlay
    from .translator import NllbCT2Translator, TranslationWorker

    cfg = load_config()
    stop_event = threading.Event()
    audio_queue = queue.Queue()
    text_queue = queue.Queue()
    ui_queue = queue.Queue()

    print("=== RT Translator: モデルをロードしています(初回はダウンロードに数分かかります) ===")
    translator = NllbCT2Translator(cfg)
    speaker_detector = None
    if cfg.speaker_change_detection:
        from .speaker import SpeakerChangeDetector
        speaker_detector = SpeakerChangeDetector(cfg)
    asr = StreamingTranscriber(cfg, audio_queue, text_queue, stop_event,
                               speaker_detector=speaker_detector)
    mt = TranslationWorker(cfg, translator, text_queue, ui_queue, stop_event)

    capture = LoopbackCapture(audio_queue)
    capture.start()
    asr.start()
    mt.start()

    print("=== 準備完了。英語音声を再生すると字幕が表示されます(字幕右クリックで終了) ===")
    overlay = SubtitleOverlay(cfg, ui_queue, stop_event)
    try:
        overlay.run()
    finally:
        stop_event.set()
        capture.stop()


if __name__ == "__main__":
    main()
