"""保存音声で ASR/翻訳の処理時間を測る。録音・字幕UIは起動しない。

python -m rt_translator.benchmark --audio sample.wav --language en --profile rtx5090
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import queue
import statistics
import threading
import time

from .main import _add_nvidia_dll_dirs


def _measure(operation, repeats):
    operation()  # 初回の CUDA 初期化などを計測から除外
    samples = []
    outputs = []
    for _ in range(repeats):
        started = time.perf_counter()
        outputs.append(operation())
        samples.append(round((time.perf_counter() - started) * 1000, 2))
    return {"median_ms": statistics.median(samples), "samples_ms": samples,
            "outputs": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--profile", default="", help="空なら従来設定")
    parser.add_argument("--language", choices=("en", "ja"), default="en")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--translation-text", default=
                        "We will finish the integration test by Friday. "
                        "The budget is thirteen million dollars.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats は 1 以上を指定してください")
    if not args.audio.is_file():
        parser.error(f"音声ファイルがありません: {args.audio}")

    _add_nvidia_dll_dirs()
    import ctranslate2
    from faster_whisper.audio import decode_audio
    from .asr import StreamingTranscriber
    from .config import load_config
    from .translator import create_translator, TranslationWorker

    cfg = load_config(args.profile)
    cfg.source_language = args.language
    cfg.log_latency = False
    audio = decode_audio(str(args.audio), sampling_rate=16000)
    if not audio.size:
        parser.error("音声ファイルが空です")
    report = {"profile": args.profile or "default", "ctranslate2": ctranslate2.__version__,
              "audio": str(args.audio), "audio_seconds": len(audio) / 16000,
              "note": "ウォームアップ後の単体処理時間。字幕全体の遅延・認識精度の評価ではありません。",
              "settings": {key: value for key, value in asdict(cfg).items()
                           if key.startswith(("whisper_", "nllb_", "gemma_", "hymt_")) or key in
                           {"translation_backend", "device", "device_index", "asr_bilingual",
                            "partial_beam_size", "final_beam_size"}}}
    asr = StreamingTranscriber(cfg, queue.Queue(), queue.Queue(), threading.Event())
    report["asr_device"] = asr.model.model.device
    report["asr_compute_type"] = asr.model.model.compute_type
    print("[benchmark] ASR partial / final")
    for kind, beam in (("partial", cfg.partial_beam_size), ("final", cfg.final_beam_size)):
        report[f"asr_{kind}"] = _measure(lambda: asr._transcribe(audio, beam, kind), args.repeats)
    if args.language == "en":
        translator = create_translator(cfg)
        try:
            worker = TranslationWorker(cfg, translator, queue.Queue(), queue.Queue(), threading.Event())
            report["translation_runtime"] = translator.runtime_info()
            report["translation_device"] = report["translation_runtime"].get("device")
            report["translation_compute_type"] = report["translation_runtime"].get("compute_type")
            report["translation_source"] = args.translation_text
            print("[benchmark] Translation partial / final")
            report["translation_partial"] = _measure(
                lambda: worker._translate_fast(args.translation_text), args.repeats)
            report["translation_final"] = _measure(
                lambda: worker._translate_final(args.translation_text), args.repeats)
        finally:
            translator.close()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
