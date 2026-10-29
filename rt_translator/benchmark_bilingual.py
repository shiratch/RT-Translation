"""保存した日英音声を実時間で再生し、字幕キュー到着までを比較する。

音声デバイス・字幕画面は起動しない。モデルは通常どおりロードする。
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import queue
import statistics
import threading
import time

from .main import _add_nvidia_dll_dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english-audio", required=True, type=Path)
    parser.add_argument("--japanese-audio", required=True, type=Path)
    parser.add_argument("--profile", default="translategemma12b")
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats は1以上")
    for path in (args.english_audio, args.japanese_audio):
        if not path.is_file():
            parser.error(f"音声がありません: {path}")

    _add_nvidia_dll_dirs()
    import numpy as np
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import get_speech_timestamps
    from .asr import StreamingTranscriber
    from .config import load_config
    from .dictionary import UserDictionary
    from .speaker import SpeakerChangeDetector
    from .translator import create_translator, TranslationWorker

    cfg = replace(load_config(args.profile), source_language="en", log_latency=False,
                  shutdown_drain_seconds=0)
    dictionary = UserDictionary(cfg.user_dictionary)
    english = decode_audio(str(args.english_audio), sampling_rate=16000)
    japanese = decode_audio(str(args.japanese_audio), sampling_rate=16000)
    if not english.size or not japanese.size:
        parser.error("空の音声は測定できません")
    model_owner = StreamingTranscriber(cfg, queue.Queue(), queue.Queue(), threading.Event())
    model, model_lock = model_owner.model, model_owner.model_lock
    if model.model.device != "cuda":
        raise RuntimeError("GPU比較のためCUDA実行が必要です")

    report = {"profile": args.profile, "device": model.model.device,
              "compute_type": model.model.compute_type,
              "note": "Warmed realtime 100ms replay; UI queue arrival, no audio devices or UI. "
                      "Mixed speech is separate utterances with 0.8s gaps.",
              "english_audio": str(args.english_audio), "japanese_audio": str(args.japanese_audio),
              "runs": []}
    engine = create_translator(cfg)
    try:
        report["translation_runtime"] = engine.runtime_info()
        engine.translate("We will finish the integration test by Friday.")
        for bilingual in (False, True):
            model_owner.bilingual = bilingual
            model_owner._transcribe(english[:16000], cfg.partial_beam_size, "partial")

        def replay(audio, bilingual, name):
            events = []
            origin = time.perf_counter()

            class TraceQueue(queue.Queue):
                def __init__(self, stage):
                    super().__init__()
                    self.stage = stage

                def put(self, item, *args, **kwargs):
                    events.append({"stage": self.stage,
                                   "seconds": round(time.perf_counter() - origin, 4), **item})
                    return super().put(item, *args, **kwargs)

            aq, tq, uq = queue.Queue(), TraceQueue("asr"), TraceQueue("ui")
            astop, mstop = threading.Event(), threading.Event()
            run_cfg = replace(cfg, asr_bilingual=bilingual)
            asr = StreamingTranscriber(
                run_cfg, aq, tq, astop, model=model, model_lock=model_lock,
                dictionary=dictionary,
                speaker_detector=SpeakerChangeDetector(cfg) if cfg.speaker_change_detection else None)
            mt = TranslationWorker(run_cfg, engine, tq, uq, mstop, dictionary=dictionary)
            asr.start()
            mt.start()
            padded = np.concatenate((audio, np.zeros(32000, dtype=np.float32)))
            origin = time.perf_counter()
            try:
                for offset in range(0, len(padded), 1600):
                    time.sleep(max(0, origin + (offset + 1600) / 16000 - time.perf_counter()))
                    aq.put(padded[offset:offset + 1600])
            finally:
                # 生産側を終了してから、翻訳側をdrainする。
                astop.set()
                asr.join(30)
                mstop.set()
                mt.join(30)
            if asr.is_alive() or mt.is_alive():
                raise RuntimeError("再生ワーカーが終了しませんでした")
            onset = get_speech_timestamps(audio)[0]["start"] / 16000
            first = next((e["seconds"] for e in events
                          if e["stage"] == "ui" and e["kind"] == "partial" and e["ja"]), None)
            result = {"name": name, "bilingual": bilingual, "first_gray_seconds": first,
                      "speech_onset_seconds": onset,
                      "first_gray_after_onset": None if first is None else first - onset,
                      "asr_finals": [e for e in events if e["stage"] == "asr" and e["kind"] == "final"],
                      "ui_finals": [e for e in events if e["stage"] == "ui" and e["kind"] == "final"],
                      "events": events}
            report["runs"].append(result)
            print(json.dumps({k:v for k,v in result.items() if k != "events"}, ensure_ascii=False), flush=True)

        for repeat in range(args.repeats):
            for bilingual in ((False, True) if repeat % 2 == 0 else (True, False)):
                replay(english, bilingual, "english")
        gap = np.zeros(12800, dtype=np.float32)
        replay(np.concatenate((english, gap, japanese, gap, english)), True, "en-ja-en")

        # 同じモデルで相手音声の日英判定と、日本語固定マイクを同時実行する。
        remote = StreamingTranscriber(replace(cfg, asr_bilingual=True), queue.Queue(), queue.Queue(),
                                      threading.Event(), model=model, model_lock=model_lock)
        mic = StreamingTranscriber(cfg, queue.Queue(), queue.Queue(), threading.Event(),
                                   model=model, model_lock=model_lock, source_language="ja",
                                   emit_partials=False, use_hotwords=False)
        concurrent = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for audio in (english, japanese, english):
                rf = pool.submit(remote._transcribe, audio, cfg.final_beam_size, "final")
                mf = pool.submit(mic._transcribe, japanese, cfg.final_beam_size, "final")
                remote_text, mic_text = rf.result(), mf.result()
                concurrent.append({"remote": remote_text, "remote_language": remote.detected_language,
                                   "mic": mic_text, "mic_language": mic.detected_language})
        report["concurrent"] = concurrent
        report["first_gray_median_seconds"] = {
            str(bilingual): statistics.median(r["first_gray_seconds"] for r in report["runs"]
                                             if r["name"] == "english" and r["bilingual"] == bilingual)
            for bilingual in (False, True)}
        print(json.dumps(report["first_gray_median_seconds"]), flush=True)
    finally:
        engine.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
