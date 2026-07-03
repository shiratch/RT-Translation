"""WASAPI ループバックでスピーカー出力音声をキャプチャし、16kHz mono float32 で流す。

マイクや仮想オーディオデバイスは不要。既定の再生デバイスに対応する
loopback デバイスを PyAudioWPatch で開く。
"""
import queue
import numpy as np
import pyaudiowpatch as pyaudio

TARGET_RATE = 16000


class LoopbackCapture:
    def __init__(self, out_queue: queue.Queue, block_seconds: float = 0.1):
        self.out_queue = out_queue
        self.block_seconds = block_seconds
        self._pa = None
        self._stream = None
        self._channels = 1
        self._rate = 48000

    def start(self):
        self._pa = pyaudio.PyAudio()
        device = self._find_loopback_device()
        self._rate = int(device["defaultSampleRate"])
        self._channels = max(1, int(device["maxInputChannels"]))
        frames = int(self._rate * self.block_seconds)
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._channels,
            rate=self._rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=frames,
            stream_callback=self._callback,
        )
        print(f"[audio] キャプチャ開始: {device['name']} "
              f"({self._rate}Hz, {self._channels}ch -> {TARGET_RATE}Hz mono)")

    def _find_loopback_device(self) -> dict:
        wasapi = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = self._pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if default_out.get("isLoopbackDevice"):
            return default_out
        for lb in self._pa.get_loopback_device_info_generator():
            if default_out["name"] in lb["name"]:
                return lb
        raise RuntimeError(
            f"既定出力デバイス '{default_out['name']}' の loopback が見つかりません。")

    def _callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.float32)
        if self._channels > 1:
            data = data.reshape(-1, self._channels).mean(axis=1)
        if self._rate != TARGET_RATE:
            n_out = int(len(data) * TARGET_RATE / self._rate)
            x_out = np.linspace(0.0, len(data) - 1, n_out)
            data = np.interp(x_out, np.arange(len(data)), data).astype(np.float32)
        self.out_queue.put(data)
        return (None, pyaudio.paContinue)

    def stop(self):
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa is not None:
            self._pa.terminate()
