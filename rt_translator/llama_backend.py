"""ローカル翻訳用llama-serverの所有・接続・ストリーム処理。"""
import atexit
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from .config import Config
from .gemma_setup import ROOT, RUNTIME_VERSION, ensure_model, ensure_runtime, get_model_spec
from .translator import Translator
from .win_process import bind_child_lifetime, kill_orphaned_processes


class LocalLlamaTranslator(Translator):
    def __init__(self, cfg: Config, *, model_key: str, model_path: str,
                 server_path: str, context_size: int,
                 startup_timeout: float = 300, request_timeout: float = 30):
        if cfg.device != "cuda":
            raise ValueError("この翻訳プロファイルはCUDA用です。device=cudaを指定してください。")
        if cfg.source_language not in {"en", "eng", "english", "en-US", "en_us"}:
            raise ValueError("この翻訳プロファイルは英語→日本語用です。英語用起動ファイルを使ってください。")
        if cfg.target_lang != "jpn_Jpan":
            raise ValueError("この翻訳プロファイルの翻訳先は日本語です。")
        if context_size < 512 or context_size > 2048:
            raise ValueError("字幕用コンテキストは512〜2048で指定してください。")
        self.cfg = cfg
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._process = None
        self._log = None
        self._lock = threading.Lock()
        self.last_timings = {}
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._api_key = secrets.token_urlsafe(32)
        if model_path:
            self.model_path = Path(model_path).resolve()
            self._model_label = self.model_path.stem
            self._compute_type = "custom"
        else:
            spec = get_model_spec(model_key)
            self._model_label = spec.label
            self._compute_type = spec.compute_type
            self.model_path = ensure_model(model_key)
        server = Path(server_path).resolve() if server_path else ensure_runtime()
        if not server.is_file() or not self.model_path.is_file():
            raise FileNotFoundError("翻訳モデルまたはllama-serverが見つかりません。")
        log_dir = ROOT / ".runtime" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{cfg.translation_backend}-{os.getpid()}-{time.time_ns()}.log"
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self._base_url = f"http://127.0.0.1:{port}"
        command = [str(server), "--model", str(self.model_path),
                   "--host", "127.0.0.1", "--port", str(port),
                   "--ctx-size", str(context_size), "--parallel", "1",
                   "--n-gpu-layers", "999", "--device", f"CUDA{cfg.device_index}",
                   "--fit", "off", "--flash-attn", "on", "--jinja", "--no-webui",
                   "--skip-chat-parsing", "--verbosity", "4",
                   "--cache-ram", "0",
                   "--no-context-shift", *self._server_template_args()]
        environment = os.environ.copy()
        environment["LLAMA_API_KEY"] = self._api_key
        # 他のllama.cppアプリの環境設定を、この所有プロセスへ持ち込まない。
        for key in list(environment):
            if key.startswith("LLAMA_ARG_"):
                environment.pop(key)
        # 前回の異常終了で残ったサーバーが VRAM を占有していると、今回のロードが
        # 共有メモリへあふれて極端に遅くなる(起動時に固まったように見える)。
        # 利用者が指定した llama.cpp には他用途のサーバーがありうるので、同梱版に限る。
        if not server_path:
            kill_orphaned_processes(server.parent, server.name)
        try:
            self._log = self.log_path.open("wb")
            self._process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
                cwd=server.parent, env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            # コンソールの×や強制終了では atexit が走らないため、OS 側でも道連れにする。
            bind_child_lifetime(self._process)
            atexit.register(self.close)
            print(f"[mt] {self._model_label}をGPUへロード中...")
            self._wait_ready()
            log = self.log_path.read_text(encoding="utf-8", errors="replace")
            offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
            if not offload or int(offload[1]) != int(offload[2]) or int(offload[1]) == 0:
                raise RuntimeError(f"全レイヤーのGPU配置を確認できません: {self.log_path}")
            self._offloaded_layers = int(offload[1])
            print(f"[mt] {self._model_label} / CUDA:{cfg.device_index} 準備完了 "
                  f"({self._offloaded_layers}レイヤーをGPUへ配置)")
        except BaseException:
            self.close()
            raise

    def _open_request(self, endpoint: str, payload=None, timeout=None):
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._base_url + endpoint, data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._api_key})
        return self._opener.open(request, timeout=timeout or self._request_timeout)

    def _request(self, endpoint: str, payload=None, timeout=None):
        with self._open_request(endpoint, payload, timeout) as response:
            return json.load(response)

    def _wait_ready(self):
        started = time.monotonic()
        deadline = started + self._startup_timeout
        next_report = started + 15
        while time.monotonic() < deadline:
            if time.monotonic() >= next_report:
                print(f"[mt] ロード中... {time.monotonic() - started:.0f}秒経過", flush=True)
                next_report += 15
            if self._process.poll() is not None:
                raise RuntimeError(f"llama-serverが起動に失敗しました。ログ: {self.log_path}")
            try:
                if self._request("/health", timeout=2).get("status") == "ok":
                    self._request("/props", timeout=2)  # 所有サーバーへの認証を確認
                    return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(0.25)
        raise TimeoutError(f"翻訳モデルのロードが時間切れになりました。ログ: {self.log_path}")

    def _server_template_args(self):
        return []

    def translate(self, text: str, beam_size: int = 1,
                  length_penalty: float = 1.0, repetition_penalty: float = 1.0,
                  no_repeat_ngram_size: int = 0, max_decoding_length: int = 256,
                  max_unit_chars: int = 180, split_sentences: bool = True) -> str:
        if not text.strip():
            return ""
        # NLLBのbeam/文分割設定は使用しない。全文を公式翻訳テンプレートに渡す。
        payload = self._translation_payload(text, max_decoding_length, stream=False)
        with self._lock:
            self._check_running()
            result = self._request("/v1/chat/completions", payload)
            choice = result["choices"][0]
            content = choice.get("message", {}).get("content")
            self._check_completion(content, choice.get("finish_reason"))
            self.last_timings = result.get("timings", {})
            return content.strip()

    def _translation_payload(self, text, max_decoding_length, stream):
        raise NotImplementedError

    def _check_running(self):
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("翻訳モデルの実行プロセスが停止しています。")

    @staticmethod
    def _check_completion(content, reason):
        if reason != "stop":
            raise RuntimeError("翻訳が途中で停止しました。生成長・入力長を確認してください。")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("翻訳モデルが空の翻訳を返しました。")

    def translate_stream(self, text: str, max_decoding_length: int = 256, **_kwargs):
        """累積した訳を順次返す。正常終了の検証前は暫定表示だけに使う。"""
        if not text.strip():
            return
        payload = self._translation_payload(text, max_decoding_length, stream=True)
        with self._lock:
            self._check_running()
            content, reason = "", None
            self.last_timings = {}
            with self._open_request("/v1/chat/completions", payload) as response:
                for event in self._stream_events(response):
                    if "error" in event:
                        raise RuntimeError(f"翻訳モデル: {event['error']}")
                    if "timings" in event:
                        self.last_timings = event["timings"]
                    choices = event.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {}).get("content")
                    if isinstance(delta, str) and delta:
                        content += delta
                        yield content.strip()
                    if choice.get("finish_reason") is not None:
                        reason = choice["finish_reason"]
            self._check_completion(content, reason)

    @staticmethod
    def _stream_events(response):
        # SSEをイベント境界で解析。ping/comment、複数data行、UTF-8を扱う。
        data = []
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:
                if data:
                    payload = "\n".join(data)
                    data = []
                    if payload == "[DONE]":
                        return
                    yield json.loads(payload)
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
        # 終端がない切断を成功扱いしない。
        raise RuntimeError("翻訳ストリームが途中で切断されました。")

    def runtime_info(self) -> dict:
        return {"device": "cuda", "device_index": self.cfg.device_index,
                "compute_type": self._compute_type, "model": self._model_label,
                "runtime": f"llama.cpp {RUNTIME_VERSION}",
                "offloaded_layers": self._offloaded_layers, "model_path": str(self.model_path)}

    def close(self):
        process, self._process = self._process, None
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None
            atexit.unregister(self.close)
