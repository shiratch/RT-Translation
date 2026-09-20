"""ローカル翻訳用GGUFと公式llama.cpp Windows CUDAバイナリの準備。"""
import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelSpec:
    label: str
    repo_id: str
    revision: str
    filename: str
    compute_type: str
    download_gb: float


MODEL_SPECS = {
    "12b-q8_0": ModelSpec(
        "TranslateGemma 12B Q8_0", "mradermacher/translategemma-12b-it-GGUF",
        "fdf84c9f6fe14e69d58814f14e7b5b63bb6a1b28", "translategemma-12b-it.Q8_0.gguf",
        "Q8_0", 13),
    "27b-q6_k": ModelSpec(
        "TranslateGemma 27B Q6_K", "mradermacher/translategemma-27b-it-GGUF",
        "0f2c24b456631d519a919e20429119bbd524d86e", "translategemma-27b-it.Q6_K.gguf",
        "Q6_K", 22.2),
    "hymt2-30b-a3b-q4_k_m": ModelSpec(
        "Hy-MT2 30B-A3B Q4_K_M", "tencent/Hy-MT2-30B-A3B-GGUF",
        "fd3dcbb6b31e9e03923ef4f9f42500f74c3851c3", "Hy-MT2-30B-A3B-Q4_K_M.gguf",
        "Q4_K_M", 18.2),
}


def get_model_spec(model: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model]
    except KeyError:
        raise ValueError(f"未知の翻訳モデル: {model}（{', '.join(MODEL_SPECS)}）") from None


RUNTIME_VERSION = "b10964"
RUNTIME_DIR = ROOT / ".runtime" / "llama-b10964-cuda13.3"
ASSETS = {
    "llama-b10964-bin-win-cuda-13.3-x64.zip":
        "cd63ae76ad78a1540aa0f30f6c6284bab14c146d99a58f70c3f0a38cb9c62351",
    "cudart-llama-bin-win-cuda-13.3-x64.zip":
        "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e",
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(archive: Path, destination: Path):
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            if not (root / entry.filename).resolve().is_relative_to(root):
                raise ValueError(f"アーカイブに不正なパスがあります: {entry.filename}")
        bundle.extractall(root)


def ensure_runtime() -> Path:
    server = RUNTIME_DIR / "llama-server.exe"
    marker = RUNTIME_DIR / ".ready"
    if marker.is_file() and server.is_file():
        return server
    if sys.platform != "win32":
        raise RuntimeError("自動セットアップはWindows用です。gemma_server_pathを指定してください。")
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    downloads = ROOT / ".runtime" / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for filename, expected in ASSETS.items():
        archive = downloads / filename
        if not archive.is_file() or _digest(archive) != expected:
            url = f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_VERSION}/{filename}"
            print(f"[setup] llama.cppを取得: {filename}", flush=True)
            temporary = archive.with_suffix(".download")
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if _digest(temporary) != expected:
                raise RuntimeError(f"ダウンロードのSHA256が一致しません: {filename}")
            temporary.replace(archive)
        _extract(archive, RUNTIME_DIR)
    if not server.is_file():
        raise RuntimeError(f"llama-server.exeが見つかりません: {RUNTIME_DIR}")
    marker.write_text(RUNTIME_VERSION + "\n", encoding="utf-8")
    return server


def ensure_model(model: str = "12b-q8_0") -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    spec = get_model_spec(model)
    kwargs = dict(repo_id=spec.repo_id, filename=spec.filename, revision=spec.revision,
                  cache_dir=str(ROOT / ".model-cache"))
    try:
        return Path(hf_hub_download(**kwargs, local_files_only=True))
    except LocalEntryNotFoundError:
        print(f"[setup] {spec.label}を取得します（約{spec.download_gb:g}GB）...", flush=True)
        return Path(hf_hub_download(**kwargs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_SPECS, default="12b-q8_0")
    args = parser.parse_args()
    print(ensure_runtime())
    print(ensure_model(args.model))


if __name__ == "__main__":
    main()
