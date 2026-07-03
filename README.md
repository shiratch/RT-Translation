# RT Translator — リアルタイム英→日 会議字幕

Google Meet などのリモート会議で、相手の英語音声をリアルタイムに日本語字幕として
画面最前面のオーバーレイに表示する Windows 用アプリ。

- **完全ローカル処理**(音声はどこにも送信されない、API コストなし)
- 音声認識: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3-turbo`(GPU)
- 翻訳: NLLB-200-distilled-600M(CTranslate2, GPU)
- 音声取得: WASAPI ループバック — **マイクや仮想オーディオデバイスの設定は不要**。
  スピーカーに出ている音をそのまま拾う
- 体感遅延: 発話から暫定字幕(グレー)まで 1〜1.5 秒、確定字幕(白)は無音検出後

## 必要環境

- Windows 11(Windows 10 でも可)
- NVIDIA GPU(RTX 2080 / 8GB で動作確認。VRAM 使用量は約 2.5GB)
- Python 3.10〜3.12
- CUDA Toolkit のインストールは**不要**(pip の nvidia wheel から DLL を読み込む)

## セットアップ & 実行

**`run.bat` をダブルクリックするだけ。** venv がなければ初回に自動で作成して
依存をインストールし(数分)、以降は即起動する。

手動でやる場合:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m rt_translator.main
```

### 別の PC で使う

コードをコピーして `run.bat` を実行するだけ(NVIDIA GPU + Python 3.10〜3.12 が必要)。
ただし **`.venv` フォルダはコピーしないこと**(venv はマシン間で移植できない。
削除しておけば初回起動時に作り直される)。モデルは初回起動時に再ダウンロードされる。

- 初回起動時に Whisper(約 1.6GB)と NLLB(約 600MB)のモデルを自動ダウンロード
- 起動後、英語音声(会議・YouTube 等)を再生すると画面下部に字幕が出る
- 字幕帯は**左ドラッグで移動**、**右クリック**でフォントサイズ変更・字幕クリア・終了

## 設定

プロジェクト直下に `config.json` を置くと既定値を上書きできる。
キー一覧と既定値は [rt_translator/config.py](rt_translator/config.py) を参照。例:

```json
{
  "font_size": 28,
  "show_source": true,
  "partial_interval": 0.6,
  "whisper_model": "medium.en"
}
```

主なチューニング項目:

| キー | 効果 |
|---|---|
| `partial_interval` | 暫定字幕の更新間隔。小さいほど反応が速いが GPU 負荷増 |
| `silence_finalize` | この秒数の無音で発話を確定。小さいほど確定が速いが文が細切れに |
| `whisper_model` | `large-v3-turbo` / `medium.en` / `small.en` など。小さいほど速く精度低下 |
| `show_source` | 英語原文も字幕上に小さく表示 |
| `device` | `"cuda"` / `"cpu"`(CUDA 初期化失敗時は自動で CPU フォールバック) |

## 仕組み

```
スピーカー出力 → WASAPI loopback (16kHz mono)
  → Silero VAD で発話区間管理
  → faster-whisper で 0.8 秒ごとに暫定認識 / 無音で確定認識
  → NLLB-200 で英→日翻訳(1 文あたり数十 ms)
  → 透過オーバーレイに表示(暫定=グレー、確定=白)
```

翻訳エンジンは `rt_translator/translator.py` の `Translator` インターフェースを
実装すれば DeepL / Claude API 等に差し替え可能。

## ライセンスについての注意

- 翻訳モデル NLLB-200 の重みは [CC-BY-NC-4.0](https://huggingface.co/facebook/nllb-200-distilled-600M)(**非商用限定**)。
  商用利用する場合は翻訳エンジンを別のもの(DeepL API 等)に差し替えること
- Whisper(MIT)、faster-whisper(MIT)、CTranslate2(MIT)

## トラブルシューティング

- **字幕が出ない**: 音の出力先が「既定の再生デバイス」か確認。ヘッドセットを
  既定デバイスにして会議音声をそこに出す
- **CUDA エラーが出る**: NVIDIA ドライバを更新。`config.json` に
  `{"device": "cpu"}` を書けば CPU でも動く(`whisper_model` を `small.en` 程度に)
- **認識が途切れる/細切れ**: `silence_finalize` を 0.8〜1.0 に上げる
