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

**`run_ja.bat` または `run_en.bat` をダブルクリックするだけ。**
venv がなければ初回に自動で作成して
依存をインストールし(数分)、以降は即起動する。

- `run_ja.bat`: 日本語音声をそのまま文字起こし。翻訳モデルはロードしない
- `run_en.bat`: 英語音声を認識して日本語字幕に翻訳。議事録の相手音声は英語原文で保存
- どちらのモードでも、マイク音声の議事録保存は自動で有効になり、日本語として保存する
- `run.bat`: `config.json` の `source_language` 設定どおりに起動

手動でやる場合:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m rt_translator.main
```

### 別の PC で使う

コードをコピーして `run_ja.bat` または `run_en.bat` を実行するだけ
(NVIDIA GPU + Python 3.10〜3.12 が必要)。
ただし **`.venv` フォルダはコピーしないこと**(venv はマシン間で移植できない。
削除しておけば初回起動時に作り直される)。モデルは初回起動時に再ダウンロードされる。

- 初回起動時に Whisper(約 1.6GB)と NLLB(約 600MB)のモデルを自動ダウンロード
- 起動後、英語音声(会議・YouTube 等)を再生すると画面下部に字幕が出る(同時表示は最大7行)
- 字幕帯は**左ドラッグで移動**、**右クリック**でフォントサイズ変更・字幕クリア・終了
- **マウスホイール**で過去の字幕に遡れる(最大200件)。右端の**シークバー**の
  ドラッグ/クリックでもジャンプ可能。遡り中は新しい字幕が来ても表示が固定され、
  最下部まで戻る(または右クリック→「最新に戻る」)と追従を再開する
- **話者交代を自動検出**して改行する。同じ話者が話し続けている間は同じ行に
  連結され、話者が変わると「– 」から始まる新しい行になる(声紋ベクトルの
  類似度による判定。初回に検出モデル 40MB を自動ダウンロード)
- 確定した字幕は `transcripts/session_YYYYMMDD_HHMMSS.txt` に自動保存される
- 任意でマイク音声も議事録に追記できる。マイク音声は字幕には表示されない

## 設定

プロジェクト直下に `config.json` を置くと既定値を上書きできる。
ただし `run_ja.bat` / `run_en.bat` で起動した場合は、そのモードが
`source_language` / 入力元別の議事録保存形式 / `mic_source_language` を上書きする。
キー一覧と既定値は [rt_translator/config.py](rt_translator/config.py) を参照。例:

```json
{
  "font_size": 28,
  "show_source": true,
  "transcript_format": "both",
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
| `source_language` | 音声認識する言語。英語は `en`、日本語文字起こしは `ja` |
| `show_source` | 英語原文も字幕上に小さく表示 |
| `device` | `"cuda"` / `"cpu"`(CUDA 初期化失敗時は自動で CPU フォールバック) |
| `translation_final_beam_size` | 確定翻訳の探索幅(既定 4)。大きいほど自然になりやすいが遅くなる |
| `translation_partial_beam_size` | 暫定翻訳の探索幅(既定 1)。反応速度優先 |
| `translation_repetition_penalty` / `translation_no_repeat_ngram_size` | 確定翻訳の繰り返し抑制 |
| `translation_buffer_final_fragments` | 短い確定断片を少し結合してから翻訳する(既定 true) |
| `translation_fragment_flush_seconds` | 断片結合を待つ最大秒数(既定 1.2) |
| `translation_suppressed_phrases` | 翻訳結果に出たら破棄する hallucination 語句リスト |
| `translation_reject_short_cjk` | 長い英語入力から短い日本語名詞片だけが出た翻訳を破棄する(既定 true) |
| `final_lines` | 画面に同時表示する最大行数、折り返し込み(既定 7) |
| `speaker_change_detection` | 話者交代の検出(既定 true)。false で1確定=1行の表示に |
| `speaker_change_threshold` | 交代判定の閾値(既定 0.45)。誤検出が多ければ下げ、見逃しが多ければ上げる |
| `asr_suppressed_phrases` | 無音時に出やすい定型 hallucination を破棄する語句リスト |
| `english_asr_reject_cjk` | 英語認識中に日本語文字だけが出た結果を破棄する(既定 true) |
| `transcript_enabled` | 確定字幕をテキストファイルへ保存する(既定 true) |
| `transcript_path` | 保存先。`{timestamp}` は起動時刻の `YYYYMMDD_HHMMSS` に置換される |
| `transcript_format` | 保存形式。`both` で英語原文+日本語訳、`en` で原文のみ、`ja` で訳文のみ |
| `remote_transcript_format` | 相手音声の保存形式。空なら `transcript_format` と同じ |
| `mic_transcript_format` | マイク音声の保存形式(既定 `ja`) |
| `transcript_timestamps` | 保存行に時刻を付ける(既定 true) |
| `transcript_source_labels` | 議事録に `REMOTE` / `MIC` などの入力元ラベルを付ける(既定 true) |
| `mic_transcript_enabled` | 既定マイク入力も議事録に保存する(既定 false)。字幕には表示しない |
| `mic_source_language` | マイク音声の認識言語(既定 `ja`) |
| `mic_device_name` | 使うマイク名の一部。空なら既定の録音デバイス |
| `user_dictionary` | ユーザー辞書ファイルのパス(後述) |

## 文字起こし保存

起動ごとに `transcripts/` 配下へ UTF-8 のテキストファイルを作成し、確定した
発話だけを追記する。既定では英語原文と日本語訳を両方保存する。保存を止めたい
場合は `config.json` に `{"transcript_enabled": false}` を書く。

`run_en.bat` では、画面字幕は日本語訳で表示し、議事録の `REMOTE` は英語原文だけを
保存する。マイク音声は日本語として認識し、`MIC` に日本語で保存する。

日本語会議をそのまま文字起こしする場合は、`config.json` に次のように書く。
この場合は翻訳モデルをロードせず、Whisper の日本語認識結果を字幕とログに保存する。
`medium.en` / `small.en` などの英語専用 Whisper モデルは日本語認識に使えないため、
日本語では既定の `large-v3-turbo` など多言語モデルを使う。

```json
{
  "source_language": "ja",
  "transcript_format": "ja"
}
```

自分のマイク音声も議事録に入れる場合は、次も追加する。マイク側は字幕には
表示されず、議事録に `MIC` として保存される。

```json
{
  "mic_transcript_enabled": true,
  "mic_source_language": "ja"
}
```

`run_ja.bat` / `run_en.bat` で起動する場合は `mic_transcript_enabled` も自動で
有効になる。マイクが拾えない場合は、起動ログの `[mic] キャプチャ開始: ...` を見て、
必要に応じて `mic_device_name` に `NVIDIA Broadcast` や `EMEET` などマイク名の一部を指定する。

## ユーザー辞書

`user_dictionary.txt`(または `config.json` の `user_dictionary` で指定した
ファイル)に 1 行 1 エントリで書く。編集は次の認識から反映(再起動不要):

```
# コメント行
間違い=正しい   ← 出力テキストを確実に置換(認識後の英文・翻訳後の和文の両方に適用)
専門用語        ← Whisper の hotwords として認識をその語に寄せる
```

辞書ファイルは .gitignore 済みなのでリポジトリにはコミットされない。
別ツールと同じ辞書を共用したい場合は `config.json` で絶対パスを指定する。
英語認識モードでは、日本語を含む辞書語は ASR の hotwords / 認識直後の置換には使わず、
翻訳後の日本語テキストへの置換だけに使う。日本語辞書語が英語ASRへ混入して
誤認識を誘発するのを避けるため。

数字・型番・製品名が誤認識される場合は、辞書に候補語を追加すると認識が寄りやすい。
例えば `4000番`、`クレスト4000番`、`レガリス6000番` のように 1 行 1 語で書く。
特定の誤りを必ず直したい場合だけ、`40番=4000番` のような置換を書く。

## 仕組み

```
スピーカー出力 → WASAPI loopback (16kHz mono)
  → Silero VAD で発話区間管理
  → faster-whisper で 0.8 秒ごとに暫定認識 / 無音で確定認識
  → NLLB-200 で英→日翻訳(日本語音声モードでは翻訳を省略)
  → 透過オーバーレイに表示(暫定=グレー、確定=白)
```

暫定字幕は反応速度優先で短い断片を翻訳するが、確定時は暫定翻訳を流用しない。
短い確定断片は最大 1.2 秒だけ前後と結合し、確定した ASR テキスト全体を
NLLB で beam search して翻訳し直す。

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
