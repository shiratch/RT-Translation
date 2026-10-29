# RT Translator — リアルタイム英→日 会議字幕

Google Meet などのリモート会議で、相手の英語音声をリアルタイムに日本語字幕として
画面最前面のオーバーレイに表示する Windows 用アプリ。

- **完全ローカル処理**(音声はどこにも送信されない、API コストなし)
- 音声認識: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3-turbo`(GPU)
- 翻訳: NLLB-200-distilled-1.3B(CTranslate2 int8, GPU)
- 音声取得: WASAPI ループバック — **マイクや仮想オーディオデバイスの設定は不要**。
  スピーカーに出ている音をそのまま拾う
- 体感遅延: 発話から暫定字幕(グレー)まで 1〜1.5 秒、確定字幕は無音検出後
- 最新の確定結果は全文ライムグリーン。複数文でもまとめて色付けし、次の結果が確定すると前の結果は白に戻る

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

- 初回起動時に Whisper(約 1.6GB)と NLLB 1.3B int8(約 1.4GB)のモデルを自動ダウンロード
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

### RTX 5090向け・バランス設定

`run_5090_ja.bat`（日本語文字起こし）または `run_5090_en.bat`（英語→日本語）を
実行する。既存の起動ファイルと `config.json` はそのまま使える。
この設定では `profiles/rtx5090.json` を `config.json` の後に重ね、最後に起動モードを適用する。
フォント、辞書、マイクデバイスなどの個人設定は引き継ぐ。

| 項目 | 従来の既定値 | RTX 5090設定 |
|---|---|---|
| 音声認識 | Whisper large-v3-turbo / INT8混合 | 同じモデル / FP16 |
| 暫定 / 確定の探索幅 | 1 / 1 | 1 / 5 |
| 暫定字幕の更新間隔 | 0.8秒 | 0.6秒 |
| 相手音声とマイクの推論 | 1ワーカー | 2ワーカー（同じモデルを共有） |
| 翻訳モデル | NLLB 1.3B | NLLB 3.3B |
| 翻訳の計算形式 | INT8混合 | FP16 |
| 句点のない確定断片の待ち時間 | 3秒 | 1秒 |
| 式などの文脈待ち時間 | 4秒 | 1.5秒 |

マイク側はどちらの設定でも確定認識だけを行う。字幕に使わない暫定認識を省き、
相手音声とGPUを取り合う機会を減らした。日本語モードでは翻訳モデルをロードしない。
Flash Attentionは既定では無効。

5090版は CTranslate2 4.8.2 / faster-whisper 1.2.1 / CUDA 12.9系のpip DLLで検証した。
古い仮想環境を使う場合は、最初に次を実行する（通常の起動ファイルは依存更新を自動実行しない）。

```powershell
.venv\Scripts\python -m pip install -r requirements-rtx5090.txt
```

新規環境では `run.bat` で環境を作成してから上記を実行する。
NLLB 3.3Bは初回にダウンロードし、プロジェクト内の `.model-cache/` に保存する。
取得する重みはINT8変換済みのため、FP16で実行しても量子化前の重みの精度には戻らない。
狙いは5090上での処理速度と、大きい翻訳モデル・広い確定探索を両立させること。
元の設定へ戻すときは `run_ja.bat` / `run_en.bat` を使う。

#### 保存音声での性能比較

```powershell
.venv\Scripts\python -m rt_translator.benchmark --audio sample.wav --language en --output benchmark-results/default.json
.venv\Scripts\python -m rt_translator.benchmark --audio sample.wav --language en --profile rtx5090 --output benchmark-results/rtx5090.json
```

WAV以外にもfaster-whisperが読める音声ファイルを指定できる。`--language ja` では翻訳測定を省く。
マイク・ループバック・字幕画面は起動せず、ウォームアップ1回の後、各処理を3回測る。
`--repeats` で回数を変えられる。モデルは通常と同様、未取得ならダウンロードする。
結果には実際の実行デバイス・計算形式・認識/翻訳文を含める。

2026-09-20、RTX 5090 / ドライバ616.92、Microsoft Ziraで作った約9.9秒の英語音声1件での
中央値は次のとおり。両方とも同じ5090で測定し、従来設定側にも共通のバグ修正を適用している。

| 単体処理 | 従来設定 | 5090設定 |
|---|---:|---:|
| 暫定認識 | 124ms | 86ms |
| 確定認識 | 138ms | 90ms |
| 暫定翻訳 | 136ms | 140ms |
| 確定翻訳 | 267ms | 276ms |

翻訳入力は固定の英語2文。これは少数の合成音声による動作・速度の目安で、
実会議での精度向上や字幕全体の遅延を保証する値ではない。
音声認識の終了を待つ時間、VAD、翻訳断片の結合待ち、同時発話の負荷は別に掛かる。
今回の認識結果は両設定で単語が一致したが、5090版の確定結果では句点が少なかったため、
句点待ちの上限も短くした。実会議で途切れが気になる場合はプロファイルで待ち時間を調整する。

GPUランタイムの対応状況は [CTranslate2変更履歴](https://github.com/OpenNMT/CTranslate2/blob/master/CHANGELOG.md)、
複数ワーカーの挙動は [faster-whisper実装・引数説明](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)
を参照。レビュー内容と残る注意点は [REVIEW.md](REVIEW.md) にまとめた。

### TranslateGemma 12B版（英語→日本語）

**`01-会議翻訳文字起こし_TranslateGemma12B.bat` をダブルクリックする。** Whisperは5090向けのTurbo FP16を使い、
翻訳をTranslateGemma 12B Q8_0へ切り替える。日本語のマイク音声は従来どおり文字起こし保存のみ。
相手音声は発話ごとに日英を判定し、英語は翻訳、日本語はそのまま字幕に表示・保存する。
言語判定のために音声を追加で待つ処理は入れず、認識と同じGPU計算結果を再利用する。
5090 / Gemma 27B / Hy-MT2の英語用起動ファイルでも有効。[日英切り替えの仕様・実測](BILINGUAL.md)。
NLLB版に戻す場合はアプリを終了し、`run_5090_en.bat` を起動する。
日本語の文字起こしには引き続き `run_5090_ja.bat` を使う。

`profiles/translategemma12b.json` を追加適用するため、個人用 `config.json` の編集は不要。
初回はモデル約12.6GBと実行環境約0.54GBを取得する。展開後の実行環境も含めてディスク容量に余裕を確保する。
このPCでは取得と実機検証を完了している。新規環境でモデルだけ先に準備する場合:

```powershell
.venv\Scripts\python -m rt_translator.gemma_setup
```

使用するもの:

- [Google TranslateGemma 12B](https://huggingface.co/google/translategemma-12b-it) の
  [mradermacherによるQ8_0変換版](https://huggingface.co/mradermacher/translategemma-12b-it-GGUF)。
  モデルのリビジョンを固定し、`.model-cache/` に保存する。ライセンスは[Gemma利用条件](https://ai.google.dev/gemma/terms)。
- [公式llama.cpp b10964](https://github.com/ggml-org/llama.cpp/releases/tag/b10964) のWindows x64 CUDA 13.3版。
  ZIPのSHA256を確認して `.runtime/` に展開する。CUDA ToolkitやPyTorchの追加インストールは不要。
  RTX 5090、ドライバ616.92で検証済み。
- 英日翻訳専用のテンプレート。モデル付属テンプレートの英日指示文を維持し、llama.cppが受け取る
  文字列形式に合わせた。コンテキストはモデルカードに合わせて2048トークン。

アプリが自分専用の `llama-server` を起動し、通常終了時に停止する。
接続は `127.0.0.1` の空きポートのみで、起動ごとに認証キーを作る。
モデル取得後の音声認識・翻訳処理はローカルで行う。
他のllama.cppやOllamaのプロセスは使用・停止しない。
ロード失敗時の診断ログは `.runtime/logs/` に残る。

このプロファイルでは暫定認識を0.4秒間隔とし、翻訳は文を細かく分割せず、一度の入力で文脈を渡す。
訳文の生成完了を待たず、生成中の訳をグレー字幕へ順次表示する。
更新時は表示済みの訳が数文字へ戻らないようにし、生成途中の描画は最大約12回/秒に抑える。
暫定認識では温度を変える再試行を省き、次の音声を待って再認識する。確定認識のbeam 5と再試行は維持する。
生成は `temperature=0`。NLLB用のbeam幅・反復ペナルティはGemmaには適用しない。
会議固有の用語置換と、訳文の金額を位置順に上書きする補正を無効にしている。
ユーザー辞書と共通の繰り返し抑制は引き続き適用する。
生成上限で途中終了した翻訳は確定訳として出さず、元の英語は議事録に残す。生成途中のグレー字幕は議事録へ保存しない。

2026-09-20の実機検証では、前節と同じ英語2文の翻訳が約300ms（3回の中央値）。
NLLB 3.3Bの約276msに対して約24ms増だった。別の会議例文8件では各3回の中央値が189〜407ms。
Whisperとの同時推論も成功し、推論後のGPU全体の使用量は18526MiB（約18.1GiB、デスクトップ等も含む）。
通常の終了処理で所有サーバーが停止することも確認した。

```powershell
.venv\Scripts\python -m rt_translator.benchmark --audio sample.wav --language en --profile translategemma12b --output benchmark-results/translategemma12b.json
```

これらは保存音声と固定文章による検証で、実会議の字幕遅延や翻訳精度を保証するものではない。
長時間の実会議での安定性・マイクとオーバーレイの動作確認は別途必要。
結果の例は [TRANSLATEGEMMA.md](TRANSLATEGEMMA.md) を参照。

### TranslateGemma 27B版（英語→日本語）

**`run_gemma27b_en.bat` をダブルクリックする。**
`profiles/translategemma27b.json` で翻訳を27B Q6_Kへ切り替える。
12B版に戻す場合はアプリを終了し、`01-会議翻訳文字起こし_TranslateGemma12B.bat` を起動する。
暫定認識の0.4秒間隔、生成途中のグレー表示、確定認識のbeam 5は12B版と共通。

[Google TranslateGemma 27B](https://huggingface.co/google/translategemma-27b-it) の
[mradermacherによるQ6_K変換版](https://huggingface.co/mradermacher/translategemma-27b-it-GGUF) を使用する。
モデルは約22.2GB（20.6GiB）、リビジョンは `0f2c24b456631d519a919e20429119bbd524d86e` に固定。
12Bのモデルとは別に保存し、llama.cppの実行環境を共用する。
RTX 5090の32GBでWhisperも併用するためQ6_Kを選択した。
このPCでは取得・全63層のGPU配置・音声認識2本との同時実行を確認済み。
GPU全体で約27.3GiBを使用した。例文8件の翻訳完了時間の中央値は約374ms（12Bは約276ms）。
同時に起動する字幕アプリは1つにする。モデルだけ先に取得する場合:

```powershell
.venv\Scripts\python -m rt_translator.gemma_setup --model 27b-q6_k
```

モデルの選択は `gemma_model`（`12b-q8_0` / `27b-q6_k`）で行い、起動時にモデル名と量子化形式を表示する。
27Bプロファイルでは個人設定の `gemma_model_path` を空に戻し、指定した27Bモデルを選ぶ。
元のモデル付属翻訳テンプレートが12Bと同一であることを確認し、同じ英日テンプレートを使用する。
速度・メモリ・訳文の比較は [TRANSLATEGEMMA.md](TRANSLATEGEMMA.md) を参照。

### Hy-MT2 30B-A3B版（英語→日本語）

**`run_hymt_en.bat` をダブルクリックする。**
`profiles/hymt2.json` を追加適用し、[Tencent公式Q4_K_M版](https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF) を使う。
モデルは約18.2GBで、このPCでは取得・全49層のGPU配置・音声認識2本との同時実行を確認済み。
既存のllama.cppを共用し、モデル付属テンプレートと英日翻訳の公式指示文を使う。

暫定認識の0.4秒間隔、生成途中のグレー表示はGemma版と共通。
最新の確定結果は、複数文を含んでも全文ライムグリーンで表示する。
2026-09-20の例文8件では翻訳完了の中央値が約111ms（Gemma 12Bは約276ms、27Bは約374ms）。
合成音声で最初のグレー字幕が出るまでの時間は約0.60〜0.68秒で、初回表示は認識待ちも含む。
GPU全体の使用量は約23.0GiB。詳しい条件・訳文比較は [HY_MT2.md](HY_MT2.md) を参照。

字幕の揺れを抑えるため `hymt_temperature=0` を使用（公式推奨は0.7）。
Gemma版へ戻す場合はアプリを終了し、`01-会議翻訳文字起こし_TranslateGemma12B.bat` または `run_gemma27b_en.bat` を起動する。
モデルだけ先に取得する場合:

```powershell
.venv\Scripts\python -m rt_translator.gemma_setup --model hymt2-30b-a3b-q4_k_m
```

日本語音声の文字起こしには、引き続き `run_5090_ja.bat` を使用する。
Hy-MT2は音声認識モデルではなく、認識されたテキストの翻訳に使う。

### 共通設定

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
| `silence_finalize` | この秒数の無音で発話を確定(既定 0.45)。小さいほど確定が速いが文が細切れに |
| `max_utterance` | 無音がなくてもこの秒数で強制確定(既定 12.0)。小さいほど字幕の確定が速い |
| `min_speech` | VAD で実発話と判定された累計秒数がこれ未満なら破棄(既定 0.30) |
| `whisper_model` | `large-v3-turbo` / `medium.en` / `small.en` など。小さいほど速く精度低下 |
| `source_language` | 音声認識する言語。英語は `en`、日本語文字起こしは `ja` |
| `show_source` | 英語原文も字幕上に小さく表示 |
| `device` | `"cuda"` / `"cpu"`(CUDA 初期化失敗時は自動で CPU フォールバック) |
| `nllb_repo` | 翻訳モデル。既定は `JustFrederik/nllb-200-distilled-1.3B-ct2-int8` |
| `translation_final_beam_size` | 確定翻訳の探索幅(既定 4)。大きいほど自然になりやすいが遅くなる |
| `translation_partial_beam_size` | 暫定翻訳の探索幅(既定 1)。反応速度優先 |
| `translation_final_single_pass` | 確定翻訳で短〜中程度の段落を文分割せず一括翻訳する(既定 true) |
| `translation_single_pass_max_chars` | 一括翻訳する最大文字数(空白除外、既定 160) |
| `translation_unit_max_chars` | 句点なし長文を翻訳前に分割する最大文字数(既定 120) |
| `translation_max_decoding_length` | NLLB の最大生成長(既定 256) |
| `translation_repetition_penalty` / `translation_no_repeat_ngram_size` | 確定翻訳の繰り返し抑制 |
| `translation_buffer_final_fragments` | 短い確定断片を少し結合してから翻訳する(既定 true) |
| `translation_fragment_flush_seconds` | 断片結合を待つ最大秒数(既定 0.6) |
| `translation_incomplete_fragment_flush_seconds` | ピリオド等で終わっていない断片を待つ最大秒数(既定 3.0) |
| `translation_deferred_fragment_flush_seconds` | 式など文脈待ち断片を待つ最大秒数(既定 4.0) |
| `translation_defer_formula_fragments` | `Study plus practice` など式っぽい短い断片を次セグメントと結合する(既定 true) |
| `translation_suppressed_phrases` | 翻訳結果に出たら破棄する hallucination 語句リスト |
| `translation_reject_short_cjk` | 長い英語入力から短い日本語名詞片だけが出た翻訳を破棄する(既定 true) |
| `final_lines` | 画面に同時表示する最大行数、折り返し込み(既定 7) |
| `latest_final_color` | 最新の確定結果全体の文字色(既定 `#B5E853`、ライムグリーン)。日本語文字起こしでも共通 |
| `speaker_change_detection` | 話者交代の検出(既定 true)。false で1確定=1行の表示に |
| `speaker_change_threshold` | 交代判定の閾値(既定 0.45)。誤検出が多ければ下げ、見逃しが多ければ上げる |
| `asr_suppressed_phrases` | 無音時に出やすい定型 hallucination を破棄する語句リスト |
| `asr_suppressed_substring_max_chars` | 短い認識結果に抑制語句が含まれる場合も破棄する最大文字数(既定 12) |
| `asr_min_segment_peak` | ほぼ無音の塊を Whisper に渡さないピーク音量しきい値(既定 0.001) |
| `asr_min_segment_rms` | 相手音声の RMS 音量ゲート。0 で無効(既定 0) |
| `hallucinations_file` | 1 行 1 語句の幻覚抑制リスト(既定 `hallucinations.txt`) |
| `english_asr_reject_cjk` | 英語認識中に日本語文字だけが出た結果を破棄する(既定 true) |
| `transcript_enabled` | 確定字幕をテキストファイルへ保存する(既定 true) |
| `transcript_path` | 保存先。`{timestamp}` は起動時刻の `YYYYMMDD_HHMMSS` に置換される |
| `transcript_format` | 保存形式。`both` で英語原文+日本語訳、`en` で原文のみ、`ja` で訳文のみ |
| `remote_transcript_format` | 相手音声の保存形式。空なら `transcript_format` と同じ |
| `mic_transcript_format` | マイク音声の保存形式(既定 `ja`) |
| `transcript_timestamps` | 保存行に時刻を付ける(既定 true) |
| `transcript_source_labels` | 議事録に `REMOTE` / `MIC` などの入力元ラベルを付ける(既定 true) |
| `transcript_english_label` | `both` 保存時の英語原文ラベル(既定 `EN_SOURCE`) |
| `transcript_translation_label` | `both` 保存時の日本語訳ラベル(既定 `JA_TRANSLATION`) |
| `transcript_rejected_translation_label` | 破棄した訳候補を議事録に残すラベル(既定 `JA_TRANSLATION_REJECTED`) |
| `mic_transcript_enabled` | 既定マイク入力も議事録に保存する(既定 false)。字幕には表示しない |
| `mic_source_language` | マイク音声の認識言語(既定 `ja`) |
| `mic_device_name` | 使うマイク名の一部。空なら既定の録音デバイス |
| `mic_dictionary_hotwords` | マイク認識でもユーザー辞書語を hotwords に使う(既定 false) |
| `mic_min_segment_rms` | マイク音声だけに掛ける RMS 音量ゲート(既定 0.015)。`VOICE_STREAM_SILENCE_THRESHOLD` でも指定可 |
| `user_dictionary` | ユーザー辞書ファイルのパス(後述) |

## 文字起こし保存

起動ごとに `transcripts/` 配下へ UTF-8 のテキストファイルを作成し、確定した
発話だけを追記する。既定では英語原文と日本語訳を両方保存する。保存を止めたい
場合は `config.json` に `{"transcript_enabled": false}` を書く。

`run_en.bat` では、画面字幕は日本語訳で表示し、議事録の `REMOTE` は英語原文を
`EN_SOURCE`、日本語訳を `JA_TRANSLATION` として両方保存する。マイク音声は
日本語として認識し、`MIC` に日本語で保存する。
翻訳が hallucination 抑制や短すぎる訳判定で破棄された場合は、字幕には出さず、
議事録には `JA_TRANSLATION_REJECTED` として訳候補を残す。

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

## 訳文の補正ルール(会議固有の用語)

NLLB が特定の会議の用語を毎回同じように誤訳する場合は、`translation_fixes.txt` に
補正ルールを書き、`config.json` で `"translation_meeting_fixes": true` にする(既定は無効)。
会議固有の語を公開しないよう、このファイルは .gitignore 済み。
別のファイルを使う場合は `translation_fixes_file` でパスを指定する。

```
# コメント行
誤訳=正訳                       ← 常に置換
原文の語,原文の語 | 誤訳=正訳    ← 英語原文に全ての語を含むときだけ置換
原文の語 | *=正訳               ← 条件に合えば訳文全体を置き換える
```

ルールは別の会議では誤訳を作りうるので、その会議に限って有効にすることを勧める。

## ハルシネーション抑制

無音や環境音で `ご視聴ありがとうございました`、`栗子` のような定型句だけが出る場合は、
`hallucinations.txt` に 1 行 1 語句で追加する。起動時に読み込まれ、
ASR と翻訳結果の両方の抑制リストへ追加される。

マイク音声は無音・環境音を拾いやすいため、既定でユーザー辞書の hotwords は使わず、
`mic_min_segment_rms` で低音量の短い塊を議事録に入れない。小声が落ちる場合は
`config.json` で `mic_min_segment_rms` を下げる。

## 仕組み

```
スピーカー出力 → WASAPI loopback (16kHz mono)
  → Silero VAD で発話区間管理
  → faster-whisper で 0.8 秒ごとに暫定認識 / 無音で確定認識
  → NLLB-200 で英→日翻訳(日本語音声モードでは翻訳を省略)
  → 透過オーバーレイに表示(暫定=グレー、最新の確定結果全体=ライムグリーン、過去の確定=白)
```

暫定字幕は反応速度優先で短い断片を翻訳するが、確定時は暫定翻訳を流用しない。
短い確定断片は最大 1.2 秒だけ前後と結合し、確定した ASR テキスト全体を
NLLB で beam search して翻訳し直す。短〜中程度の確定段落は文ごとに分割せず、
前後文脈を保ったまま NLLB に一括入力する。
`Study plus practice` のような式や短い断片は最大 4 秒だけ待ち、次の確定断片と
結合してから翻訳する。
英語 ASR の `25,0` / `10,0` のような千単位の取り損ねは `25,000` / `10,000` に補正する。
また、原文に `$13 million` のようなドル金額がある場合は、翻訳後の
`130万ドル` のような桁落ちを原文側の数値に合わせて `1300万ドル` に補正する。
単独で出た `Amazon` は ASR の短い hallucination として破棄するが、
`Amazon Web Services` のように他の語を伴う場合は残す。

既定の翻訳モデルはリアルタイム性を優先して `NLLB-200-distilled-1.3B` の
CTranslate2 int8 版にしている。3.3B は訳質検証用として使えるが、初回ダウンロード、
ロード時間、VRAM 使用量、翻訳待ち時間が大きく増える。試す場合は `config.json` に
`{"nllb_repo": "KomorebiAI/nllb-200-3.3B-int8-ct2"}` を書く。さらに軽くしたい場合は
`{"nllb_repo": "JustFrederik/nllb-200-distilled-600M-ct2-int8"}` を書くと
従来の軽量モデルへ戻せる。

翻訳エンジンは `rt_translator/translator.py` の `Translator` インターフェースを
実装すれば DeepL / Claude API 等に差し替え可能。

## ライセンスについての注意

- 翻訳モデル NLLB-200 の重みは [CC-BY-NC-4.0](https://huggingface.co/facebook/nllb-200-distilled-1.3B)(**非商用限定**)。
  商用利用する場合は翻訳エンジンを別のもの(DeepL API 等)に差し替えること
- Whisper(MIT)、faster-whisper(MIT)、CTranslate2(MIT)

## トラブルシューティング

- **字幕が出ない**: 音の出力先が「既定の再生デバイス」か確認。ヘッドセットを
  既定デバイスにして会議音声をそこに出す
- **CUDA エラーが出る**: NVIDIA ドライバを更新。`config.json` に
  `{"device": "cpu"}` を書けば CPU でも動く(`whisper_model` を `small.en` 程度に)
- **認識が途切れる/細切れ**: `silence_finalize` を 0.8〜1.0、`max_utterance` を 12〜18 に上げる
