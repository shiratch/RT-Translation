# Hy-MT2とTranslateGemmaの比較

2026-09-20、RTX 5090 / ドライバ616.92 / llama.cpp b10964 CUDA 13.3。
Hy-MT2版は `run_hymt_en.bat`。比較はローカルの作成例文と合成音声で行い、実会議の音声は使っていない。

## 構成と条件

- [Hy-MT2 30B-A3B](https://huggingface.co/tencent/Hy-MT2-30B-A3B) は日本語対応のテキスト翻訳モデル。
- [Tencent公式GGUF](https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF) のQ4_K_Mを使用。18236702880バイト。
- リビジョン `fd3dcbb6b31e9e03923ef4f9f42500f74c3851c3`、ファイル `Hy-MT2-30B-A3B-Q4_K_M.gguf`。
- モデル付属chat templateと公式の英語による日本語翻訳指示を使用。既存の実行環境で全49層をGPUに配置できた。
- 字幕の揺れを抑え、比較の再現性を高めるため温度は0。公式推奨の0.7とは異なる。
- `top_p=1.0`、`top_k=-1`、反復ペナルティ1.0。生成上限256トークン、コンテキスト2048。
- 音声認識は全構成でWhisper large-v3-turbo FP16、2ワーカー、確定beam 5、暫定間隔0.4秒。

量子化形式が異なるため、同一精度の重みでモデルの能力だけを比較する実験ではない。
このPCで実際に使う構成同士の比較として扱う。

## 翻訳単体の速度とメモリ

同じ会議例文8件を各3回翻訳。各例文の中央値を取り、その8件の中央値を比較した。
音声の入力待ち・認識・UI描画は含まない。ストリームで先頭4文字以上が届く時刻も測定した。

| 指標 | Gemma 12B Q8_0 | Gemma 27B Q6_K | Hy-MT2 30B-A3B Q4_K_M |
|---|---:|---:|---:|
| 翻訳開始から先頭部分の受信 | 73.66ms | 106.17ms | 31.63ms |
| 翻訳完了 | 275.84ms | 374.17ms | 111.25ms |
| 例文別の翻訳完了時間の範囲 | 196〜430ms | 301〜724ms | 77〜189ms |
| 音声認識2本との同時実行後のGPU使用量 | 18534MiB | 27965MiB | 23590MiB |

GPU使用量はデスクトップ等も含む全体値。Hy-MT2は約23.0GiBで、実行後も所有サーバーが終了することを確認した。
音声認識2本は英語の相手音声と日本語のマイク音声を想定した合成音声で、翻訳と同時に実行した。

## 音声からの字幕表示

約9.9秒の合成英語音声を100msずつ実時間で投入し、認識・辞書・話者検出・翻訳を通して各2回測定した。
音声入力開始から最初のグレー字幕がUIキューへ届くまでの時間:

| 構成 | 最初のグレー字幕 |
|---|---:|
| Gemma 12B | 0.641〜0.643秒 |
| Gemma 27B | 0.653〜0.672秒 |
| Hy-MT2 | 0.603〜0.677秒 |

Hy-MT2でも3文すべてが確定した。翻訳単体は速くなったが、音声認識の待ち時間もあるため、
この短い音声で最初の字幕が出るまでの時間は大きく変わらなかった。実デバイスとUI描画の時間は含まない。

## 訳文の比較

以下は辞書や句読点の後処理前の出力。

| 英語 | Gemma 27B | Hy-MT2 |
|---|---|---|
| The Facebook schedule will be updated after the integration tests pass. | 統合テストが完了次第、Facebookのスケジュールを更新します。 | 統合テストに合格した後、Facebookのスケジュールが更新されます。 |
| We are not ruling it out, but don't count on it yet. | 可能性は否定しませんが、まだ確実とは言えません。 | 可能性を否定はしませんが、まだ期待はしないでください。 |
| Could you walk me through the trade-offs of rolling back this change? | この変更を元に戻すことによるメリットとデメリットについて、詳しく説明していただけますか？ | この変更をロールバックする際のトレードオフについて、詳しく説明していただけますか？ |

Hy-MT2はテストの「合格」条件を保持した。専門用語はカタカナを使う傾向が見られ、Gemmaの説明的な訳と好みが分かれる。
一方、`more than 200 milliseconds` はHy-MT2が「200ミリ秒以上」、Gemma 27Bが「200ミリ秒を超えて」と訳した。
厳密な境界条件は後者が正確であり、この8件だけで総合的な品質の優劣は決められない。

## 検証と再実行

モデル切替、テンプレート、接続切れ、生成長による途中停止を含む回帰テスト36件に合格。
共通化した実行部分について、Gemma 12B/27Bの通常応答とストリーム応答も実機で再確認した。

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:HF_HUB_OFFLINE='1'
.venv\Scripts\python benchmark-results/compare_gemma.py --profile hymt2 --stream
.venv\Scripts\python benchmark-results/partial_latency.py --profile hymt2 --output benchmark-results/partial-latency-hymt2.json
```

生の結果は `benchmark-results/comparison-hymt2-stream.json` と `partial-latency-hymt2.json`。
比較元は `comparison-translategemma12b-stream.json` と `comparison-translategemma27b-stream.json`。
比較スクリプト・合成音声・結果はローカルの `benchmark-results/` にあり、Git管理対象外。
