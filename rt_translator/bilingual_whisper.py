"""日英の判定と認識で、同じ Whisper encoder 出力を使う。

faster-whisper 1.2.1 の language=None は判定後に再度 encode するため、
アプリが切り出した短い発話を generate_segments の入口で判定する。
multilingual=True を指定した呼び出しだけ日英判定を行う。モデルを共有する
日本語固定のマイクには影響せず、判定結果も呼び出しローカルに保持する。
"""
from dataclasses import replace

from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim


class BilingualWhisperModel(WhisperModel):
    def generate_segments(self, features, tokenizer, options, log_progress,
                          encoder_output=None):
        if not options.multilingual:
            yield from super().generate_segments(
                features, tokenizer, options, log_progress, encoder_output)
            return

        # この経路はアプリの発話バッファ（既定最大12秒、clipなし）専用。
        frames = min(features.shape[-1] - 1, self.feature_extractor.nb_max_frames)
        if encoder_output is None:
            encoder_output = self.encode(pad_or_trim(features[:, :frames]))
        probabilities = self.model.detect_language(encoder_output)[0]
        language_token, _ = max(
            ((token, probability) for token, probability in probabilities
             if token in {"<|en|>", "<|ja|>"}),
            key=lambda item: item[1],
        )
        language = language_token[2:-2]
        tokenizer.language = tokenizer.tokenizer.token_to_id(language_token)
        tokenizer.language_code = language
        # 英語用の hotwords で日本語認識を引っ張らない。日本語の辞書置換は
        # 認識後に行う。基底クラスによる再判定もここでは不要。
        options = replace(options, multilingual=False,
                          hotwords=options.hotwords if language == "en" else None)
        for segment in super().generate_segments(
                features, tokenizer, options, log_progress, encoder_output):
            segment.language = language
            yield segment
