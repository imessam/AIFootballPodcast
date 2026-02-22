
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# English example
model = ChatterboxTTS.from_pretrained(device="cuda")

text = "What's up guys, welcome back to our podcast, today we're gonna talk about the latest football newwwwws."
wav = model.generate(text)
ta.save("test-english.wav", wav, model.sr)

# Multilingual examples
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

arabic_text = "ايه يا باشا عامل ايه النهارده في ماتشات الاهلي والزمالك"
wav_arabic = multilingual_model.generate(arabic_text, language_id="ar")
ta.save("test-arabic.wav", wav_arabic, model.sr)
