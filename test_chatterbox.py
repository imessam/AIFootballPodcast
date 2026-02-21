from chatterbox import ChatterTTS

tts = ChatterTTS()
print("Available voices:", tts.list_voices())
# Try generating audio
audio = tts.synthesize("Hello world", voice=tts.list_voices()[0] if tts.list_voices() else None)
print(type(audio))
