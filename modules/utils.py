# Decoupled utilities for LangGraph flow

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    
    """
    Saves PCM audio data to a WAV file.

    Args:
        filename (str): The name of the file to which the audio data will be saved.
        pcm (bytes): The PCM audio data to be saved.
        channels (int, optional): The number of audio channels. Defaults to 1.
        rate (int, optional): The sample rate (samples per second). Defaults to 24000.
        sample_width (int, optional): The sample width in bytes. Defaults to 2.
    """

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)