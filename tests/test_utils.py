import os
import wave
import pytest
from modules.utils import wave_file

def test_wave_file_creation(tmp_path):
    test_file = tmp_path / "test.wav"
    pcm_data = b"\x00\x00" * 1000 # dummy PCM
    
    wave_file(str(test_file), pcm_data, channels=1, rate=24000, sample_width=2)
    
    assert os.path.exists(test_file)
    with wave.open(str(test_file), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 24000
        assert wf.getsampwidth() == 2
        assert wf.readframes(1000) == pcm_data
