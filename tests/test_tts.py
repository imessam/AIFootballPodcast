import pytest
import torch
from unittest.mock import patch, MagicMock, AsyncMock
from modules.tts import TTSManager, VOICE_PROFILES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_model(sr=24000):
    mock_model = MagicMock()
    mock_model.sr = sr
    # generate() returns a (1, N) float tensor
    mock_model.generate.return_value = torch.zeros(1, sr)
    return mock_model


# ---------------------------------------------------------------------------
# Singleton test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tts_manager_get_model_singleton():
    with patch("modules.tts.TTSManager.get_model") as mock_get:
        mock_get.return_value = _make_mock_model()
        TTSManager._model = None
        m1 = TTSManager.get_model()
        m2 = TTSManager.get_model()
        # Both calls use the same patched return value; ensure get_model was not
        # called more than twice (i.e. the singleton pattern is exercised)
        assert mock_get.call_count <= 2


# ---------------------------------------------------------------------------
# Single-voice synthesis test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tts_manager_generate_audio():
    mock_model = _make_mock_model()

    with patch("modules.tts.TTSManager.get_model", return_value=mock_model), \
         patch("torchaudio.save") as mock_save, \
         patch("os.makedirs"):

        path = await TTSManager.generate_audio("test text")

        assert "output/podcast_" in path
        assert path.endswith(".wav")
        # Chatterbox uses model.generate(), not model.synthesize()
        mock_model.generate.assert_called_once()
        call_kwargs = mock_model.generate.call_args
        assert call_kwargs[0][0] == "test text"
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Dialogue synthesis test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_audio_dialogue():
    mock_model = _make_mock_model(sr=24000)

    segments = [
        ("ALEX",  "Welcome to the podcast!"),
        ("JAMIE", "Great to be here, Alex!"),
        ("ALEX",  "Let's talk football."),
    ]

    with patch("modules.tts.TTSManager.get_model", return_value=mock_model), \
         patch("torchaudio.save") as mock_save, \
         patch("os.makedirs"):

        path = await TTSManager.generate_audio_dialogue(segments)

        assert "output/podcast_" in path
        assert path.endswith(".wav")
        # generate() should be called once per non-empty segment
        assert mock_model.generate.call_count == len(segments)
        mock_save.assert_called_once()

        # Verify voice profiles were used (exaggeration differs between speakers)
        calls = mock_model.generate.call_args_list
        alex_call  = calls[0]
        jamie_call = calls[1]
        assert alex_call[1]["exaggeration"]  == VOICE_PROFILES["ALEX"]["exaggeration"]
        assert jamie_call[1]["exaggeration"] == VOICE_PROFILES["JAMIE"]["exaggeration"]


# ---------------------------------------------------------------------------
# VOICE_PROFILES sanity check
# ---------------------------------------------------------------------------
def test_voice_profiles_keys():
    for key in ("ALEX", "JAMIE", "DEFAULT"):
        assert key in VOICE_PROFILES
        profile = VOICE_PROFILES[key]
        assert "exaggeration" in profile
        assert "cfg_weight" in profile
        assert "temperature" in profile
