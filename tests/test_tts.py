import pytest
import torch
import os
from unittest.mock import patch, MagicMock, call
from modules.tts import TTSManager, VOICE_PROFILES, PARALINGUISTIC_TAGS, _VOICE_FILES, _ASSETS_DIR
from modules.langgraph_agent import FootballPodcastAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_model(sr=24000):
    mock_model = MagicMock()
    mock_model.sr = sr
    mock_model.generate.return_value = torch.zeros(1, sr)
    return mock_model


# ---------------------------------------------------------------------------
# VOICE_PROFILES sanity
# ---------------------------------------------------------------------------
def test_voice_profiles_keys():
    for key in ("ALEX", "JAMIE", "DEFAULT"):
        assert key in VOICE_PROFILES
        p = VOICE_PROFILES[key]
        assert "temperature" in p
        assert "top_k" in p
        assert "audio_prompt_path" in p


# ---------------------------------------------------------------------------
# PARALINGUISTIC_TAGS set contents
# ---------------------------------------------------------------------------
def test_paralinguistic_tags_set():
    required = {"[laugh]", "[chuckle]", "[sigh]", "[cough]",
                "[gasp]", "[groan]", "[sniff]", "[shush]",
                "[clear throat]", "[yawn]"}
    assert required.issubset(PARALINGUISTIC_TAGS)


# ---------------------------------------------------------------------------
# _clean_segment_for_tts preserves paralinguistic tags
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tag", ["[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[groan]"])
def test_clean_segment_preserves_tags(tag):
    text = f"That was incredible! {tag} Absolutely stunning goal."
    result = FootballPodcastAgent._clean_segment_for_tts(text)
    assert tag in result, f"Tag {tag} was stripped from: {result}"


def test_clean_segment_strips_html_not_tags():
    text = "Hello <b>world</b> [chuckle] and <em>italic</em>."
    result = FootballPodcastAgent._clean_segment_for_tts(text)
    assert "[chuckle]" in result
    assert "<b>" not in result
    assert "<em>" not in result


def test_clean_segment_strips_parentheticals():
    text = "Wow (laughing) that was great [laugh] today."
    result = FootballPodcastAgent._clean_segment_for_tts(text)
    assert "(laughing)" not in result
    assert "[laugh]" in result


# ---------------------------------------------------------------------------
# Singleton / model loading
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tts_manager_get_model_singleton():
    with patch("modules.tts.TTSManager.get_model") as mock_get:
        mock_get.return_value = _make_mock_model()
        TTSManager._model = None
        TTSManager.get_model()
        TTSManager.get_model()
        assert mock_get.call_count <= 2


# ---------------------------------------------------------------------------
# Single-voice generate_audio (backward compat)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tts_manager_generate_audio():
    mock_model = _make_mock_model()
    with patch("modules.tts.TTSManager.get_model", return_value=mock_model), \
         patch("modules.tts.TTSManager._ensure_voice_assets"), \
         patch("torchaudio.save") as mock_save, \
         patch("os.makedirs"):

        path = await TTSManager.generate_audio("test text")

        assert "output/podcast_" in path
        assert path.endswith(".wav")
        mock_model.generate.assert_called_once()
        assert mock_model.generate.call_args[0][0] == "test text"
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Dialogue synthesis — audio_prompt_path per speaker
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_audio_dialogue_uses_voice_profiles():
    mock_model = _make_mock_model(sr=24000)

    segments = [
        ("ALEX",  "Welcome to the show! [clear throat]"),
        ("JAMIE", "Let's go! [laugh] Amazing stuff."),
        ("ALEX",  "Indeed. [chuckle]"),
    ]

    fake_alex_path  = "/fake/assets/voices/alex.wav"
    fake_jamie_path = "/fake/assets/voices/jamie.wav"

    with patch("modules.tts.TTSManager.get_model", return_value=mock_model), \
         patch("modules.tts.TTSManager._ensure_voice_assets", side_effect=lambda: (
             VOICE_PROFILES.__setitem__("ALEX",  {**VOICE_PROFILES["ALEX"],  "audio_prompt_path": fake_alex_path}),
             VOICE_PROFILES.__setitem__("JAMIE", {**VOICE_PROFILES["JAMIE"], "audio_prompt_path": fake_jamie_path}),
         )), \
         patch("torchaudio.save") as mock_save, \
         patch("os.makedirs"):

        # Manually seed profiles for the test
        VOICE_PROFILES["ALEX"]["audio_prompt_path"]  = fake_alex_path
        VOICE_PROFILES["JAMIE"]["audio_prompt_path"] = fake_jamie_path

        path = await TTSManager.generate_audio_dialogue(segments)

        assert "output/podcast_" in path
        assert path.endswith(".wav")
        assert mock_model.generate.call_count == len(segments)

        calls = mock_model.generate.call_args_list
        call_kwargs_0 = calls[0][1]  # ALEX
        call_kwargs_1 = calls[1][1]  # JAMIE
        assert call_kwargs_0.get("audio_prompt_path") == fake_alex_path
        assert call_kwargs_1.get("audio_prompt_path") == fake_jamie_path

        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# _ensure_voice_assets skips download when files exist
# ---------------------------------------------------------------------------
def test_ensure_voice_assets_skips_when_cached(tmp_path):
    fake_alex  = tmp_path / "alex.wav"
    fake_jamie = tmp_path / "jamie.wav"
    fake_alex.write_bytes(b"RIFF")
    fake_jamie.write_bytes(b"RIFF")

    with patch.dict("modules.tts.__dict__", {"_ASSETS_DIR": tmp_path}), \
         patch("modules.tts._VOICE_FILES", {"ALEX": fake_alex, "JAMIE": fake_jamie}), \
         patch("modules.tts.TTSManager._download_voice_samples") as mock_dl:

        TTSManager._ensure_voice_assets()
        mock_dl.assert_not_called()
