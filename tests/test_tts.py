import pytest
from unittest.mock import patch, MagicMock
from modules.tts import TTSManager

@pytest.mark.asyncio
async def test_tts_manager_get_model_singleton():
    # Mock the entire qwen_tts.models module to avoid import issues
    mock_qwen_tts = MagicMock()
    mock_models = MagicMock()
    mock_qwen3tts = MagicMock()
    
    with patch.dict("sys.modules", {
        "qwen_tts": mock_qwen_tts,
        "qwen_tts.models": mock_models,
        "torch": MagicMock()
    }):
        mock_models.Qwen3TTS = mock_qwen3tts
        mock_qwen3tts.from_pretrained.return_value = MagicMock()
        
        # Reset singleton state for test
        TTSManager._model = None
        
        model1 = TTSManager.get_model()
        model2 = TTSManager.get_model()
        
        assert model1 == model2
        mock_qwen3tts.from_pretrained.assert_called_once()

@pytest.mark.asyncio
async def test_tts_manager_generate_audio():
    with patch("modules.tts.TTSManager.get_model") as mock_get_model, \
         patch("torchaudio.save") as mock_save, \
         patch("os.path.exists", return_value=True), \
         patch("os.makedirs"):
        
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_audio_tensor = MagicMock()
        mock_model.synthesize.return_value = mock_audio_tensor
        
        path = await TTSManager.generate_audio("test text")
        
        assert "output/podcast_" in path
        assert path.endswith(".wav")
        mock_model.synthesize.assert_called_once_with("test text")
        mock_save.assert_called_once()
