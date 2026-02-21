import pytest
from unittest.mock import patch, MagicMock
from modules.tools import get_matches_by_date, local_text_to_speech
from modules.constants import PREMIER_LEAGUE

def test_get_matches_by_date_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"matches": [{"id": 1, "competition": {"name": "Premier League"}}]}
    
    with patch("requests.get", return_value=mock_response):
        result = get_matches_by_date("2024-05-21", [PREMIER_LEAGUE])
        assert result["matches"][0]["id"] == 1
        assert result["matches"][0]["competition"]["name"] == "Premier League"

def test_get_matches_by_date_future():
    # Today's date + 1 day
    from datetime import datetime, timedelta
    future_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    result = get_matches_by_date(future_date, [PREMIER_LEAGUE])
    assert "error" in result
    assert "future" in result["error"]

def test_get_matches_by_date_api_error():
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not found"
    
    with patch("requests.get", return_value=mock_response):
        result = get_matches_by_date("2024-05-21", [PREMIER_LEAGUE])
        assert result["status"] == "error"
        assert "404" in result["error"]

@pytest.mark.asyncio
async def test_local_text_to_speech():
    with patch("modules.tools.TTSManager.generate_audio", return_value="test_output.wav") as mock_gen:
        path = await local_text_to_speech("hello")
        assert path == "test_output.wav"
        mock_gen.assert_called_once_with("hello")
