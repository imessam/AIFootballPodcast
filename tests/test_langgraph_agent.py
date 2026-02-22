import pytest
from unittest.mock import patch, MagicMock
from modules.langgraph_agent import FootballPodcastAgent, AgentState
from modules.constants import DEFAULT_COMPETITIONS

@pytest.fixture
def agent():
    with patch("modules.langgraph_agent.ChatOpenAI"):
        return FootballPodcastAgent()

def test_fetch_matches_node(agent):
    mock_result = {"matches": [{"homeTeam": {"name": "Arsenal"}}], "status": "success"}
    with patch("modules.langgraph_agent.get_matches_by_date", return_value=mock_result) as mock_get:
        state: AgentState = {"query": "test", "matches": {}, "news": [], "script": "", "audio_path": "", "errors": []}
        result = agent.fetch_matches_node(state)
        assert result["matches"] == mock_result
        assert result["errors"] == []
        # Verify it's called with the constant
        mock_get.assert_called_once()
        args, _ = mock_get.call_args
        assert args[1] == DEFAULT_COMPETITIONS

def test_search_news_node(agent):
    state: AgentState = {
        "query": "test", 
        "matches": {"matches": [{"homeTeam": {"name": "Arsenal"}, "awayTeam": {"name": "Chelsea"}, "score": {"fullTime": {"home": 2, "away": 1}}}]}, 
        "news": [], "script": "", "audio_path": "", "errors": []
    }
    result = agent.search_news_node(state)
    assert "Arsenal vs Chelsea" in result["news"][0]
    assert "2-1" in result["news"][0]

def test_generate_script_node(agent):
    mock_llm_response = MagicMock()
    mock_llm_response.content = "<script>This is a test script</script>"
    agent.llm.invoke.return_value = mock_llm_response
    
    state: AgentState = {"query": "test", "matches": {}, "news": ["Match: A vs B. Result: 1-0."], "script": "", "audio_path": "", "errors": []}
    result = agent.generate_script_node(state)
    assert result["script"] == "This is a test script"

@pytest.mark.asyncio
async def test_tts_node(agent):
    with patch("modules.langgraph_agent.local_text_to_speech", return_value="output/test.wav"):
        state: AgentState = {"query": "test", "matches": {}, "news": [], "script": "hello", "audio_path": "", "errors": []}
        result = await agent.tts_node(state)
        assert result["audio_path"] == "output/test.wav"
