import os
import pytest
from datetime import datetime
from modules.tools import get_matches_by_date
from modules.constants import PREMIER_LEAGUE
from dotenv import load_dotenv

load_dotenv()

@pytest.mark.skipif(not os.getenv("FOOTBALL_DATA_API_KEY"), reason="API key not found")
def test_fetch_pl_matches():
    today = datetime.now().strftime("%Y-%m-%d")
    result = get_matches_by_date(today, [PREMIER_LEAGUE])
    
    if "status" in result and result["status"] == "error":
        # If API is down or key is invalid, this might happen, but for a unit test
        # we might want to mock it. For now, let's just assert success if it's meant to work.
        pytest.fail(f"API Error: {result.get('error')}")

    matches = result.get("matches", [])
    assert isinstance(matches, list)

    print(matches)
    
    for match in matches:
        competition = match.get('competition', {}).get('name', 'Unknown')
        assert competition == "Premier League"
