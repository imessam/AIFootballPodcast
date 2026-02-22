import os
import sys
import requests

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from datetime import datetime
from typing import Dict, Optional, Union
from dotenv import load_dotenv

import asyncio
import logging
from duckduckgo_search import DDGS
from modules.utils import wave_file


load_dotenv()

print(f"SPORT_DEV_KEY Key set: {'Yes' if os.environ.get('SPORT_DEV_KEY') and os.environ['SPORT_DEV_KEY'] != 'SPORT_DEV_KEY' else 'No (REPLACE PLACEHOLDER!)'}")


def get_matches_by_date(date_str: str, leagues_id: list) -> dict:

    """
        Fetches all matches for a given date from the Sportdevs API.

        Args:
            date_str (str): The date to fetch matches for in the format YYYY-MM-DD (e.g. "2023-05-31").
        Returns:
           dict: A dictionary with a status key set to "success" in case matches are found, "error" in case no matches are found. 
                In case of matches are found, the result will contain a dictionary with match ID as the key,
                and a dictionary containing the competition, home team, away team, home score and away score as the value if matches are found,
                or an error message if no matches are found.
    """ 
    tool_name = "get_matches_by_date"

    result: Dict[str, Union[str, Dict]] = {"status": "success"}

    print(f"--- Tool : {tool_name} called for date: {date_str} ---")

    date = datetime.now()
    current_date = datetime.now().date()
    
    if date_str is not None:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
    
    if date > current_date:
        return {"error": f"Date must not be in the future, provided date: {date}, current date: {current_date}"}
        
    date_formatted = date.strftime("%Y-%m-%d")
    from datetime import timedelta
    date_to = (date + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"--- Tool : {tool_name} Fetching matches for date: {date_formatted}, current date: {current_date}")

    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    # Football-Data.org uri for matches
    competitions = ""
    if leagues_id:
        competitions = "&competitions=" + ",".join(map(str, leagues_id))
    
    uri = f"https://api.football-data.org/v4/matches?dateFrom={date_formatted}&dateTo={date_to}{competitions}"

    print(f"--- Tool : {tool_name} URI: {uri} ---")

    headers = { 'X-Auth-Token': f"{api_key}" }

    response = requests.get(uri, headers=headers)
    
    if response.status_code != 200:
        print(f"--- Tool : {tool_name} Error: API returned status {response.status_code} ---")
        print(f"--- Raw Error Response: {response.text} ---")
        return {"status": "error", "error": f"API returned status {response.status_code}", "matches": {}}

    try:
        response_json = response.json()
        print(f"--- Tool : {tool_name} Raw API Response Successfully Logged ---")
        # Log limited version of the response to avoid cluttering but show structure
        # print(f"--- Raw JSON: {response_json} ---") 
    except Exception as e:
        print(f"--- Tool : {tool_name} Error parsing JSON: {e} ---")
        return {"status": "error", "error": "Invalid JSON response", "matches": {}}

    return response_json


def search_football_news(query: str, max_results: int = 3) -> str:
    """
    Searches the web for recent news related to a specific football query using DuckDuckGo.
    
    Args:
        query (str): The search term (e.g. "Real Madrid vs Barcelona news").
        max_results (int): The maximum number of news snippets to return.
        
    Returns:
        str: A concatenated string of news snippets found, or an empty string if none.
    """
    tool_name = "search_football_news"
    print(f"--- Tool : {tool_name} called for query: {query} ---")
    
    try:
        results = []
        with DDGS() as ddgs:
            # First try the news endpoint
            ddgs_news = list(ddgs.news(query, max_results=max_results))
            if ddgs_news:
                for r in ddgs_news:
                    results.append(f"- {r.get('title', '')}: {r.get('body', '')}")
            else:
                # Fallback to general text search
                ddgs_text = list(ddgs.text(query, max_results=max_results))
                for r in ddgs_text:
                    results.append(f"- {r.get('title', '')}: {r.get('body', '')}")
                    
        print(f"--- Tool : {tool_name} found {len(results)} results ---")
        return "\n".join(results)
    except Exception as e:
        print(f"--- Tool : {tool_name} Error searching web: {e} ---")
        return f"Error fetching news: {str(e)}"


from .tts import TTSManager

async def local_text_to_speech(text: str, speaker_name: str = "en-US-ChristopherNeural") -> str:
    """
    Legacy wrapper for TTSManager to maintain compatibility with existing nodes.
    """
    return await TTSManager.generate_audio(text)



if __name__ == "__main__":
    import asyncio
    # Simple test for local tts
    async def test():
        await local_text_to_speech("Testing local podcast generation.")
    asyncio.run(test())