import os
import requests

from datetime import datetime
from typing import Dict, Union
from dotenv import load_dotenv

from google.adk.tools.tool_context import ToolContext
from google.adk.tools import google_search  # Import the tool


load_dotenv()

print(f"FOOTBALL_DATA_API Key set: {'Yes' if os.environ.get('FOOTBALL_DATA_API_KEY') and os.environ['FOOTBALL_DATA_API_KEY'] != 'FOOTBALL_DATA_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")

def get_weather_stateful(city: str, tool_context: ToolContext) -> dict:
    """Retrieves weather, converts temp unit based on session state."""
    print(f"--- Tool: get_weather_stateful called for {city} ---")

    # --- Read preference from state ---
    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius") # Default to Celsius
    print(f"--- Tool: Reading state 'user_preference_temperature_unit': {preferred_unit} ---")

    city_normalized = city.lower().replace(" ", "")

    # Mock weather data (always stored in Celsius internally)
    mock_weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
    }

    if city_normalized in mock_weather_db:
        data = mock_weather_db[city_normalized]
        temp_c = data["temp_c"]
        condition = data["condition"]

        # Format temperature based on state preference
        if preferred_unit == "Fahrenheit":
            temp_value = (temp_c * 9/5) + 32 # Calculate Fahrenheit
            temp_unit = "°F"
        else: # Default to Celsius
            temp_value = temp_c
            temp_unit = "°C"

        report = f"The weather in {city.capitalize()} is {condition} with a temperature of {temp_value:.0f}{temp_unit}."
        result = {"status": "success", "report": report}
        print(f"--- Tool: Generated report in {preferred_unit}. Result: {result} ---")

        # Example of writing back to state (optional for this tool)
        tool_context.state["last_city_checked_stateful"] = city
        print(f"--- Tool: Updated state 'last_city_checked_stateful': {city} ---")

        return result
    else:
        # Handle city not found
        error_msg = f"Sorry, I don't have weather information for '{city}'."
        print(f"--- Tool: City '{city}' not found. ---")
        return {"status": "error", "error_message": error_msg}



def get_matches_by_date(date_str: str , tool_context: ToolContext) -> dict:

    """
        Fetches all matches for a given date from the Football Data API.

        Args:
            date_str (str): The date to fetch matches for in the format YYYY-MM-DD (e.g. "2023-05-31").
        Returns:
           dict: A dictionary with a status key set to "success" in case matches are found, "error" in case no matches are found. 
                In case of matches are found, the result will contain a dictionary with match ID as the key,
                and a dictionary containing the competition, home team, away team, home score and away score as the value if matches are found,
                or an error message if no matches are found.
    """ 

    agent_name = ""

    if tool_context is not None:
        agent_name = tool_context.agent_name

    tool_name = "get_matches_by_date"

    result: Dict[str, Union[str, Dict]] = {"status": "success"}

    print(f"--- Tool : {tool_name} called for date: {date_str} by agent: {agent_name} ---")

    date = datetime.now()
    current_date = datetime.now().date()
    
    if date_str is not None:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
    
    if date > current_date:
        return {"error": f"Date must not be in the future, provided date: {date}, current date: {current_date}"}
        
    date_formatted = date.strftime("%Y-%m-%d")

    print(f"--- Tool : {tool_name} Fetching matches for date: {date_formatted}, current date: {current_date}")

    api_key = os.getenv("FOOTBALL_DATA_API_KEY")

    uri = f"https://api.football-data.org/v4/matches/?date={date_formatted}"

    print(f"--- Tool : {tool_name} URI: {uri}")

    headers = { 'X-Auth-Token': api_key }

    response = requests.get(uri, headers=headers)

    response_json = response.json()

    error_code = response_json.get("errorCode", -1)

    if error_code > 0:

        result["status"] = "error"
        result["error"] = f"API error code : {error_code}, response: {response_json}"

        return result

    matches = response_json.get("matches", [])

    if len(matches) == 0:

        result["status"] = "error"
        result["error"] = f"No matches found for date: {date}"

        return result
    
    matches_dict = {}

    for match in matches:

        competition = match["competition"]["name"]

        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]

        score = match["score"]["fullTime"]

        home_score = score["home"]
        away_score = score["away"]

        matches_dict[str(match["id"])] = {
            "competition": competition,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score
        }

    print(f"--- Tool : {tool_name} Matches found: {matches_dict} ---")

    result["matches"] = matches_dict

    return result