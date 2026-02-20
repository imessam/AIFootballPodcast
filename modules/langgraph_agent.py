import os
from typing import Annotated, TypedDict, List, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from modules.tools import get_matches_by_date, local_text_to_speech
from modules.utils import wave_file

# Define the State
class AgentState(TypedDict):
    query: str
    matches: Dict[str, Any]
    news: List[str]
    script: str
    audio_path: str
    errors: List[str]

# Node 1: Fetch Matches
def fetch_matches_node(state: AgentState):
    print("--- Node: fetch_matches_node ---")
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        # leagues_id [27] as a default for now based on original tools.py
        result = get_matches_by_date(today, [27])
        
        # LOGGING: Raw API Response
        print("\n[LOG] --- RAW MATCHES API RESPONSE ---")
        print(result)
        print("--------------------------------------\n")
        
        # Football-Data.org returns matches in a 'matches' key
        matches_list = result.get("matches", [])
        print(f"--- Matches Fetched: {len(matches_list)} ---")
        
        # Normalize to the dictionary format expected by search_news_node if needed
        # Or I'll just update search_news_node to handle the list directly.
        return {"matches": result, "errors": []}
    except Exception as e:
        print(f"--- Error in fetch_matches_node: {e} ---")
        return {"errors": [f"Error fetching matches: {str(e)}"]}

# Node 2: Search Web for News
def search_news_node(state: AgentState):
    print("--- Node: search_news_node ---")
    matches_data = state.get("matches", {})
    matches_list = matches_data.get("matches", [])
    
    if not matches_list:
        return {"news": ["No matches found today."]}
    
    all_news = []
    for match in matches_list:
        home = match.get('homeTeam', {}).get('name', 'Unknown')
        away = match.get('awayTeam', {}).get('name', 'Unknown')
        score = f"{match.get('score', {}).get('fullTime', {}).get('home', '?')}-{match.get('score', {}).get('fullTime', {}).get('away', '?')}"
        all_news.append(f"Match: {home} vs {away}. Result: {score}.")
    
    return {"news": all_news}

# Initialize the LLM once
def get_llm():
    local_base_url = os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:11434/v1")
    local_model = os.getenv("LOCAL_MODEL_NAME", "qwen3:0.6b")
    return ChatOpenAI(
        model=local_model,
        openai_api_key="none",
        base_url=local_base_url,
    )

llm_instance = get_llm()

# Node 3: Generate Script
def generate_script_node(state: AgentState):
    print("--- Node: generate_script_node ---")
    
    context = "\n".join(state.get("news", []))
    print("\n[LOG] --- CONTEXT SENT TO LLM ---")
    print(context[:1000] + ("..." if len(context) > 1000 else "")) # Truncate if too long
    print("--------------------------------\n")
    
    prompt = f"""
    Write a short, exciting podcast script summarizing today's football matches.
    If matches are listed below, summarize the results or upcoming fixtures.
    If no matches are listed, mention that it's a quiet day in football.
    
    Matches Info:
    {context}
    
    The script should be conversational and professional. 
    Wrap the final script in <script> tags.
    """
    
    messages = [
        SystemMessage(content="You are a football podcast writer. You provide concise match summaries."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = llm_instance.invoke(messages)
        content = response.content
        
        # LOGGING: Raw LLM Response
        print("\n[LOG] --- RAW LLM RESPONSE ---")
        print(content)
        print("-----------------------------\n")
        
        # Extract script between <script> tags if present, else strip <think> tags
        import re
        script_match = re.search(r'<script>(.*?)</script>', content, re.DOTALL)
        if script_match:
            final_script = script_match.group(1).strip()
        else:
            # More robust: remove <think>...</think> OR everything after <think> if not closed
            # This is common with small models that might time out or fail to close tags
            final_script = re.sub(r'<think>.*?(?:</think>|$)', '', content, flags=re.DOTALL).strip()
            # Also remove any leftover <script> tags just in case
            final_script = re.sub(r'</?script>', '', final_script).strip()
        
        print("\n[LOG] --- FINAL PODCAST SCRIPT ---")
        print(final_script)
        print("---------------------------------\n")
        
        if not final_script:
            print("--- Warning: Final script is empty after cleaning! Using raw content as fallback. ---")
            final_script = content
        
        print(f"--- Script Generated Successfully ---")
        return {"script": final_script}
    except Exception as e:
        print(f"--- Error in generate_script_node: {e} ---")
        return {"script": "", "errors": state.get("errors", []) + [f"Error generating script: {str(e)}"]}

# Node 4: Text to Speech
async def tts_node(state: AgentState):
    print("--- Node: tts_node ---")
    script = state.get("script", "")
    if not script:
        return {"errors": ["No script available for TTS."]}
    
    try:
        audio_path = await local_text_to_speech(script)
        return {"audio_path": audio_path}
    except Exception as e:
        return {"errors": [f"Error in local TTS: {str(e)}"]}

# Build the Graph
def create_podcast_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("fetch_matches", fetch_matches_node)
    workflow.add_node("search_news", search_news_node)
    workflow.add_node("generate_script", generate_script_node)
    workflow.add_node("tts", tts_node)

    workflow.add_edge(START, "fetch_matches")
    workflow.add_edge("fetch_matches", "search_news")
    workflow.add_edge("search_news", "generate_script")
    workflow.add_edge("generate_script", "tts")
    workflow.add_edge("tts", END)

    return workflow.compile()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    import asyncio
    graph = create_podcast_graph()
    
    async def run_example():
        # Using a default query
        async for event in graph.astream({"query": "Today's football matches"}):
            print(event)

    asyncio.run(run_example())
