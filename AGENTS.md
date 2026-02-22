# AI Development Guidelines for AIFootballPodcast

## 1. Project Overview
AIFootballPodcast automates the creation of football (soccer) podcast episodes. It retrieves match data from `api.football-data.org`, gathers real-time scores, optionally gathers news, utilizes a Large Language Model (LLM) to craft engaging podcast scripts, and synthesizes speech using TTS (Text-to-Speech) engines.

## 2. Architecture & Tech Stack
- **Agent Orchestration**: LangGraph (`modules/langgraph_agent.py`) coordinates the stateful workflow.
- **LLM Integrations**: 
  - Local Models (e.g., Qwen run via Ollama or custom local server: `http://localhost:11434/v1`) using `langchain_openai`.
- **TTS**: Local TTS using Chatterbox TTS.
- **Workflow Pipeline**:
  1. `fetch_matches`: Uses `get_matches_by_date` using the `FOOTBALL_DATA_API_KEY`.
  2. `search_news`: Creates textual context from matches.
  3. `generate_script`: Uses the LLM to structure a podcast script (expected to be wrapped in `<script>...</script>` tags).
  4. `tts`: Converts script to audio.
  
## 3. Directory Structure
- `modules/` - Core implementation:
  - `langgraph_agent.py`: LangGraph state and execution nodes.
  - `tools.py`: Helper functions, external API request wrappers.
  - `tts.py`: Audio generation manager.
  - `constants.py`: Configurations (e.g., fallback competitions).
- `output/` - Contains the final rendered audio files (`.wav`).


## 4. Development Guidelines for AI Agents
When generating code or refactoring this repository, AI agents MUST follow these instructions:

### A. Web Search & External APIs
- **Web Search**: For any web search functionality (e.g., finding the latest football news or match contexts), AI agents MUST prioritize using free web search tools or libraries (such as `duckduckgo-search` or Wikipedia APIs) over paid APIs.

### B. Environment & Storage
- **Secrets Management**: Never hardcode API keys. Rely on `os.getenv` or `dotenv`. Ensure `.env` is loaded before instantiating API clients.
- **File Output**: Synthesized artifacts should exclusively be deposited within the `output/` directory, using coherent naming patterns (ideally timestamped/unique keys to prevent accidental overwrites).

### B. Agent State & Graph Flow
- **State Integrity**: `AgentState(TypedDict)` represents the single source of truth across steps. If introducing new logic (like translation or fact-checking), define corresponding TypedDict keys and ensure default empty states are handled.
- **Resiliency**: If a node fails to fetch external data (e.g., Football API rate limits), it should append an error message to `state["errors"]` rather than unceremoniously throwing exceptions crashing the graph runner.

### C. Async & Performance
- **Asynchronous Execution**: The TTS capabilities and graph invocations (`ainvoke`) run asynchronously. Adhere to `async/await` conventions for external requests wherever possible to improve parallel operations.

### D. Observability
- Follow the established debug logging format (`print(f"--- [NodeName] Action Description ---")`) across tools and nodes. This guarantees easy traceability in CI logs or local execution traces.
