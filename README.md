# AIFootballPodcast

**AIFootballPodcast** is an AI-powered pipeline that automatically generates football (soccer) podcast episodes. It fetches live match data from [Football-Data.org](https://www.football-data.org/), searches for related news via DuckDuckGo, generates an engaging podcast script using a local LLM (via Ollama), and converts the script to speech using [Chatterbox TTS](https://github.com/chatterbox-tts/chatterbox).

The project is part of [Google's ADK Hackathon](https://googlecloudmultiagents.devpost.com/).

---

## Agent Pipeline Diagram

![AIFootballPodcastDiagram](assets/diagram.png)

The pipeline is orchestrated as a **LangGraph** state machine with four sequential nodes:

```
fetch_matches → search_news → generate_script → tts
```

| Node | Description |
|---|---|
| `fetch_matches` | Fetches today's matches from Football-Data.org for configured competitions |
| `search_news` | Searches DuckDuckGo for recent news snippets for each match |
| `generate_script` | Prompts a local LLM to write a conversational podcast script |
| `tts` | Synthesizes the script into a `.wav` audio file using Chatterbox TTS |

---

## Demo

[![FootballPodcastDemo](https://img.youtube.com/vi/hOF1Z-7p4qY/0.jpg)](https://www.youtube.com/watch?v=hOF1Z-7p4qY)

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | Local model via [Ollama](https://ollama.com/) (e.g. `qwen3:0.6b`) with `langchain-openai` |
| Web search | [duckduckgo-search](https://pypi.org/project/duckduckgo-search/) |
| Football data | [Football-Data.org API v4](https://www.football-data.org/) |
| Text-to-speech | [Chatterbox TTS](https://github.com/chatterbox-tts/chatterbox) (CUDA / CPU) |
| Python | ≥ 3.13 |

---

## Project Structure

```
AIFootballPodcast/
├── modules/
│   ├── langgraph_agent.py  # LangGraph state machine & node definitions
│   ├── tools.py            # Football-Data API client & DuckDuckGo search helper
│   ├── tts.py              # ChatterboxTTS singleton manager (async, CUDA/CPU)
│   ├── constants.py        # Default competitions (Premier League, etc.)
│   └── utils.py            # Shared utility helpers
├── output/                 # Generated .wav podcast files (timestamped)
├── tests/                  # Unit & integration tests (pytest)
├── run_local.py            # CLI entry point
├── pyproject.toml          # Project metadata & dependencies (uv / pip)
├── requirements.txt        # Pip-compatible dependency list
└── .env                    # API keys & model configuration (not committed)
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/imessam/AIFootballPodcast.git
cd AIFootballPodcast
```

### 2. Install dependencies

**With `uv` (recommended):**
```bash
uv sync
```

**With `pip`:**
```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama

Download [Ollama](https://ollama.com/) and pull the model you want to use:

```bash
ollama pull qwen3:0.6b
```

### 4. Configure environment variables

Create a `.env` file in the project root (or export the variables in your shell):

```bash
# Required: Football-Data.org API key (free tier available)
FOOTBALL_DATA_API_KEY="your_football_data_api_key"

# Optional: Override defaults if using a different local model server
LOCAL_OPENAI_BASE_URL="http://localhost:11434/v1"   # default
LOCAL_MODEL_NAME="qwen3:0.6b"                       # default
```

---

## Usage

```bash
# Default: today's football highlights
python run_local.py

# Podcast about a specific match
python run_local.py "Ahly vs Real Madrid"

# Podcast for matches from a specific date
python run_local.py "2025-05-31"
```

Generated audio is saved to `output/podcast_<YYYYMMDD_HHMMSS>.wav`.

---

## Running Tests

```bash
pytest
# or with uv:
uv run pytest
```

---

## Known Issues

- The TTS step occasionally fails because the LLM does not produce a `<script>…</script>` block. A fallback regex is applied, but empty scripts will be caught and reported in `state["errors"]`.
- Chatterbox TTS requires a GPU for fast inference; CPU synthesis is supported but significantly slower.

---

## TODO

- [ ] Support additional competitions beyond the Premier League
- [ ] Support other LLMs (OpenAI, Gemini, etc.)
- [ ] Support multiple output languages
- [ ] Longer, more detailed podcast scripts
- [ ] Multiple speaker voices and characters
- [ ] Generate animated video podcasts

---

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).

---

## Acknowledgements

- [Chatterbox TTS](https://github.com/chatterbox-tts/chatterbox) for local text-to-speech synthesis
- [Ollama](https://ollama.com/) for local LLM serving
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [Football-Data.org](https://www.football-data.org/) for football match data
- [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) for free web search
