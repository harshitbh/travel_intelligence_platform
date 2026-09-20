# Singapore Travel Planning Assistant

This project is a local, retrieval-augmented travel assistant for Singapore. It combines a vector database of destination knowledge with live MCP-style tool calls for current travel information such as weather and currency conversion.

The application is built for a coursework assignment focused on:
- semantic retrieval from a travel knowledge base
- itinerary and attraction recommendations
- MCP tool integration for real-time data
- source-aware answer generation
- multi-turn conversational context

## What the app does

- Answers destination questions from local knowledge files in the `data/` folder
- Uses Chroma + embedding-based retrieval for travel information
- Detects user intent for weather and currency requests
- Calls external APIs for live weather and conversion data
- Combines RAG knowledge with tool results in a single response
- Tracks prior preferences such as budget, family travel, and activity style
- Prints compact source and usage traces for demo clarity

## Architecture overview

- `travel_assistant.py` handles the main orchestration, prompt flow, retrieval, tool use, and CLI/demo mode
- `external_tools.py` contains the weather and currency integrations and intent detection
- `knowledge_base_builder.py` preprocesses markdown files and creates the Chroma vector store
- `markdown_converter.py` can scrape and convert public sources into markdown content
- `data/` stores the prepared travel documents used by the knowledge base
- `models/chroma_db/` stores the persisted vector database

## Tech stack

- Python 3.10+
- LangChain + Chroma
- HuggingFace embeddings (`all-MiniLM-L6-v2`)
- OpenAI-compatible LM Studio backend
- `requests` for external API access
- `python-dotenv` for local environment configuration

## Project structure

```text
travel_intelligence_platform/
├── travel_assistant.py
├── external_tools.py
├── knowledge_base_builder.py
├── markdown_converter.py
├── requirements.txt
├── .env
├── data/
│   ├── itineraries_complete.md
│   ├── neighbourhoods_complete.md
│   ├── things_to_do_complete.md
│   └── wikivoyage_complete.md
├── models/
│   └── chroma_db/
├── README.md
└── .gitignore
```

## Environment setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

Create a `.env` file in the project root with the following values:

```env
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_MODEL=qwen2.5-7b-instruct
```

Notes:
- This project uses LM Studio as the LLM backend with an OpenAI-compatible API.
- The model name should match the model you have running in LM Studio.
- If you use a different local model or endpoint, update the values accordingly.

## Required local setup

Before running the app:

1. Start LM Studio locally
2. Load a compatible model (for example `qwen2.5-7b-instruct`)
3. Start the local OpenAI-compatible server on `http://localhost:1234/v1`

The app expects the LM Studio server to be running before it initializes the assistant.

## Build the knowledge base

Run:

```bash
python knowledge_base_builder.py
```

This script:
- reads the markdown files under `data/`
- cleans and chunks the content
- generates embeddings
- stores them in the Chroma persistence directory under `models/chroma_db/`

If the vector store is already present, this step can be skipped unless you want to rebuild it.

## Run the assistant

### Demo mode

```bash
python travel_assistant.py
```

This runs a set of demo queries covering:
- top attractions
- itinerary planning
- family-friendly travel advice
- weather-aware recommendations
- currency conversion

### Interactive CLI mode

```bash
python travel_assistant.py cli
```

This opens an interactive prompt loop where you can ask travel questions naturally.

## Example prompts

- What are the best attractions in Singapore?
- Which neighborhoods are best for food and shopping?
- Suggest a 3-day itinerary in Singapore for a family.
- What is the weather forecast for the next 3 days?
- Convert 60000 INR to SGD.
- I have a budget of 50000 INR. Convert it to SGD and suggest a 3-day itinerary.
- Which attractions in Singapore are suitable for young children?

## Prompt and retrieval strategy

The app uses a lightweight prompt strategy designed for assignment clarity:

- destination questions rely on retrieved vector-store context
- weather and currency requests trigger tool calls
- combined queries use both knowledge-base facts and current tool data
- the system keeps recent user history for context
- source references and chunk metadata are printed in the trace summary
- unsupported information is not presented as fact

## Data source notes

The project uses local travel documents as a knowledge base. These can be generated or refreshed from public sources using the scraper utility in `markdown_converter.py`.

The current project already includes knowledge files such as:
- `itineraries_complete.md`
- `neighbourhoods_complete.md`
- `things_to_do_complete.md`
- `wikivoyage_complete.md`

These were prepared to support Singapore-specific destination knowledge.

## Assignment alignment

This project addresses the core assignment requirements:

- knowledge base from travel resources: implemented
- embedding-based semantic retrieval: implemented
- weather via MCP/tool integration: implemented
- currency conversion via MCP/tool integration: implemented
- at least one combined RAG + tool response: implemented
- multi-turn conversation context: implemented
- source-aware answer flow: implemented
- simple usable interface: implemented

## Troubleshooting

### LM Studio connection fails

Check:
- LM Studio is running
- the local server is available at `http://localhost:1234/v1`
- the model name in `.env` matches the loaded model

### Chroma/vector database issues

If retrieval fails or the app reports no vector store found:

```bash
python knowledge_base_builder.py
```

### Weather or currency tool errors

The app has fallback logic, but if the external APIs are unavailable then tool output may fail gracefully instead of inventing data.

## Notes

This project is designed as a practical Singapore travel assistant and is suitable for a course assignment focused on RAG, MCP/tool-driven current info, and grounded travel recommendations.