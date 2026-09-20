# Singapore Travel Planning Assistant

GitHub Repository: https://github.com/harshitbh/travel_intelligence_platform.git

This project is a local, travel-focused Retrieval-Augmented Generation (RAG) system for Singapore. It combines a locally stored travel knowledge base with live external tools for current information such as weather and currency conversion.

The system is designed around a simple but important idea: travel planning requires both destination facts and time-sensitive current information. The assistant retrieves relevant destination knowledge from local documents and uses live tool responses when the user asks about current weather, money conversion, or travel conditions.

The project is built for a coursework assignment that requires:
- a document-based travel knowledge base
- embedding-based retrieval
- a local or hosted language model
- MCP-style tool integration
- grounded responses with source awareness
- clear handling of current travel data and destination guidance

---

## Project goal

The goal of this project is to help a user plan a trip to Singapore by combining:
- stable knowledge about attractions, neighborhoods, activities, transport, and local culture
- current information such as weather and currency conversion
- a conversational assistant that remembers user preferences across the session

This gives the user a more practical travel planner rather than a general-purpose chatbot.

---

## Full workflow: from data to final answer

### 1. Data collection and source preparation

The project starts with travel source material for Singapore. These documents are stored under `data/` and include content related to:
- attractions
- neighborhoods
- itineraries
- cultural and practical travel guidance
- places to visit in Singapore

The scraper utility in `markdown_converter.py` can be used to fetch and convert public travel content into markdown files if the knowledge base needs to be regenerated.

### 2. Content cleaning and preprocessing

Before the data is indexed, the content is cleaned before being stored in the vector database. The preprocessing step removes unnecessary noise, extra formatting artifacts, and low-value text so the knowledge base contains cleaner, more useful travel information.

This is handled by `knowledge_base_builder.py`, which loads markdown documents, cleans them, and prepares them for chunking.

### 3. Chunking and indexing

The cleaned content is split into chunks using `RecursiveCharacterTextSplitter` with:
- chunk size: 500
- chunk overlap: 50

Each text chunk is stored as a document with metadata such as:
- source file name
- chunk index

These chunks are then embedded and stored in Chroma, a local vector store.

This creates the retrieval layer used by the assistant whenever a user asks a travel question.

### 4. Embedding model

The project uses the Hugging Face embedding model:
- `all-MiniLM-L6-v2`

This converts travel text into vector representations so semantic retrieval can find relevant travel information even when the user uses different wording.

### 5. Local LLM backend via LM Studio

The assistant uses LM Studio as the local LLM backend through an OpenAI-compatible API endpoint.

The environment variables are configured in `.env`:

```env
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_MODEL=qwen2.5-7b-instruct
```

This means the app runs locally without needing an external hosted AI service.

### 6. Query understanding and intent detection

When the user sends a request, the assistant checks whether the question is about:
- travel knowledge
- weather
- currency conversion
- or a combination of both

This logic is implemented in `external_tools.py` and is used to decide whether the system should:
- retrieve from the knowledge base
- call a tool
- or do both

### 7. RAG retrieval flow

For destination questions, the assistant runs a semantic search against Chroma and retrieves the top related chunks. The retrieved content is inserted into the model context, and the LLM generates the final answer based on the retrieved travel evidence.

This makes the answer grounded in the local knowledge base instead of a generic response.

### 8. MCP-style tool use

The system includes live tool support for:
- weather
- currency conversion

The external tool layer calls public APIs and returns structured results. These results are then passed into the final answer generation flow.

This is important for time-sensitive travel questions such as:
- Is it going to rain during the trip?
- Should I carry an umbrella?
- How much is 50000 INR in SGD?

### 9. Combined travel planning response

The strongest feature of the project is the combined response. For example, if the user asks:

“Convert 50000 INR to SGD and suggest a 3-day itinerary considering the weather forecast.”

The assistant does the following:
1. detects that the query needs both currency conversion and weather information
2. calls the relevant tool(s)
3. retrieves itinerary and attraction information from the vector database
4. builds a final answer that combines all of these elements

This matches the assignment requirement for a combined RAG + MCP solution.

### 10. Source trace and metrics

The app also prints a source trace and performance summary that includes:
- which knowledge sources were used
- retrieved chunk information
- which tool(s) were called
- input/output token counts
- latency

This provides transparency during demo and evaluation.

---

## System architecture

The project has the following main files:

- `travel_assistant.py` — main assistant logic, query routing, retrieval, tool orchestration, demo runner, and CLI
- `external_tools.py` — weather, currency conversion, and intent detection logic
- `knowledge_base_builder.py` — cleans markdown content, chunks it, and builds the Chroma vector store
- `markdown_converter.py` — optional scraper for building source markdown from web content
- `data/` — local travel knowledge documents
- `models/chroma_db/` — persisted Chroma database
- `README.md` — project overview and execution guide
- `.gitignore` — excludes local environment and large generated artifacts

---

## Project structure

```text
travel_intelligence_platform/
├── travel_assistant.py
├── external_tools.py
├── knowledge_base_builder.py
├── markdown_converter.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
├── sample_results.md
├── data/
│   ├── itineraries_complete.md
│   ├── neighbourhoods_complete.md
│   ├── things_to_do_complete.md
│   └── wikivoyage_complete.md
├── models/
│   └── chroma_db/
└── README_TEMPLATE.md
```

---

## Technology stack

- Python
- Chroma vector store
- LangChain components
- Hugging Face embeddings
- LM Studio with OpenAI-compatible API
- OpenAI Python SDK
- `requests` for external API access
- `pypdf` for PDF-based work
- `beautifulsoup4` and `markdownify` for scraping and conversion
- `python-dotenv` for environment configuration

---

## Environment setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

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

### 4. Start LM Studio

Before running the app:
1. Start LM Studio locally
2. Load a compatible model
3. Start the local OpenAI-compatible server at `http://localhost:1234/v1`

---

## Build the vector database

Run:

```bash
python knowledge_base_builder.py
```

This script does the following:
- loads the source markdown files
- cleans and normalizes the content
- splits them into chunks
- generates embeddings
- stores them in Chroma

This step prepares the retrieval backend used by the assistant.

---

## Run the application

### Demo mode

```bash
python travel_assistant.py
```

This runs a short demo flow covering:
- attraction recommendations
- family itinerary generation
- weather assessment
- currency conversion
- combined weather + budget itinerary planning

### Interactive CLI mode

```bash
python travel_assistant.py cli
```

This starts the interactive travel assistant so the user can ask custom questions.

---

## Example demo questions

- What are the best attractions in Singapore?
- Suggest a 3-day itinerary in Singapore for a family.
- Convert 60000 INR to SGD.
- What is the weather like in Singapore for the next 3 days? Should I bring an umbrella?
- I have a budget of 50000 INR. Convert it to SGD and suggest a 3-day itinerary considering the weather forecast.
- Which neighborhoods in Singapore are best for cultural experiences and food?
- What indoor attractions can I visit in Singapore if it rains?

---

## Sample results

A short curated set of sample outputs is available in `sample_results.md`.

This includes:
- demo metrics
- per-query metrics
- sample answers for attraction, weather, currency, and combined queries

---

## Prompt and retrieval strategy

The app uses a straightforward prompt strategy designed for assignment clarity:

- destination questions rely on vector-store retrieval
- weather and currency requests trigger tool use
- combined queries use both retrieved knowledge and live tool results
- user preferences are tracked through conversation history
- unsupported information is not presented as fact
- source traces help explain where the answer came from

---

## Data source notes

The project uses local travel content for Singapore. These documents are prepared to support the following kinds of queries:
- attractions
- neighborhoods
- food and shopping areas
- cultural sites
- family-friendly experiences
- practical travel guidance
- sample itineraries

The knowledge base can be updated or rebuilt if more travel content is added.

---

## Assignment alignment

This project addresses the assignment requirements as follows:

- knowledge base from travel resources: implemented
- embedding-based semantic retrieval: implemented
- weather via MCP/tool integration: implemented
- currency conversion via MCP/tool integration: implemented
- combined RAG + tool scenario: implemented
- multi-turn conversation handling: implemented
- source-aware answer generation: implemented
- simple usable interface: implemented

---

## Troubleshooting

### LM Studio not responding

Check:
- LM Studio is running
- the server is available at `http://localhost:1234/v1`
- `.env` values match the running model and endpoint

### No vector database found

Run:

```bash
python knowledge_base_builder.py
```

### Weather or currency tool fails

The application includes graceful fallback behavior, but if the external API is unavailable the app should fail safely instead of inventing data.

---

## Git and branch setup

The project repository is hosted here:

https://github.com/harshitbh/travel_intelligence_platform.git

Recommended Git workflow:

```bash
git init
git add .
git branch -M main
git commit -m "Final Singapore travel assistant"
git remote add origin https://github.com/harshitbh/travel_intelligence_platform.git
git push -u origin main
```

If the remote already exists, use:

```bash
git remote set-url origin https://github.com/harshitbh/travel_intelligence_platform.git
git push -u origin main
```

The project should use the `main` branch as the default branch for tracking the final codebase.

---

## Final note

This project demonstrates the full lifecycle of a local, assignment-ready travel assistant:
- travel content is collected
- cleaned and transformed
- indexed into a vector store
- retrieved using semantic search
- combined with live tools
- passed to a local LLM
- and turned into a final, grounded travel answer

This full pipeline is the main value of the project and is what the final demo should emphasize.