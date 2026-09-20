"""
Singapore Travel Planning Assistant.
RAG + MCP tools + multi-turn conversation orchestration.
"""

# Standard library imports
import os
import re
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List

# Third-party imports
from dotenv import load_dotenv

# Load environment variables from the local .env file before creating any Chroma clients.
load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")


def disable_chroma_telemetry() -> None:
    """Silence Chroma/PostHog warnings caused by the installed dependency version."""
    try:
        import logging
        import posthog
        import chromadb.telemetry.product.posthog as chroma_posthog

        posthog.disabled = True
        logging.getLogger("posthog").disabled = True
        chroma_posthog.posthog.disabled = True
        chroma_posthog.Posthog.capture = lambda self, event: None
        chroma_posthog.Posthog._direct_capture = lambda self, event: None
    except Exception:
        pass


disable_chroma_telemetry()

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

from external_tools import mcp


# ---------------------------------------------------------------------------
# Conversation manager
# ---------------------------------------------------------------------------
class ConversationManager:
    """Track conversation history and user travel preferences."""

    def __init__(self):
        self.messages: List[Dict] = []
        self.preferences: Dict[str, Any] = {}

    def add(self, role: str, content: str, tools_used: List[str] = None):
        """Add a message to the conversation history."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if tools_used:
            msg["tools_used"] = tools_used
        self.messages.append(msg)

    def get_history(self) -> List[Dict]:
        """Return the full conversation history for context."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def get_last_n(self, n: int = 4) -> List[Dict]:
        """Return the most recent messages."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages[-n:]]

    def extract_prefs(self, text: str):
        """Extract travel preferences from the user message."""
        text_lower = text.lower()

        # Budget-related preferences
        if "budget" in text_lower or "inr" in text_lower or "sgd" in text_lower:
            self.preferences["has_budget"] = True

        # Traveller type preferences
        if any(word in text_lower for word in ["family", "children", "kids"]):
            self.preferences["traveler_type"] = "family"
        elif any(word in text_lower for word in ["solo", "alone"]):
            self.preferences["traveler_type"] = "solo"
        elif any(word in text_lower for word in ["couple", "partner"]):
            self.preferences["traveler_type"] = "couple"

        # Preferred activity types
        if any(word in text_lower for word in ["outdoor", "nature", "hiking"]):
            self.preferences["activity"] = "outdoor"
        elif any(word in text_lower for word in ["indoor", "museum", "shopping"]):
            self.preferences["activity"] = "indoor"
        elif any(word in text_lower for word in ["cultural", "heritage", "temple"]):
            self.preferences["activity"] = "cultural"

    def clear(self):
        """Reset the conversation state."""
        self.messages = []
        self.preferences = {}


# ---------------------------------------------------------------------------
# Main assistant
# ---------------------------------------------------------------------------
class TravelAssistant:
    """Combine knowledge-base retrieval, tool access, and multi-turn conversation."""

    def __init__(self, model: str = None):
        # Local LM Studio is the only supported LLM backend for this project.
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.chunking_method = "RecursiveCharacterTextSplitter"
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.lm_base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        self.lm_api_key = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
        self.lm_model = os.environ.get("LMSTUDIO_MODEL", "qwen2.5-7b-instruct")

        if not self.lm_base_url or not self.lm_model:
            raise ValueError("Set LMSTUDIO_BASE_URL and LMSTUDIO_MODEL in .env")

        self.client = OpenAI(
            base_url=self.lm_base_url,
            api_key=self.lm_api_key,
        )
        self.model = model or self.lm_model

        print("=" * 70)
        print("RUNTIME CONFIGURATION")
        print("=" * 70)
        print(f"Embedding model: {self.embedding_model_name}")
        print(f"Chunking method: {self.chunking_method}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Chunk overlap: {self.chunk_overlap}")
        print(f"LLM backend: LM Studio (OpenAI-compatible)")
        print(f"LLM endpoint: {self.lm_base_url}")
        print(f"LLM model: {self.model}")

        # Initialize knowledge-base retrieval
        print("Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        db_paths = ["./models/chroma_db", "./chroma_singapore_db"]
        last_error = None
        self.vectorstore = None
        self.retriever = None
        self.chroma_db_path = None

        for db_path in db_paths:
            try:
                self.vectorstore = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings,
                    client_settings=Settings(anonymized_telemetry=False),
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})
                self.chroma_db_path = db_path
                print(f"Loaded vector store from: {db_path}")
                break
            except Exception as e:
                last_error = e

        if self.retriever is None:
            print(f"Vector store not found in either location: {last_error}")
        else:
            try:
                total_chunks = self.vectorstore._collection.count()
            except Exception:
                total_chunks = "unknown"
            print(f"Using Chroma DB: {self.chroma_db_path}")
            print(f"Total indexed chunks: {total_chunks}")

        # Initialize conversation and MCP tools
        self.conv = ConversationManager()
        self.mcp = mcp

    def _print_query_trace(self, user_query: str):
        """Display the active backend and retrieval setup for a user query."""
        print("=" * 70)
        print("QUERY TRACE")
        print("=" * 70)
        print(f"Embedding model: {self.embedding_model_name}")
        print(f"Chroma DB: {self.chroma_db_path or 'Not loaded'}")
        print(f"Chunking: {self.chunking_method} | size={self.chunk_size} | overlap={self.chunk_overlap}")
        print(f"LLM backend: LM Studio")
        print(f"LLM model: {self.model}")
        print(f"Query: {user_query}")
        print("=" * 70)

    @staticmethod
    def _summarize_chunk_details(details: List[Dict[str, Any]]) -> List[str]:
        """Reduce full chunk metadata into a readable per-source summary for the demo."""
        grouped: Dict[str, List[int]] = {}
        for item in details:
            source = item.get("source_file", "unknown")
            chunk_index = item.get("chunk_index")
            if chunk_index is None:
                continue
            try:
                chunk_index = int(chunk_index)
            except (TypeError, ValueError):
                continue
            grouped.setdefault(source, []).append(chunk_index)

        summaries: List[str] = []
        for source, chunks in sorted(grouped.items()):
            unique_chunks = sorted(set(chunks))
            preview = unique_chunks[:4]
            suffix = ", ..." if len(unique_chunks) > 4 else ""
            summaries.append(f"{source}[{', '.join(map(str, preview))}{suffix}]")
        return summaries

    def _print_source_trace(self, trace: Dict[str, Any]):
        """Print a compact source summary without repeating the query summary."""
        print("\n" + "=" * 70)
        print("SOURCE TRACE")
        print("=" * 70)
        print(f"RAG: {'Yes' if trace.get('rag_used') else 'No'}")
        if trace.get('rag_used'):
            print(f"RAG DB: {trace.get('rag_db')}")
            source_summary = trace.get('rag_sources', [])
            if source_summary:
                print(f"RAG sources: {source_summary}")
            details = trace.get('rag_source_details', [])
            if details:
                chunk_summaries = self._summarize_chunk_details(details)
                print(f"Chunk summary: {', '.join(chunk_summaries)}")
            print(f"Retrieved chunks: {len(details)}")
            print(f"Chunking: {trace.get('chunking_method')} | size={trace.get('chunk_size')} | overlap={trace.get('chunk_overlap')}")
        print(f"MCP tools: {trace.get('mcp_tools_used', [])}")
        print(f"LLM: {'Yes' if trace.get('llm_used') else 'No'}")
        if trace.get('llm_used'):
            print(f"LLM model: {trace.get('llm_model')}")
            print(f"LLM backend: {trace.get('llm_backend')}")
            print(f"Input tokens: {trace.get('input_tokens')}")
            print(f"Output tokens: {trace.get('output_tokens')}")
            print(f"Total tokens: {trace.get('total_tokens')}")
        print(f"Latency: {trace.get('latency_ms')} ms")
        print(f"Final source: {trace.get('source')}")
        print("=" * 70)

    @staticmethod
    def _clean_model_output(text: str) -> str:
        """Remove reasoning traces, wrapper wording, and leaked code snippets from model output."""
        if not text:
            return text
 
        cleaned = text
        lower_text = cleaned.lower()

  

        while "<think>" in lower_text:
            start = lower_text.find("<think>")
            end = lower_text.find("</think>", start)
            if end == -1:
                cleaned = cleaned[:start]
                lower_text = cleaned.lower()
                break
            cleaned = cleaned[:start] + cleaned[end + len("</think>") :]
            lower_text = cleaned.lower()

        code_pattern = r"(?im)^\s*(?:@staticmethod\b|def\s+[A-Za-z_]\w*\s*\(|class\s+[A-Za-z_]\w*\s*(?:\(|:)|import\s+[A-Za-z_][\w.]*|from\s+[A-Za-z_][\w.]*\s+import\s+)"
        code_match = re.search(code_pattern, cleaned)
        if code_match:
            cleaned = cleaned[:code_match.start()].strip()

        cleaned = re.sub(r"(?is)^\s*(here\s+is\s+the\s+answer|answer:|final\s+answer:|summary:)\s*", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = cleaned.strip()
 
        if not cleaned:
            return "The knowledge base does not contain a complete list for this topic."
        return cleaned

    def _system_prompt(self) -> str:
        """Return the instruction prompt used for responses."""
        return """You are a concise Singapore Travel Planning Assistant.

Follow these rules strictly:

1. Answer directly and briefly; do not narrate your process.
2. Do not show internal reasoning, chain-of-thought, or <think> blocks.
3. Use only information that is actually present in the knowledge base or tool output.
4. If data is incomplete, say so briefly and do not invent details.
5. Keep citations simple and factual, for example: [Knowledge Base], [Weather Forecast], [Currency Conversion], [Recommendation].
6. Never fabricate attractions, prices, routes, timings, or travel facts.
7. For itineraries, give a simple day-by-day plan only when supported by evidence.
8. Remember user preferences such as budget, family needs, and activities across the conversation.
9. Do not repeat the same fact more than once.
10. Keep total answer length short: usually 3 to 5 bullets or a short paragraph.
11. Do not start with phrases like "We are given" or "The sources show"; just give the answer.
12. Never paste raw source code, function definitions, class bodies, or file snippets from the system or retrieved data; answer only with user-facing travel guidance.
13. If no exact data is available, say: "The knowledge base does not contain a complete list for this topic."

Answer style:
- Clear and natural
- Brief and direct
- No internal explanations or defensive repetition
- No Markdown tables unless necessary
- Final answer must be user-facing only"""

    def search_kb(self, query: str, trace: Dict[str, Any]) -> str:
        """Search the knowledge base for relevant travel information."""
        if not self.retriever:
            trace["rag_used"] = False
            return "[Knowledge Base Not Available]"

        try:
            docs = self.retriever.invoke(query)
            if not docs:
                trace["rag_used"] = False
                return "[No relevant KB information found]"

            trace["rag_used"] = True
            trace["rag_db"] = self.chroma_db_path
            trace["rag_sources"] = []
            trace["rag_source_details"] = []
            source_counts = {}
            results = []
            for doc in docs:
                source = doc.metadata.get("source_file", "Unknown")
                chunk_index = doc.metadata.get("chunk_index")
                detail = {
                    "source_file": source,
                    "chunk_index": chunk_index,
                }
                trace["rag_source_details"].append(detail)
                source_counts[source] = source_counts.get(source, 0) + 1
                results.append(f"[Source: {source} | chunk {chunk_index}]\n{doc.page_content}")

            trace["rag_sources"] = [
                f"{name} ({count} chunks)" for name, count in sorted(source_counts.items())
            ]

            return "\n\n---\n\n".join(results)
        except Exception as e:
            trace["rag_used"] = False
            return f"[KB Error: {str(e)}]"

    def _build_direct_tool_response(self, user_query: str, tool_analysis: Dict, tool_results: Dict) -> str:
        """Return a direct answer only for pure tool-only queries."""
        if tool_analysis.get("needs_currency") and tool_results.get("currency", {}).get("status") == "success":
            data = tool_results["currency"]["data"]
            return (
                f"{data['amount']:.0f} {data['from']} = {data['converted']:.2f} {data['to']} "
                f"(rate: {data['rate']:.4f})."
            )

        if tool_analysis.get("needs_weather") and tool_results.get("weather", {}).get("status") == "success":
            days = tool_results["weather"]["data"][:3]
            summary = []
            for day in days:
                summary.append(f"{day['date']}: {day['max_temp']}°C max, {day['rain_prob']}% rain")
            rain_probabilities = [day["rain_prob"] for day in days]
            umbrella_note = "Yes, carry an umbrella; rain is likely across the next 3 days." if max(rain_probabilities, default=0) >= 80 else "Rain is possible, so an umbrella is still a good idea."
            return "Singapore weather for the next 3 days: " + "; ".join(summary) + ". " + umbrella_note

        return ""

    def call_tools(self, tool_analysis: Dict) -> Dict[str, Any]:
        """Call the relevant MCP tools based on the user query."""
        results = {}

        if tool_analysis["needs_weather"]:
            print("Calling weather tool...")
            results["weather"] = self.mcp.get_weather(days=3)

        if tool_analysis["needs_currency"]:
            print("Calling currency tool...")
            params = tool_analysis["currency_params"]
            if params.get("amount") and params.get("from") and params.get("to"):
                results["currency"] = self.mcp.convert_currency(
                    params["amount"],
                    params["from"],
                    params["to"],
                )

        return results

    def format_tools_output(self, results: Dict) -> str:
        """Format tool output into a clean block for the model."""
        if not results:
            return ""

        output = "\n## Current Information from MCP Tools:\n"

        if "weather" in results and results["weather"]["status"] == "success":
            output += "\n### Weather Forecast:\n"
            for day in results["weather"]["data"][:3]:
                output += f"- {day['date']}: {day['max_temp']}°C max, {day['rain_prob']}% rain\n"

        if "currency" in results and results["currency"]["status"] == "success":
            d = results["currency"]["data"]
            output += "\n### Currency Conversion:\n"
            output += f"- {d['amount']} {d['from']} = {d['converted']:.2f} {d['to']} (Rate: {d['rate']:.4f})\n"

        return output

    def respond(self, user_query: str) -> Dict[str, Any]:
        """Generate a response using RAG and MCP tools."""
        start_time = time.perf_counter()
        self._print_query_trace(user_query)
        trace = {
            "query": user_query,
            "rag_used": False,
            "rag_db": self.chroma_db_path,
            "rag_sources": [],
            "rag_source_details": [],
            "llm_used": False,
            "llm_model": self.model,
            "llm_backend": "LM Studio",
            "mcp_tools_used": [],
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "latency_ms": None,
            "chunking_method": self.chunking_method,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "source": "unknown",
        }

        # Add the user message and extract preferences
        self.conv.add("user", user_query)
        self.conv.extract_prefs(user_query)

        # Decide which tools and retrieval steps are needed
        tool_analysis = self.mcp.detect_tool_needs(user_query)

        # Build the context for the model
        context_parts = []

        # Add knowledge-base retrieval if needed
        if tool_analysis["needs_kb"]:
            kb_content = self.search_kb(user_query, trace)
            context_parts.append(f"## Knowledge Base Results:\n{kb_content}")

        # Add MCP tool output if needed
        tool_results = {}
        if tool_analysis["tools_to_call"]:
            trace["mcp_tools_used"] = tool_analysis["tools_to_call"]
            tool_results = self.call_tools(tool_analysis)
            context_parts.append(self.format_tools_output(tool_results))

        direct_response = self._build_direct_tool_response(user_query, tool_analysis, tool_results)
        is_pure_tool_query = tool_analysis["query_type"] == "tools_only" and not tool_analysis["needs_kb"]
        if direct_response and is_pure_tool_query:
            trace["source"] = "mcp_tool"
            trace["llm_used"] = False
            trace["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
            self.conv.add("assistant", direct_response, tools_used=tool_analysis["tools_to_call"])
            self._print_source_trace(trace)
            return {
                "response": direct_response,
                "query_type": tool_analysis["query_type"],
                "tools_used": tool_analysis["tools_to_call"],
                "tool_results": tool_results,
                "trace": trace,
            }

        full_context = "\n".join(context_parts)

        # Prepare messages for the model
        messages = [{"role": "system", "content": self._system_prompt()}]

        # Add recent conversation context
        for msg in self.conv.get_last_n(4)[:-1]:
            messages.append(msg)

        if full_context:
            user_input = f"{full_context}\n\n**User's Question:** {user_query}"
        else:
            user_input = user_query

        messages.append({"role": "user", "content": user_input})

        # Call the LLM
        print(f"Calling {self.model}...")
        trace["llm_used"] = True
        trace["source"] = "llm" if not trace["rag_used"] else "rag_and_llm"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
            )
            assistant_msg = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            if usage is not None:
                trace["input_tokens"] = getattr(usage, "prompt_tokens", None)
                trace["output_tokens"] = getattr(usage, "completion_tokens", None)
                trace["total_tokens"] = getattr(usage, "total_tokens", None)
        except Exception as e:
            assistant_msg = f"Error: {str(e)}"

        assistant_msg = self._clean_model_output(assistant_msg)

        # Save the assistant reply to the conversation history
        trace["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
        self.conv.add("assistant", assistant_msg, tools_used=tool_analysis["tools_to_call"])
        self._print_source_trace(trace)

        return {
            "response": assistant_msg,
            "query_type": tool_analysis["query_type"],
            "tools_used": tool_analysis["tools_to_call"],
            "tool_results": tool_results,
            "trace": trace,
        }

    def reset(self):
        """Clear the conversation state."""
        self.conv.clear()
        print("Conversation cleared")


# ---------------------------------------------------------------------------
# Demo and CLI interfaces
# ---------------------------------------------------------------------------
def run_demo():
    """Run a demonstration of the app features."""
    print("=" * 70)
    print("SINGAPORE TRAVEL ASSISTANT - DEMO")
    print("=" * 70)

    try:
        assistant = TravelAssistant()
    except ValueError as e:
        print(f"{e}")
        print("Fix: create a .env file with HF_TOKEN=your_token")
        return

    demo_queries = [
        "What are the best attractions in Singapore?",
        "Suggest a 3-day itinerary in Singapore for a family.",
        "Convert 60000 INR to SGD.",
        "What is the weather like in Singapore for the next 3 days? Should I bring an umbrella?",
        "I have a budget of 50000 INR. Convert it to SGD and suggest a 3-day itinerary considering the weather forecast.",
        "Which attractions in Singapore are best for young children?",
        "What are the best cultural and heritage places to visit in Singapore?",
        "What are the best neighborhoods in Singapore for food and shopping?",
    ]

    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"DEMO {i}: {query}")
        print(f"{'=' * 70}")

        result = assistant.respond(query)

        print(f"\nAnalysis:")
        print(f"  Query Type: {result['query_type']}")
        print(f"  Tools Used: {result['tools_used']}")

        print(f"\nResponse:")
        preview_limit = 1200
        print(result["response"][:preview_limit] + ("..." if len(result["response"]) > preview_limit else ""))

        if result["tool_results"]:
            print("\nTool Results:")
            for tool, data in result["tool_results"].items():
                if data["status"] == "success":
                    print(f"  {tool.upper()}: {data['data']}")
                else:
                    print(f"  {tool.upper()}: {data['message']}")

        print("\nMoving to next demo query automatically...\n")

    print("\n" + "=" * 70)
    print("Demo complete")
    print("=" * 70)


def run_cli():
    """Run the interactive command-line interface."""
    try:
        assistant = TravelAssistant()
    except ValueError as e:
        print(f"{e}")
        return

    print("Singapore Travel Assistant (Type 'exit' to quit, 'clear' to reset)")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if query.lower() == "clear":
            assistant.reset()
            continue

        if not query:
            continue

        result = assistant.respond(query)
        print(f"\nAssistant:\n{result['response']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        run_cli()
    else:
        run_demo()