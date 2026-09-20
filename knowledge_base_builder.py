"""
Build a vector store from markdown files in the /data directory.
Use this script to rebuild embeddings from existing markdown files.
Run: python knowledge_base_builder.py
"""

# Standard library imports
import os
import re
import warnings
from pathlib import Path

# Third-party imports
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
# Disable Chroma/PostHog telemetry in local database builds.

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
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Seed content for top-attraction answers
# ---------------------------------------------------------------------------
TOP_ATTRACTIONS_TEXT = """# Singapore Top Attractions

Singapore is renowned for a mix of iconic waterfront, heritage, nature, and cultural experiences. The most consistently highlighted attractions include:

## Top Attractions
- Marina Bay
- Gardens by the Bay
- Sentosa Island
- Singapore Flyer
- Kampong Gelam
- Little India
- Chinatown
- Orchard Road
- Civic District
- Singapore River

## Why these stand out
Marina Bay and Gardens by the Bay are central to Singapore's modern skyline and garden city identity. Sentosa Island offers beach, amusement, and family-friendly experiences. Kampong Gelam, Little India, and Chinatown add heritage and cultural depth. Orchard Road and the Civic District round out the classic city experience.
"""


def create_top_attractions_seed_file() -> None:
    """Create a clean, curated attraction document so KB retrieval has a direct top-attraction source."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    seed_path = data_dir / "top_attractions_singapore.md"
    seed_path.write_text(TOP_ATTRACTIONS_TEXT, encoding="utf-8")


# ---------------------------------------------------------------------------
# Markdown cleanup for better retrieval
# ---------------------------------------------------------------------------
def clean_markdown_content(content: str) -> str:
    """Remove website navigation, duplicate language links, and noisy markdown before indexing."""
    if not content:
        return ""

    text = content.replace("\r\n", "\n")
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://|/)[^)]+\)", r"\1", text)
    text = re.sub(r"data:image[^\s)]+", " ", text)
    text = re.sub(r"\b(Source|Generated|Total Pages):.*", " ", text)
    text = re.sub(r"\b(Table of Contents|Start Page|Get Recommendations|Featured Neighbourhoods|Top Things To Do|Travel Tips|Essential Information|Things To Do|Neighbourhoods|Visit Singapore)\b", " ", text, flags=re.I)

    cleaned_lines = []
    nav_keywords = {
        "Global",
        "Get Recommendations",
        "What's Happening",
        "All Happenings",
        "Featured Neighbourhoods",
        "Top Things To Do",
        "Things To Do",
        "Travel Tips",
        "Essential Information",
        "Shop",
        "Dine",
        "Wellness",
        "Tours",
        "Deutsch",
        "Bahasa Indonesia",
        "日本語",
        "한국어",
        "Tiếng Việt",
        "ไทย",
        "中文",
        "Visit Singapore",
        "City in Nature",
        "Culture & Heritage",
        "Unique Experiences",
        "Iconic Architecture",
        "Family Fun",
        "After Dark",
        "Museums & Galleries",
        "Login",
        "Register",
        "Edit",
        "View history",
        "Discussion",
        "Create account",
    }

    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if "data:image" in cleaned:
            continue
        if any(keyword.lower() in cleaned.lower() for keyword in nav_keywords):
            continue
        if re.match(r"^\* \[[^\]]+\]\(https?://.*\)$", cleaned):
            continue
        if re.match(r"^\*\s*(?:\[[^\]]+\]\()?(https?://|/).*$", cleaned):
            continue
        if re.match(r"^\d+\.?\s*$", cleaned):
            continue
        if cleaned.startswith("#") or cleaned.startswith("##") or cleaned.startswith("###"):
            cleaned_lines.append(cleaned)
            continue
        if re.match(r"^\d+\.?\s", cleaned):
            cleaned_lines.append(cleaned)
            continue
        if cleaned.startswith("[") and "](" in cleaned and "http" in cleaned:
            continue

        cleaned_lines.append(cleaned)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Vector store builder
# ---------------------------------------------------------------------------
def build_vectorstore():
    """Build and persist a Chroma vector store from markdown files."""

    data_dir = Path("data")
    if not data_dir.exists():
        print("Error: 'data/' directory not found")
        print("Create it and add your markdown files (.md)")
        return False

    create_top_attractions_seed_file()
    md_files = sorted(data_dir.glob("*.md"))
    if not md_files:
        print("Error: No .md files found in data/")
        return False

    print(f"Found {len(md_files)} markdown files")

    # Load and split documents into chunks for embedding
    print("Loading and chunking documents...")
    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    for filepath in md_files:
        print(f"Processing {filepath.name}...")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            cleaned_content = clean_markdown_content(content)
            if not cleaned_content:
                print(f"Skipping empty cleaned content for {filepath.name}")
                continue

            chunks = splitter.split_text(cleaned_content)

            for i, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source_file": filepath.stem,
                            "chunk_index": i,
                        },
                    )
                )

    print(f"Created {len(documents)} chunks from {len(md_files)} files")

    # Initialize the embedding model used to convert text into vector form
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Build the Chroma vector store from the processed documents
    persist_dir = "./models/chroma_db"
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    print("Embedding and indexing documents into Chroma...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        client_settings=Settings(anonymized_telemetry=False),
    )

    print(f"Vector store ready: {persist_dir}/")
    return True


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    success = build_vectorstore()
    raise SystemExit(0 if success else 1)