"""
HR Policy Assistant - Document Ingestion Script
==============================================
Loads the Enso Group HR Policy Manual (hr_manual.pdf), cleans text noise (running
headers/footers, corrupted glyphs, hyphenated line breaks), splits it into chunks,
extracts 1-indexed page numbers and nearest section headings, generates
embeddings with Google Gemini (gemini-embedding-001), and persists them into ChromaDB.
Safely wipes old chroma_db before indexing to prevent duplicate records.
"""

import os
import re
import sys
import time
import shutil
import warnings
from dotenv import load_dotenv

# Suppress deprecation warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env
load_dotenv()

# Verify Google API Key is present
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: Missing Google API Key. Please define GOOGLE_API_KEY in your .env file.", file=sys.stderr)
    sys.exit(1)

# Ensure GOOGLE_API_KEY is available in os.environ for LangChain integrations
os.environ["GOOGLE_API_KEY"] = api_key
if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_chroma import Chroma
except ImportError as e:
    print(f"Error importing required libraries: {e}. Did you run 'pip install -r requirements.txt'?", file=sys.stderr)
    sys.exit(1)

PDF_PATH = "hr_manual.pdf"
CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 40  # Embed in batches to stay safely within free-tier 100 RPM quota

# Regex pattern to identify major sections, annexures, and policy titles
HEADING_REGEX = re.compile(
    r'(?m)^\s*(?:'
    r'(SECTION\s*[-–—]\s*[I|V|X\d]+(?:\s+[-–—:]\s*[^\n]+)?)'
    r'|(ANNEXURE\s*[-–—]?\s*[I|V|X\d]+(?:\s+[-–—:]\s*[^\n]+)?)'
    r'|(\[[A-Z]\]\s*[^\n]+)'
    r'|([A-Za-z\s/&]+Policy(?:\s+Statement)?)'
    r'|(Company Rules and regulations?)'
    r'|(Standard of Conduct[^\n]*)'
    r'|(Gifts and Hospitality)'
    r'|(Recruitment)'
    r'|(Dress Code)'
    r'|(Punctuality and [Aa]ttendance)'
    r'|(Staff Advances)'
    r'|(Reward and [Rr]ecognition)'
    r')\s*$',
    re.IGNORECASE
)


def clean_page_text(text: str) -> str:
    """
    Cleans noisy PDF artifacts:
    1. Removes repetitive headers/footers like 'HR Manual - X'.
    2. Re-attaches broken hyphenated words at line-ends (e.g. 'compa-\nny' -> 'company').
    3. Removes font encoding artifacts/corrupted glyphs (e.g., 'Ԃ', '¤').
    4. Normalizes arrow/bullet glyphs ('▶').
    5. Collapses excessive consecutive whitespace and blank lines.
    """
    # Remove repetitive page footers like 'HR Manual - 42'
    text = re.sub(r'(?mi)^\s*HR\s+Manual\s*[-–—]\s*\d+\s*$', '', text)
    # Re-connect hyphenated line-end splits
    text = re.sub(r'(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)', r'\1\2', text)
    # Remove corrupted font artifacts
    text = re.sub(r'[Ԃ¤]', '', text)
    # Normalize bullet arrows
    text = re.sub(r'▶\s*\n\s*▶', '•', text)
    text = re.sub(r'▶', '•', text)
    # Normalize horizontal whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize excessive vertical breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_section_headings(chunks):
    """
    Detect the nearest section heading for each chunk sequentially.
    If a chunk contains a new section heading, update current_section.
    Subsequent chunks inherit the nearest preceding section heading.
    """
    current_section = "Introduction / General"
    for chunk in chunks:
        text = chunk.page_content
        matches = HEADING_REGEX.findall(text)
        if matches:
            for match in matches:
                heading = next((m for m in match if m), None)
                if heading:
                    cleaned = re.sub(r'\s+', ' ', heading).strip()
                    if len(cleaned) > 3 and not cleaned.endswith('...'):
                        current_section = cleaned
                        break
        chunk.metadata["section"] = current_section


def ingest_document(pdf_path: str = PDF_PATH, chroma_dir: str = CHROMA_DIR):
    """
    Ingest PDF document into ChromaDB vector store in batches with quota handling.
    Safely removes old vector database before building.
    """
    # 1. Validate PDF existence
    if not os.path.exists(pdf_path):
        print(f"Error: Source PDF '{pdf_path}' not found in project root.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading document: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Successfully loaded {len(pages)} pages.")

    # 2. Clean page text and assign 1-indexed page numbers
    print("Applying text cleaning (removing noise, headers, footers, broken words)...")
    for doc in pages:
        doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1
        doc.page_content = clean_page_text(doc.page_content)

    # 3. Split into chunks using RecursiveCharacterTextSplitter
    print(f"Splitting document into chunks (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    total_chunks = len(chunks)
    avg_len = sum(len(c.page_content) for c in chunks) / total_chunks if total_chunks > 0 else 0
    print(f"Generated {total_chunks} chunks (Average length: {avg_len:.1f} characters).")

    # 4. Detect nearest section heading per chunk
    extract_section_headings(chunks)

    # 5. Clean prior Chroma database directory if present (Safe to rerun)
    if os.path.exists(chroma_dir):
        print(f"Clearing previous vector database at '{chroma_dir}' to prevent duplicate chunks...")
        shutil.rmtree(chroma_dir)

    # 6. Initialize Google Gemini Embeddings
    print(f"Initializing embeddings ({EMBEDDING_MODEL})...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )

    vectorstore = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings
    )

    # 7. Batch indexing with automatic quota backoff
    print(f"Indexing {total_chunks} chunks into ChromaDB in batches of {BATCH_SIZE}...")
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_idx = (i // BATCH_SIZE) + 1
        num_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE

        # Retry loop for rate limits
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"  Indexing batch {batch_idx}/{num_batches} (chunks {i + 1} to {min(i + len(batch), total_chunks)})...")
                vectorstore.add_documents(documents=batch)
                break
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    wait_time = 25 * (attempt + 1)
                    print(f"  [Rate Limit / 429] Free tier quota reached. Pausing for {wait_time}s before retrying batch {batch_idx} (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"Error during batch indexing: {e}", file=sys.stderr)
                    raise e

        # Short pause between batches to prevent quota bursts
        if i + BATCH_SIZE < total_chunks:
            time.sleep(2)

    print(f"\n=======================================================")
    print(f"Ingestion complete!")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"ChromaDB persisted at: {chroma_dir}")
    print(f"=======================================================\n")
    return total_chunks


if __name__ == "__main__":
    ingest_document()
