"""
HR Policy Assistant - Query and Retrieval CLI
============================================
Performs similarity search against ChromaDB using Google Gemini embeddings.
Grounded answering with strict JSON verification:
- Top 4 chunks numbered [1] to [4]
- Gemini LLM returns strict JSON with verbatim quote and chunk number
- Programmatic grounding check verifies verbatim quote in chunk text
- Numeric grounding check verifies all numbers in answer appear in the quote
- Exact output format with Page and Section references, or 'Not found'
"""

import os
import re
import sys
import json
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

os.environ["GOOGLE_API_KEY"] = api_key
if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

try:
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError as e:
    print(f"Error importing required libraries: {e}. Did you run 'pip install -r requirements.txt'?", file=sys.stderr)
    sys.exit(1)

# Configuration
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.1-flash-lite"
TOP_K = 4

# Configurable similarity threshold: Chroma returns cosine distance
# Lower score means closer similarity. Matches worse than threshold are rejected.
THRESHOLD = float(os.getenv("THRESHOLD", "0.70"))

# Cache vectorstore and LLM instances
_vectorstore = None
_llm = None


def get_vectorstore(chroma_dir: str = CHROMA_DIR):
    """
    Load the ChromaDB vector database.
    Raises FileNotFoundError if the database directory does not exist.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if not os.path.exists(chroma_dir):
        raise FileNotFoundError(
            f"ChromaDB directory '{chroma_dir}' not found. Please run 'python ingest.py' first."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )
    _vectorstore = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings
    )
    return _vectorstore


def get_llm():
    """
    Initialize LLM using gemini-3.1-flash-lite with temperature 0.
    """
    global _llm
    if _llm is not None:
        return _llm

    _llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        timeout=60,
        max_retries=2,
        google_api_key=api_key
    )
    return _llm


def extract_text_from_response(content) -> str:
    """Helper to extract text whether response content is string or list of blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts).strip()
    return str(content).strip()


def normalize_text(text: str) -> str:
    """Normalize text for whitespace and casing comparison."""
    # Replace unicode non-breaking spaces, collapse whitespace, lowercase
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def clean_preambles(text: str) -> str:
    """Strip common conversational preambles like 'Based on the provided context'."""
    patterns = [
        r'^(?:based on the (?:provided )?context,?\s*)',
        r'^(?:according to the (?:provided )?(?:hr )?(?:policy )?manual,?\s*)',
        r'^(?:according to the context,?\s*)',
        r'^(?:based on the document,?\s*)',
        r'^(?:as per the policy,?\s*)',
    ]
    cleaned = text.strip()
    for p in patterns:
        cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE).strip()
    # Capitalize first letter if needed
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def parse_llm_json(raw_text: str):
    """
    Safely parse JSON response from LLM, stripping markdown code blocks.
    Returns dict if valid, else None.
    """
    text = raw_text.strip()
    # Remove markdown code blocks ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # Find outermost JSON object
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except Exception:
        return None


def ask(question: str):
    """
    Answer user question strictly grounded in hr_manual.pdf.

    Returns:
        tuple: (answer, quote, reference) if found, or ("Not found", "", "")
    """
    question = question.strip()
    if not question:
        print("Not found")
        return ("Not found", "", "")

    try:
        vs = get_vectorstore()
    except FileNotFoundError as e:
        print("Not found")
        return ("Not found", "", "")

    # 1. Retrieve top 4 chunks with similarity_search_with_score
    try:
        results = vs.similarity_search_with_score(question, k=TOP_K)
    except Exception as e:
        print("Not found")
        return ("Not found", "", "")

    if not results:
        print("Not found")
        return ("Not found", "", "")

    # Best score check against configurable threshold
    best_score = min(score for _, score in results)
    if best_score > THRESHOLD:
        print("Not found")
        return ("Not found", "", "")

    # Number retrieved chunks [1] to [4] in the prompt
    context_blocks = []
    for i, (doc, score) in enumerate(results, 1):
        context_blocks.append(f"[{i}]\n{doc.page_content.strip()}")
    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "You are the HR Policy Assistant for Enso Group.\n"
        "Answer the user's question using ONLY the provided numbered context chunks [1] to [4].\n"
        "You MUST reply with a strict JSON object with NO surrounding markdown or backticks.\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "found": true/false,\n'
        '  "answer": "...",\n'
        '  "quote": "...",\n'
        '  "chunk": <chunk_number_from_1_to_4>\n'
        "}\n\n"
        "Strict Grounding Rules:\n"
        "1. If the context does not explicitly contain the answer, return {\"found\": false}.\n"
        "2. \"quote\" MUST be copied verbatim from the specific chunk text containing the answer (maximum 2 sentences).\n"
        "3. \"chunk\" MUST be the integer number of the chunk where the quote is located (1, 2, 3, or 4).\n"
        "4. \"answer\" may only use facts, numbers, and units that appear directly in the quote. "
        "No unit conversions, no calculations, and no outside knowledge. "
        "For example, if the quote says \"12 weeks\", the answer must not add \"90 days\" unless \"90 days\" is in the quote.\n"
        "5. Do NOT include preambles like \"Based on the provided context\"."
    )

    user_prompt = (
        f"Numbered Context Chunks:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "JSON Response:"
    )

    llm = get_llm()
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        raw_output = extract_text_from_response(response.content)
    except Exception as e:
        print(f"API Error: {e}")
        return ("API error", "", "")

    data = parse_llm_json(raw_output)
    if not data or not isinstance(data, dict):
        print("Not found")
        return ("Not found", "", "")

    # Check if LLM flagged as not found
    if not data.get("found", False):
        print("Not found")
        return ("Not found", "", "")

    answer = clean_preambles(str(data.get("answer", "")).strip())
    quote = str(data.get("quote", "")).strip()
    chunk_num = data.get("chunk")

    if not answer or not quote or not isinstance(chunk_num, int):
        print("Not found")
        return ("Not found", "", "")

    if chunk_num < 1 or chunk_num > len(results):
        print("Not found")
        return ("Not found", "", "")

    # 3. Programmatic grounding checks
    target_doc = results[chunk_num - 1][0]
    norm_chunk = normalize_text(target_doc.page_content)
    norm_quote = normalize_text(quote)

    # Check that quote appears verbatim in the chunk text
    if norm_quote not in norm_chunk:
        # Also check if quote matches anywhere across retrieved chunks
        matched_doc = None
        for doc, _ in results:
            if norm_quote in normalize_text(doc.page_content):
                matched_doc = doc
                break
        if matched_doc is not None:
            target_doc = matched_doc
        else:
            print("Not found")
            return ("Not found", "", "")

    # Verify every number in the answer also appears in the quote
    # Extracts all integers and decimal numbers (even if attached to units like '90days')
    answer_numbers = re.findall(r'\d+(?:\.\d+)?', answer)
    quote_numbers = set(re.findall(r'\d+(?:\.\d+)?', quote))
    for num in answer_numbers:
        if num not in quote_numbers:
            # Answer introduced a number not found in the quote (e.g. calculation/conversion)
            print("Not found")
            return ("Not found", "", "")

    # 4. Build reference only from the chunk where the quote was found
    page = target_doc.metadata.get("page", "Unknown")
    section = target_doc.metadata.get("section", "General")
    reference = f'Page {page}, Section "{section}"'

    # 6. Output format:
    # Answer: <answer>
    # Supporting text: "<quote>"
    # Reference: Page X, Section "Y"
    print(f"Answer: {answer}")
    print(f'Supporting text: "{quote}"')
    print(f"Reference: {reference}")

    return (answer, quote, reference)


def cli_loop():
    """Interactive Command Line Interface for asking HR Policy questions."""
    print("=" * 60)
    print("      Enso Group HR Policy Assistant (CLI)")
    print("      Type your question or 'exit' / 'quit' to end.")
    print("=" * 60)

    # Check database status upfront
    if not os.path.exists(CHROMA_DIR):
        print(f"Notice: Database not found at '{CHROMA_DIR}'.", file=sys.stderr)
        print("Please run 'python ingest.py' first to build the vector store.", file=sys.stderr)
        sys.exit(1)

    while True:
        try:
            query = input("\nAsk HR Question > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            ask(query)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        single_question = " ".join(sys.argv[1:])
        ask(single_question)
    else:
        cli_loop()
