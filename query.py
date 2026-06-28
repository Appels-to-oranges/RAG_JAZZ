"""
Step 3: Query the RAG system.

HOW IT WORKS:
  1. Your question is embedded with the same model used to embed the chunks
     (all-MiniLM-L6-v2).  This produces a 384-dim vector.

  2. ChromaDB finds the K chunks whose vectors are closest to your question
     vector (cosine similarity).  These are the passages most *semantically
     related* to what you asked -- even if they don't share exact keywords.

  3. The retrieved chunks (with their metadata) are assembled into a prompt
     and sent to Ollama, which runs a local LLM.  The prompt instructs the
     model to answer ONLY from the provided context and cite its sources.

USAGE:
  python query.py "What is a tritone substitution?"
  python query.py "How does stride piano differ from boogie-woogie?"
  python query.py --no-llm "circle of fifths"    # retrieval only, no LLM
"""

import argparse
import io
import sys
import textwrap

import chromadb

# Force UTF-8 output on Windows so music symbols (sharps, flats) render
# instead of crashing with a codec error.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# --- Configuration ---

CHROMA_DIR = "./chroma_db"          # Must match build_vectorstore.py
COLLECTION_NAME = "jazz_piano"      # Must match build_vectorstore.py
DEFAULT_K = 5                       # Number of chunks to retrieve per query
DEFAULT_MODEL = "llama3.2"          # Ollama model to use for generation

# System prompt sent to the LLM.  This constrains the model to only use
# the provided context (no hallucinating from training data) and to cite
# which article/section each fact came from.
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a jazz piano and music theory expert.
    Answer the user's question using ONLY the context passages provided below.
    If the context does not contain enough information, say so honestly.
    Cite your sources by referencing the article title and section in parentheses.
""")


def retrieve(collection, question, k=DEFAULT_K):
    """
    Perform semantic search against the ChromaDB collection.

    How it works:
      - ChromaDB embeds the question using the same model (all-MiniLM-L6-v2)
        that was used to embed the chunks at index time.
      - It computes cosine distance between the question vector and all stored
        chunk vectors, then returns the K closest matches.
      - Each result includes the original text, its metadata, and the distance
        score (lower = more similar; 0.0 = identical, 2.0 = opposite).

    Returns a list of hit dicts sorted by relevance (most similar first):
      [{"text": "...", "metadata": {...}, "distance": 0.23}, ...]
    """
    results = collection.query(query_texts=[question], n_results=k)
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits


def build_context_block(hits):
    """
    Format retrieved chunks into a numbered text block for the LLM prompt.

    The LLM receives this as its "context" -- the only information it's
    allowed to use when answering.  Each chunk is labeled with a number
    and its source (article title + section) so the LLM can cite them.

    Example output:
      [1] Article: Tritone substitution | Section: Introduction
      The tritone substitution is a common chord substitution...

      [2] Article: Chord progression | Section: Blues changes
      ...
    """
    lines = []
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        lines.append(f"[{i}] Article: {m['title']} | Section: {m['section']}")
        lines.append(hit["text"])
        lines.append("")
    return "\n".join(lines)


def query_ollama(question, context, model=DEFAULT_MODEL):
    """
    Send the question + retrieved context to a local Ollama LLM and get an answer.

    How it works:
      - Imports the ollama Python client (HTTP client that talks to the Ollama
        server running on localhost:11434)
      - Builds a chat with two messages:
          * system: instructs the model to only use the provided context
          * user: the context block + the actual question
      - Returns (answer_text, None) on success, or (None, error_message) on failure

    Error handling:
      - ConnectionError: Ollama server isn't running
      - ResponseError with "not found": the requested model hasn't been pulled
    """
    try:
        import ollama as ol
    except ImportError:
        return None, "The 'ollama' Python package is not installed. Run: pip install ollama"

    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    try:
        response = ol.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response["message"]["content"], None
    except ConnectionError:
        return None, (
            "Cannot connect to Ollama. Make sure Ollama is running.\n"
            "  Install: https://ollama.com/download\n"
            "  Then:    ollama pull " + model
        )
    except ol.ResponseError as e:
        if "not found" in str(e).lower():
            return None, f"Model '{model}' not found. Run: ollama pull {model}"
        raise


def main():
    """
    End-to-end RAG query pipeline:
      1. Parse command-line arguments (question, number of chunks, model, etc.)
      2. Open the persisted ChromaDB collection
      3. Retrieve the K most relevant chunks via semantic search
      4. Display the retrieved chunks with similarity scores
      5. (Unless --no-llm) Send chunks + question to Ollama for a natural
         language answer, then display the answer with source citations
    """
    # --- Parse arguments ---
    parser = argparse.ArgumentParser(description="Query the Jazz Piano RAG system")
    parser.add_argument("question", help="Your question about jazz piano or music theory")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of chunks to retrieve")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--no-llm", action="store_true", help="Show retrieved chunks only, skip LLM")
    args = parser.parse_args()

    # --- Connect to the existing vector store ---
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    print(f"\nSearching {collection.count()} chunks for: \"{args.question}\"\n")

    # --- Retrieve relevant chunks ---
    hits = retrieve(collection, args.question, k=args.k)

    # --- Display retrieved chunks ---
    print("=" * 60)
    print("RETRIEVED CHUNKS")
    print("=" * 60)
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        # Convert cosine distance to similarity: similarity = 1 - distance
        sim = 1 - hit["distance"]
        print(f"\n[{i}] {m['title']} > {m['section']}  (similarity: {sim:.2%})")
        print("-" * 60)
        print(textwrap.fill(hit["text"], width=80))

    if args.no_llm:
        return

    # --- Generate LLM answer ---
    print("\n" + "=" * 60)
    print("GENERATING ANSWER...")
    print("=" * 60)

    context = build_context_block(hits)
    answer, error = query_ollama(args.question, context, model=args.model)

    if error:
        print(f"\n[Ollama unavailable] {error}")
        print("\nYou can still see the retrieved chunks above.")
        print("To enable LLM answers, install Ollama and pull a model:")
        print(f"  1. Download from https://ollama.com/download")
        print(f"  2. ollama pull {args.model}")
        print(f"  3. Re-run this script")
        return

    # --- Display answer with citations ---
    print(f"\n{answer}")
    print("\n" + "-" * 60)
    print("Sources used:")
    seen = set()
    for hit in hits:
        m = hit["metadata"]
        key = (m["title"], m["section"])
        if key not in seen:
            seen.add(key)
            print(f"  - {m['title']} > {m['section']}  ({m.get('source_url', '')})")


if __name__ == "__main__":
    main()
