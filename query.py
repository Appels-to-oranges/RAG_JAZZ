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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "jazz_piano"
DEFAULT_K = 5
DEFAULT_MODEL = "llama3.2"

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a jazz piano and music theory expert.
    Answer the user's question using ONLY the context passages provided below.
    If the context does not contain enough information, say so honestly.
    Cite your sources by referencing the article title and section in parentheses.
""")


def retrieve(collection, question, k=DEFAULT_K):
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
    lines = []
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        lines.append(f"[{i}] Article: {m['title']} | Section: {m['section']}")
        lines.append(hit["text"])
        lines.append("")
    return "\n".join(lines)


def query_ollama(question, context, model=DEFAULT_MODEL):
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
    parser = argparse.ArgumentParser(description="Query the Jazz Piano RAG system")
    parser.add_argument("question", help="Your question about jazz piano or music theory")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of chunks to retrieve")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--no-llm", action="store_true", help="Show retrieved chunks only, skip LLM")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    print(f"\nSearching {collection.count()} chunks for: \"{args.question}\"\n")

    hits = retrieve(collection, args.question, k=args.k)

    print("=" * 60)
    print("RETRIEVED CHUNKS")
    print("=" * 60)
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        sim = 1 - hit["distance"]
        print(f"\n[{i}] {m['title']} > {m['section']}  (similarity: {sim:.2%})")
        print("-" * 60)
        print(textwrap.fill(hit["text"], width=80))

    if args.no_llm:
        return

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
