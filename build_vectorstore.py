"""
Step 2: Embed chunks and store them in ChromaDB.

WHAT IS HAPPENING HERE?
  1. Each chunk of text is converted into a *vector* (a list of 384 numbers)
     by the sentence-transformers model "all-MiniLM-L6-v2".  This model was
     trained so that texts with similar *meaning* produce vectors that are
     close together in 384-dimensional space.

  2. ChromaDB stores each vector alongside the original text and its metadata.
     Under the hood it builds an index so that nearest-neighbor lookups
     ("find the 5 vectors closest to this query vector") are fast.

  3. The database is persisted to disk at ./chroma_db/ so you only need to
     run this script once.  The query script reads from the same directory.

WHY all-MiniLM-L6-v2?
  - It runs locally (no API key, no cost).
  - 384 dimensions is small and fast, but still captures meaning well.
  - It is ChromaDB's default, so zero extra config.
"""

import json
import chromadb

CHUNKS_FILE = "chunks.json"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "jazz_piano"


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        documents = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"  Added batch {i // batch_size + 1} ({len(batch)} chunks)")

    print(f"\nDone! Collection '{COLLECTION_NAME}' has {collection.count()} vectors.")
    print(f"Persisted to {CHROMA_DIR}/")

    # Quick sanity check
    print("\n--- Sanity check: querying 'tritone substitution' ---")
    results = collection.query(query_texts=["tritone substitution"], n_results=3)
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  [{dist:.4f}] {meta['title']} > {meta['section']}")
        print(f"    {doc[:120]}...")


if __name__ == "__main__":
    main()
