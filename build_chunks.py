"""
Step 1: Chunk articles and attach metadata.

WHY CHUNK?
  Embedding models and LLMs work best on short, focused passages.
  A full article is too long to embed as a single vector -- the meaning gets
  diluted.  By splitting into overlapping ~500-char chunks we get vectors that
  capture specific ideas ("tritone substitution resolves to I") rather than
  vague article-level topics.

  Overlap (100 chars) ensures that sentences split across a boundary still
  appear intact in at least one chunk.

METADATA attached to every chunk:
  - title        : article it came from (for filtering & attribution)
  - source_url   : Wikipedia link (for citations)
  - section      : nearest section heading (for context)
  - chunk_index  : position in the article (for ordering)
"""

import json
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

ARTICLES_DIR = "articles"
OUTPUT_FILE = "chunks.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

SKIP_SECTIONS = {
    "See also", "References", "External links", "Sources",
    "Further reading", "Notes", "Bibliography",
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def build_section_map(full_text):
    """Parse '== Section ==' headers from full_text to map character offsets
    to section names.  Used for articles downloaded via the direct MediaWiki API
    where sections don't carry their own text field."""
    header_re = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$", re.MULTILINE)
    markers = [(0, "Introduction")]
    for m in header_re.finditer(full_text):
        markers.append((m.start(), m.group(2)))
    return markers


def section_at(markers, char_offset):
    """Given sorted (offset, name) markers, return the section name active at
    char_offset."""
    current = "Introduction"
    for offset, name in markers:
        if offset > char_offset:
            break
        current = name
    return current


def collect_section_texts(sections, depth=0):
    """Recursively collect (section_title, text) pairs from wikipedia-api
    style nested sections."""
    results = []
    for sec in sections:
        title = sec.get("title", "")
        text = sec.get("text", "")
        if title in SKIP_SECTIONS:
            continue
        if text.strip():
            results.append((title, text))
        for sub in sec.get("subsections", []):
            sub_title = sub.get("title", "")
            sub_text = sub.get("text", "")
            if sub_title in SKIP_SECTIONS:
                continue
            if sub_text.strip():
                results.append((sub_title, sub_text))
            results.extend(collect_section_texts(sub.get("subsections", []), depth + 1))
    return results


def has_section_text(sections):
    """Check whether sections carry their own text (wikipedia-api format)."""
    for sec in sections:
        if sec.get("text", "").strip():
            return True
        for sub in sec.get("subsections", []):
            if sub.get("text", "").strip():
                return True
    return False


def chunk_article(article):
    meta = article["metadata"]
    title = meta["title"]
    url = meta.get("url", meta.get("source_url", ""))
    sections = article.get("sections", [])
    full_text = article.get("full_text", "")

    chunks = []

    if has_section_text(sections):
        # Wikipedia-api format: sections carry text
        # Chunk the summary / intro (text before first section)
        intro_parts = collect_section_texts(sections)
        intro_text = full_text.split("\n\n")[0] if full_text else ""
        if intro_text.strip():
            for i, chunk_text in enumerate(splitter.split_text(intro_text)):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "title": title,
                        "source_url": url,
                        "section": "Introduction",
                        "chunk_index": len(chunks),
                    },
                })

        for sec_title, sec_text in intro_parts:
            for chunk_text in splitter.split_text(sec_text):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "title": title,
                        "source_url": url,
                        "section": sec_title,
                        "chunk_index": len(chunks),
                    },
                })
    else:
        # Direct MediaWiki API format: all text in full_text with == headers ==
        clean = re.sub(r"^={2,}\s*.+?\s*={2,}\s*$", "", full_text, flags=re.MULTILINE)
        lines = clean.strip().split("\n")
        filtered_lines = [
            ln for ln in lines
            if ln.strip() and ln.strip() not in SKIP_SECTIONS
        ]
        clean_text = "\n".join(filtered_lines)

        section_markers = build_section_map(full_text)
        raw_chunks = splitter.split_text(clean_text)

        search_start = 0
        for chunk_text in raw_chunks:
            pos = full_text.find(chunk_text[:80], search_start)
            if pos == -1:
                pos = search_start
            sec_name = section_at(section_markers, pos)
            if sec_name in SKIP_SECTIONS:
                continue
            search_start = max(search_start, pos + 1)

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "title": title,
                    "source_url": url,
                    "section": sec_name,
                    "chunk_index": len(chunks),
                },
            })

    return chunks


def main():
    all_chunks = []
    article_stats = {}

    for filename in sorted(os.listdir(ARTICLES_DIR)):
        if filename.startswith("_") or not filename.endswith(".json"):
            continue
        filepath = os.path.join(ARTICLES_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            article = json.load(f)

        chunks = chunk_article(article)
        article_stats[article["metadata"]["title"]] = len(chunks)
        all_chunks.extend(chunks)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Total chunks: {len(all_chunks)}")
    print(f"\nPer article:")
    for title, count in sorted(article_stats.items()):
        print(f"  {title}: {count} chunks")

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
