"""
Download remaining articles using the MediaWiki API directly.
Skips articles that already exist in the articles/ directory.
"""

import json
import os
import time
from datetime import datetime, timezone

import requests

ARTICLES = [
    "Chord progression",
    "Ii-V-I progression",
    "Circle of fifths",
    "Voicing (music)",
    "Comping (jazz)",
    "Stride piano",
    "Boogie-woogie",
    "Jazz scale",
    "Pentatonic scale",
    "Rhythm changes",
    "Piano blues",
]

OUTPUT_DIR = "articles"
API_URL = "https://en.wikipedia.org/w/api.php"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "RAG_Jazz_Piano/1.0 (educational project)"})


def get_page_content(title):
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|info|categories",
        "explaintext": True,
        "exsectionformat": "wiki",
        "inprop": "url",
        "cllimit": "max",
        "format": "json",
    }
    r = SESSION.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data["query"]["pages"]
    page_id = next(iter(pages))
    if page_id == "-1":
        return None
    return pages[page_id]


def get_page_sections(title):
    params = {
        "action": "parse",
        "page": title,
        "prop": "sections",
        "format": "json",
    }
    r = SESSION.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "parse" not in data:
        return []
    return data["parse"]["sections"]


def get_section_text(title, section_index):
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "wiki",
        "exchars": 100000,
        "exlimit": 1,
        "rvsection": section_index,
        "format": "json",
    }
    r = SESSION.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data["query"]["pages"]
    page_id = next(iter(pages))
    return pages[page_id].get("extract", "")


def build_article(title, retries=3):
    for attempt in range(retries):
        try:
            page = get_page_content(title)
            if page is None:
                print(f"  [SKIP] '{title}' not found")
                return None

            time.sleep(1)
            raw_sections = get_page_sections(page.get("title", title))

            categories = [
                c["title"] for c in page.get("categories", [])
            ]

            sections = []
            for s in raw_sections:
                sections.append({
                    "title": s["line"],
                    "depth": int(s["toclevel"]),
                    "number": s["number"],
                })

            article = {
                "metadata": {
                    "title": page.get("title", title),
                    "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
                    "summary": page.get("extract", "")[:500],
                    "categories": categories,
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "source": "wikipedia",
                    "language": "en",
                },
                "full_text": page.get("extract", ""),
                "sections": sections,
            }
            return article

        except Exception as e:
            wait = 3 * (attempt + 1)
            print(f"  [RETRY {attempt + 1}/{retries}] {e} (waiting {wait}s)")
            time.sleep(wait)

    print(f"  [FAIL] Could not download '{title}' after {retries} attempts")
    return None


def slugify(title):
    return (
        title.lower()
        .replace(" ", "_")
        .replace("–", "-")
        .replace("(", "")
        .replace(")", "")
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing = set(os.listdir(OUTPUT_DIR))
    downloaded = []

    for title in ARTICLES:
        filename = slugify(title) + ".json"
        if filename in existing:
            print(f"Already exists: {filename}, skipping")
            continue

        print(f"Downloading: {title}")
        article = build_article(title)
        if article is None:
            continue

        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(article, f, indent=2, ensure_ascii=False)

        downloaded.append(article["metadata"]["title"])
        print(f"  -> saved {filepath}")
        time.sleep(2)

    print(f"\nDone! {len(downloaded)} additional articles saved.")
    if downloaded:
        print("New articles:", ", ".join(downloaded))


if __name__ == "__main__":
    main()
