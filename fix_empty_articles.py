"""Re-download articles that came through empty, using redirects and correct titles."""

import json
import time
import requests
from datetime import datetime, timezone

API_URL = "https://en.wikipedia.org/w/api.php"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "RAG_Jazz_Piano/1.0 (educational project)"})

FIXES = {
    "articles/ii-v-i_progression.json": "ii–V–I progression",
    "articles/stride_piano.json": "Stride piano",
    "articles/piano_blues.json": "Piano blues",
}


def fetch_article(title):
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|info|categories",
        "explaintext": True,
        "exsectionformat": "wiki",
        "inprop": "url",
        "cllimit": "max",
        "redirects": 1,
        "format": "json",
    }
    r = SESSION.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data["query"]["pages"]
    pid = next(iter(pages))
    if pid == "-1":
        return None
    page = pages[pid]

    time.sleep(2)
    params2 = {
        "action": "parse",
        "page": page.get("title", title),
        "prop": "sections",
        "redirects": 1,
        "format": "json",
    }
    r2 = SESSION.get(API_URL, params=params2, timeout=30)
    r2.raise_for_status()
    secs = r2.json().get("parse", {}).get("sections", [])

    return {
        "metadata": {
            "title": page.get("title", title),
            "url": page.get("fullurl", ""),
            "summary": page.get("extract", "")[:500],
            "categories": [c["title"] for c in page.get("categories", [])],
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "source": "wikipedia",
            "language": "en",
        },
        "full_text": page.get("extract", ""),
        "sections": [
            {"title": s["line"], "depth": int(s["toclevel"]), "number": s["number"]}
            for s in secs
        ],
    }


for filepath, title in FIXES.items():
    print(f"Fetching: {title}")
    article = fetch_article(title)
    if article is None or not article["full_text"]:
        print(f"  [FAIL] still empty for '{title}'")
        continue
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(article, f, indent=2, ensure_ascii=False)
    print(f"  -> saved {filepath} ({len(article['full_text'])} chars)")
    time.sleep(3)
