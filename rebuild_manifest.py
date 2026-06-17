"""Rebuild _manifest.json from all article files in the articles/ directory."""

import json
import os
from datetime import datetime, timezone

OUTPUT_DIR = "articles"

articles = []
for filename in sorted(os.listdir(OUTPUT_DIR)):
    if filename.startswith("_") or not filename.endswith(".json"):
        continue
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    articles.append({
        "title": meta.get("title", filename),
        "file": filename,
        "url": meta.get("url", ""),
        "summary_preview": meta.get("summary", "")[:120] + "...",
    })

manifest = {
    "total_articles": len(articles),
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "articles": articles,
}

manifest_path = os.path.join(OUTPUT_DIR, "_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"Manifest rebuilt with {len(articles)} articles.")
