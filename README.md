# RAG Jazz Piano

A knowledge base of 20 Wikipedia articles on jazz piano and piano music theory, structured for Retrieval-Augmented Generation (RAG).

## Articles

| Topic | File |
|---|---|
| Jazz Piano | `jazz_piano.json` |
| Jazz Harmony | `jazz_harmony.json` |
| Chord (Music) | `chord_music.json` |
| Chord Progression | `chord_progression.json` |
| ii-V-I Progression | `ii-v-i_progression.json` |
| Circle of Fifths | `circle_of_fifths.json` |
| Blues Scale | `blues_scale.json` |
| Bebop Scale | `bebop_scale.json` |
| Jazz Improvisation | `jazz_improvisation.json` |
| Voicing (Music) | `voicing_music.json` |
| Comping (Jazz) | `comping_jazz.json` |
| Stride Piano | `stride_piano.json` |
| Boogie-Woogie | `boogie-woogie.json` |
| Modal Jazz | `modal_jazz.json` |
| Twelve-Bar Blues | `twelve-bar_blues.json` |
| Tritone Substitution | `tritone_substitution.json` |
| Jazz Scale | `jazz_scale.json` |
| Pentatonic Scale | `pentatonic_scale.json` |
| Rhythm Changes | `rhythm_changes.json` |
| Piano Blues | `piano_blues.json` |

## Article JSON Structure

Each article file contains:

```json
{
  "metadata": {
    "title": "...",
    "url": "https://en.wikipedia.org/wiki/...",
    "summary": "...",
    "categories": ["Category:...", "..."],
    "downloaded_at": "2026-06-17T...",
    "source": "wikipedia",
    "language": "en"
  },
  "full_text": "...",
  "sections": [
    {
      "title": "Section Name",
      "depth": 0,
      "text": "...",
      "subsections": [...]
    }
  ]
}
```

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

- `download_articles.py` — Primary downloader using `wikipedia-api`
- `download_remaining.py` — Fallback downloader using the MediaWiki REST API directly (handles rate limits)
- `rebuild_manifest.py` — Regenerates `articles/_manifest.json` from all article files

## Next Steps

- Add richer metadata (topic tags, difficulty level, key concepts per section)
- Chunk articles into embedding-sized passages
- Build vector index for semantic search
- Wire up RAG pipeline with LLM for query-time retrieval
