# Translations

Keyword translations used to query NewsAPI (collection) and to compute BM25 relevance scores on the collected corpus (classification sampling). See appendix footnote 7.

## Contents

- `keywords/climate_eng.txt` — 27 climate-change English seed terms.
- `keywords/health_eng.txt` — 44 health English seed terms.
- `keywords/source_languages.csv` — source → primary-language mapping.
- `keyword_translations.json` — topic-keyed translations across 45 languages, generated with Claude Opus 4.6 using the instruction to produce terms as they naturally appear in the source-language news register (not literal word-for-word translations).
- `translate_keywords.py` — (re)generation script.

## JSON structure

```
{
  "climate": {"eng": [...], "ara": [...], "ben": [...], ..., "zho": [...]},
  "health":  {"eng": [...], "ara": [...], "ben": [...], ..., "zho": [...]}
}
```

Each leaf is a flat list of phrases. The generator splits multi-alternative model output on ` | ` and deduplicates, so a list of 44 seed phrases can expand to 50–90 surface forms per language (e.g., French `paludisme` / `malaria` coexist).

Consumers:

- `collection/collect_newsapi_climate.py` reads `climate[<lang>]` and `climate[eng]` to build per-source Event Registry queries.
- `collection/collect_newsapi_health.py` reads `health[<lang>]` and `health[eng]` for the health pass.
- `classification/build_bm25_scores.py` reads both topics to compute `bm25_climate`, `bm25_health`, and `bm25_avg` on `data/climate.db`, which feed the 5K stratified annotation sample.

## Regenerating

The committed JSON was produced with Claude Opus 4.6 via the Anthropic SDK. To re-run:

```bash
python translations/translate_keywords.py --topic health
python translations/translate_keywords.py --topic climate --langs fra spa
python translations/translate_keywords.py --topic both --force   # overwrite everything
```

The script is resume-safe — by default it skips (topic, language) pairs that are already populated. `CLAUDE_MODEL` env var selects a different Anthropic model if needed (default `claude-opus-4-6`).
