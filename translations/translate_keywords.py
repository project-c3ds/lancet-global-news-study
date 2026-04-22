"""Translate climate and health seed keywords into target languages with Claude Opus 4.6.

Reads English seed phrases from `translations/keywords/{topic}_eng.txt` and writes
news-register translations into `translations/keyword_translations.json`. Output is
topic-keyed:

    {"climate": {"eng": [...], "ara": [...], ...},
     "health":  {"eng": [...], "ara": [...], ...}}

The prompt instructs the model to produce terms as they naturally appear in the
target-language news register (not literal translations), and to return multiple
common alternatives separated by " | ". The script expands those alternatives
into a flat deduplicated list per language.

Usage:
    python translations/translate_keywords.py --topic health
    python translations/translate_keywords.py --topic climate --langs fra spa
    python translations/translate_keywords.py --topic both

Resume-safe: by default skips languages already present. Pass --force to overwrite.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve().parent
SEED_DIR = HERE / "keywords"
OUTPUT_JSON = HERE / "keyword_translations.json"
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-6")

# ISO-639-3 code → human name. Matches the language set used in the paper.
LANGUAGES = {
    "ara": "Arabic", "aze": "Azerbaijani", "bel": "Belarusian", "ben": "Bengali",
    "bul": "Bulgarian", "cat": "Catalan", "ces": "Czech", "cmn": "Mandarin Chinese",
    "dan": "Danish", "deu": "German", "ell": "Greek", "est": "Estonian",
    "fas": "Persian", "fra": "French", "heb": "Hebrew", "hin": "Hindi",
    "hrv": "Croatian", "hye": "Armenian", "ind": "Indonesian", "isl": "Icelandic",
    "ita": "Italian", "jpn": "Japanese", "kaz": "Kazakh", "kor": "Korean",
    "lav": "Latvian", "lit": "Lithuanian", "mal": "Malayalam", "mkd": "Macedonian",
    "msa": "Malay", "nld": "Dutch", "nob": "Norwegian Bokmål", "pol": "Polish",
    "por": "Portuguese", "ron": "Romanian", "rus": "Russian", "spa": "Spanish",
    "sqi": "Albanian", "srp": "Serbian", "swa": "Swahili", "tha": "Thai",
    "tur": "Turkish", "urd": "Urdu", "vie": "Vietnamese", "zho": "Chinese",
}


def load_seeds(topic: str) -> list[str]:
    path = SEED_DIR / f"{topic}_eng.txt"
    seeds = []
    with path.open() as f:
        for line in f:
            line = line.strip().strip('"').strip("'").strip()
            if line:
                seeds.append(line)
    return seeds


def load_output() -> dict:
    if OUTPUT_JSON.exists():
        with OUTPUT_JSON.open() as f:
            data = json.load(f)
    else:
        data = {}
    data.setdefault("climate", {})
    data.setdefault("health", {})
    return data


def save_output(data: dict) -> None:
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_prompt(topic: str, language_name: str, seeds: list[str]) -> str:
    topic_label = "climate-related" if topic == "climate" else "health-related"
    kw_list = "\n".join(f"- {kw}" for kw in seeds)
    return f"""Translate the following English {topic_label} keyword phrases into {language_name}.

These translations will be used to search for news articles, so translate them as they would naturally appear in {language_name}-language news reporting — not as literal word-for-word translations.

For each English phrase, provide the most natural equivalent(s) that a {language_name}-language journalist would use. If a phrase has multiple common translations in news contexts, include all of them (separated by " | ").

Important:
- Include both singular and plural forms where the target language distinguishes them
- Use the forms commonly seen in news headlines and articles
- Do NOT transliterate — use the native script for the language

Return ONLY a JSON array of objects, each with "en" and "translated" keys. Example:
[
  {{"en": "climate change", "translated": "changement climatique | changements climatiques"}},
  ...
]

English phrases:
{kw_list}"""


def parse_response(text: str) -> list[dict]:
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def flatten_alternatives(pairs: list[dict]) -> list[str]:
    seen = set()
    out = []
    for item in pairs:
        for phrase in item.get("translated", "").split(" | "):
            phrase = phrase.strip()
            if phrase and phrase not in seen:
                seen.add(phrase)
                out.append(phrase)
    return out


def translate_language(client: Anthropic, topic: str, language_name: str, seeds: list[str]) -> list[str]:
    prompt = build_prompt(topic, language_name, seeds)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text
    parsed = parse_response(raw)
    return flatten_alternatives(parsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", choices=["climate", "health", "both"], default="both")
    parser.add_argument("--langs", nargs="+", help="Subset of ISO-639-3 codes; default = all")
    parser.add_argument("--force", action="store_true", help="Re-translate even if already present")
    args = parser.parse_args()

    topics = ["climate", "health"] if args.topic == "both" else [args.topic]
    targets = args.langs if args.langs else list(LANGUAGES.keys())

    client = Anthropic()
    data = load_output()

    for topic in topics:
        seeds = load_seeds(topic)
        print(f"[{topic}] {len(seeds)} English seeds")
        if "eng" not in data[topic] or args.force:
            data[topic]["eng"] = list(seeds)
            save_output(data)

        for code in targets:
            if code == "eng" or code not in LANGUAGES:
                continue
            name = LANGUAGES[code]
            if code in data[topic] and not args.force:
                print(f"  skip {code:<4} ({name}) — already present ({len(data[topic][code])} phrases)")
                continue
            print(f"  {topic}/{code:<4} ({name}) ...", end=" ", flush=True)
            try:
                phrases = translate_language(client, topic, name, seeds)
                data[topic][code] = phrases
                save_output(data)
                print(f"{len(phrases)} phrases")
            except Exception as e:
                print(f"ERROR: {e}")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
