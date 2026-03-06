"""
Translate climate keywords into target languages using Gemini.
Uses ISO 639-3 codes (same as EventRegistry article 'lang' field).

Usage:
    python translate_keywords.py                  # translate for all major languages
    python translate_keywords.py --langs fra spa  # translate for specific languages
"""
import json
import os
import time
from google import genai
from dotenv import load_dotenv
from newsapi import CLIMATE_KEYWORDS_EN

load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
MODEL = 'gemini-3.1-pro-preview'

TRANSLATIONS_FILE = 'data/keyword_translations.json'

# Major languages with ISO 639-3 codes (as used by EventRegistry)
LANGUAGES = {
    "ara": "Arabic",
    "ben": "Bengali",
    "cmn": "Mandarin Chinese",  # ER uses 'zho' sometimes — we map both
    "deu": "German",
    "fra": "French",
    "hin": "Hindi",
    "ind": "Indonesian",
    "ita": "Italian",
    "jpn": "Japanese",
    "kor": "Korean",
    "msa": "Malay",
    "nld": "Dutch",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "spa": "Spanish",
    "swa": "Swahili",
    "tha": "Thai",
    "tur": "Turkish",
    "urd": "Urdu",
    "vie": "Vietnamese",
    "zho": "Chinese",
    # Additional languages found in Phase 2
    "hrv": "Croatian",
    "ces": "Czech",
    "est": "Estonian",
    "sqi": "Albanian",
    "hye": "Armenian",
    "aze": "Azerbaijani",
    "bul": "Bulgarian",
    "srp": "Serbian",
    "ell": "Greek",
    "cat": "Catalan",
    "bel": "Belarusian",
    # Additional languages from Phase 2 batch 2
    "dan": "Danish",
    "fas": "Persian",
    "heb": "Hebrew",
    "isl": "Icelandic",
    "kaz": "Kazakh",
    "lav": "Latvian",
    "lit": "Lithuanian",
    "mal": "Malayalam",
    "mkd": "Macedonian",
    "nob": "Norwegian Bokmål",
    "pol": "Polish",
}


def build_prompt(language_name, keywords):
    kw_list = "\n".join(f"- {kw}" for kw in keywords)
    return f"""Translate the following English climate-related keyword phrases into {language_name}.

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


def translate_keywords(lang_code, language_name, keywords=None):
    """Translate keywords into the target language using Gemini."""
    if keywords is None:
        keywords = CLIMATE_KEYWORDS_EN

    prompt = build_prompt(language_name, keywords)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    text = response.text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    translations = json.loads(text)

    # Flatten: split " | " alternatives into unique phrases (preserving order)
    seen = set()
    all_phrases = []
    for item in translations:
        for phrase in item["translated"].split(" | "):
            phrase = phrase.strip()
            if phrase and phrase not in seen:
                seen.add(phrase)
                all_phrases.append(phrase)

    return {
        "lang_code": lang_code,
        "language": language_name,
        "keywords": all_phrases,
        "raw_translations": translations,
    }


def load_existing():
    if os.path.exists(TRANSLATIONS_FILE):
        with open(TRANSLATIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_translations(data):
    os.makedirs(os.path.dirname(TRANSLATIONS_FILE), exist_ok=True)
    with open(TRANSLATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def translate_all(lang_codes=None):
    """Translate keywords for all (or specified) languages."""
    data = load_existing()

    # Always include English as-is
    if "eng" not in data:
        data["eng"] = {
            "lang_code": "eng",
            "language": "English",
            "keywords": CLIMATE_KEYWORDS_EN,
            "raw_translations": [{"en": kw, "translated": kw} for kw in CLIMATE_KEYWORDS_EN],
        }

    targets = lang_codes if lang_codes else list(LANGUAGES.keys())

    for code in targets:
        if code in data:
            print(f"  SKIP {code} ({LANGUAGES[code]}) — already translated")
            continue

        language_name = LANGUAGES.get(code)
        if not language_name:
            print(f"  SKIP {code} — unknown language code")
            continue

        print(f"  Translating to {code} ({language_name})...")
        try:
            result = translate_keywords(code, language_name)
            data[code] = result
            save_translations(data)
            print(f"    -> {len(result['keywords'])} phrases")
        except Exception as e:
            print(f"    ERROR: {e}")

        time.sleep(1)

    save_translations(data)
    print(f"\nTranslations saved to {TRANSLATIONS_FILE}")
    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Translate climate keywords')
    parser.add_argument('--langs', nargs='+', help='Specific language codes to translate')
    args = parser.parse_args()

    translate_all(args.langs)
