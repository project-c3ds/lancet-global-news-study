"""Generate Reverse Engineered Chain-of-Thought (RECoT) training data.

Takes labeled articles (with yes/no labels for climate, health, health_climate_impact)
and asks a teacher model to produce expert reasoning that arrives at those labels.

Usage:
    python -m recot.generate --input results/labeled/annotated/to_label_english.csv --model 1
    python -m recot.generate --input results/labeled/annotated/to_label_english.csv --model 1 --max 5
    python -m recot.generate --input results/labeled/annotated/to_label_english.csv --model 1 2 --concurrency 10
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List

import litellm
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Add parent directory to path so we can import prompts
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from prompts.base_prompt import system_instruction, cot_trigger, recot_trigger


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    id: int
    name: str
    provider: str
    model_id: str
    temperature: float = 0
    max_tokens: int = 4000
    api_base: str = None
    extra_body: Dict[str, Any] = field(default_factory=dict)


def load_model_configs(path: str = "models.json") -> List[ModelConfig]:
    with open(path) as f:
        data = json.load(f)
    return [
        ModelConfig(
            id=m["id"],
            name=m["name"],
            provider=m["provider"],
            model_id=m["model_id"],
            temperature=m.get("temperature", 0),
            max_tokens=m.get("max_tokens", 4000),
            api_base=m.get("api_base"),
            extra_body=m.get("extra_body", {}),
        )
        for m in data["models"]
    ]


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
LABEL_COLS = ["climate", "health", "health_climate_impact"]


def row_to_true_labels(row: dict) -> dict:
    """Convert yes/no strings to the YAML-style true/false dict the model outputs."""
    return {
        "climate_change": row.get("climate", "no").strip().lower() == "yes",
        "health": row.get("health", "no").strip().lower() == "yes",
        "health_effects_of_climate_change": row.get("health_climate_impact", "no").strip().lower() == "yes",
    }


def labels_to_yaml(labels: dict) -> str:
    lines = []
    for key in ["climate_change", "health", "health_effects_of_climate_change"]:
        lines.append(f"{key}:")
        lines.append(f"  label: {str(labels[key]).lower()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message building & API call
# ---------------------------------------------------------------------------
def build_messages(text: str, true_labels: dict) -> list:
    """Build RECoT messages: system instruction + text with true labels + recot trigger."""
    true_labels_yaml = labels_to_yaml(true_labels)
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_instruction,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": f"### Article:\n{text}\n\n### True Labels:\n{true_labels_yaml}\n\n{recot_trigger}",
        },
    ]


def generate_one(model_config: ModelConfig, text: str, true_labels: dict) -> dict:
    """Generate RECoT reasoning for a single example."""
    messages = build_messages(text, true_labels)
    full_messages = messages + [{"role": "user", "content": cot_trigger}]

    kwargs = dict(
        model=model_config.model_id,
        messages=full_messages,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        timeout=120,
    )
    if model_config.api_base:
        kwargs["api_base"] = model_config.api_base
    if model_config.extra_body:
        kwargs["extra_body"] = model_config.extra_body

    response = litellm.completion(**kwargs)
    return {
        "model": model_config.name,
        "response": response.choices[0].message.content,
        "usage": response.usage.model_dump(),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(path: str) -> List[dict]:
    """Load labeled CSV(s). Accepts a file or directory of CSVs.
    Only keeps rows where at least the 'climate' column is filled."""
    if os.path.isdir(path):
        frames = []
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".csv"):
                frames.append(pd.read_csv(os.path.join(path, fname)))
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)
    df = df[df["climate"].notna() & (df["climate"] != "")]
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
def run_batch(
    data: List[dict],
    model_configs: List[ModelConfig],
    output_path: str,
    concurrency: int = 5,
):
    """Process all examples across all models, writing JSONL. Supports resume."""
    already_done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    already_done.add(row["article_id"])
                except Exception:
                    pass
    if already_done:
        print(f"Resuming -- {len(already_done)} already processed, skipping them")

    remaining = [r for r in data if str(r.get("article_id", "")) not in already_done]
    total = len(remaining) * len(model_configs)
    if total == 0:
        print("All examples already processed. Nothing to do.")
        return

    print(f"Processing {len(remaining)} remaining examples ({len(data) - len(remaining)} skipped)")

    with open(output_path, "a") as f, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for row in remaining:
            true_labels = row_to_true_labels(row)
            text = f"{row.get('title', '')}\n\n{row.get('content', '')}"
            for mc in model_configs:
                fut = pool.submit(generate_one, mc, text, true_labels)
                futures[fut] = (row, mc.name, true_labels)

        for fut in tqdm(as_completed(futures), total=total, desc="RECoT"):
            row, model_name, true_labels = futures[fut]
            try:
                result = fut.result()
                out = {
                    "article_id": str(row.get("article_id", "")),
                    "title": row.get("title", ""),
                    "content": row.get("content", ""),
                    "language": row.get("language", ""),
                    "country": row.get("country", ""),
                    "source": row.get("source", ""),
                    "true_labels": true_labels,
                    "model": model_name,
                    "response": result["response"],
                    "usage": result["usage"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(f"Error [{model_name}] article_id={row.get('article_id')}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate RECoT training data")
    parser.add_argument("--input", required=True, help="Input labeled CSV file")
    parser.add_argument("--models", nargs="+", type=int, default=[1], help="Model IDs from models.json")
    parser.add_argument("--output", help="Output JSONL path (default: data/ft/<date>/recot.jsonl)")
    parser.add_argument("--max", type=int, help="Limit number of examples")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    all_models = load_model_configs()
    model_configs = [m for m in all_models if m.id in args.models]
    if not model_configs:
        print(f"No models found with IDs {args.models}. Check models.json.")
        return

    data = load_data(args.input)
    if args.max:
        data = data[: args.max]

    if args.output:
        output_path = args.output
    else:
        run_dir = os.path.join("data/ft", date.today().isoformat())
        os.makedirs(run_dir, exist_ok=True)
        output_path = os.path.join(run_dir, "recot.jsonl")

    print(f"Processing {len(data)} examples x {len(model_configs)} models -> {output_path}")
    run_batch(data, model_configs, output_path, args.concurrency)
    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
