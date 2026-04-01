#!/bin/bash
CONCURRENCY=${1:-800}
echo "Running all years with concurrency=$CONCURRENCY"

for YEAR in 2025 2024 2023 2022 2021 2020 9999; do
    echo "=== Starting year $YEAR ==="
    python serve/classify_offline.py --year $YEAR --concurrency $CONCURRENCY
done
