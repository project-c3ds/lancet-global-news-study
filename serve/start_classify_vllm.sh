#!/bin/bash
.venv/bin/vllm serve /workspace/lancet-global-news-study/lancet_qwen35_4b_full_merge \
  --served-model-name lancet-classify \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 131072 \
  --gpu-memory-utilization 0.95
