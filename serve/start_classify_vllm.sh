#!/bin/bash
.venv/bin/vllm serve iRanadheer/lancet_qwen35_4b_full_merged \
  --served-model-name lancet-classify \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 131072 \
  --gpu-memory-utilization 0.95
