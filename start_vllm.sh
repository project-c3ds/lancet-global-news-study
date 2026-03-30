#!/bin/bash
.venv/bin/vllm serve Qwen/Qwen3-Embedding-0.6B \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 65536 \
  --gpu-memory-utilization 0.95 \
  --runner pooling \
  --pooler-config '{"pooling_type": "MEAN", "enable_chunked_processing": true, "max_embed_len": 32768}'
