#!/bin/bash
.venv/bin/vllm serve Qwen/Qwen3-Embedding-0.6B \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 4096 \
  --max-num-batched-tokens 524288 \
  --gpu-memory-utilization 0.85 \
  --runner pooling \
  --pooler-config '{"pooling_type": "MEAN", "enable_chunked_processing": true, "max_embed_len": 32768}'
