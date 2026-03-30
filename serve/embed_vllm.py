# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "modal",
# ]
# ///

"""
Serve Qwen3-Embedding-0.6B on Modal with vLLM.

Deploy:
    modal deploy serve/embed_vllm.py

Test:
    modal run serve/embed_vllm.py

OpenAI client:
    from openai import OpenAI
    client = OpenAI(base_url="https://<workspace>--lancet-embeddings-serve.modal.run/v1", api_key="not-needed")
    response = client.embeddings.create(model="Qwen/Qwen3-Embedding-0.6B", input=["Hello world"])
"""

import modal

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.18.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("lancet-embeddings")

N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H200:{N_GPU}",
    scaledown_window=60 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=1000)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    import json
    pooler_config = json.dumps({"pooling_type": "MEAN", "enable_chunked_processing": True, "max_embed_len": 32768})

    cmd = [
        "vllm", "serve",
        MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--dtype", "auto",
        "--max-model-len", "8192",
        "--max-num-seqs", "512",
        "--gpu-memory-utilization", "0.90",
        "--runner", "pooling",
        "--pooler-config", pooler_config,
        "--trust-remote-code",
        "--tensor-parallel-size", str(N_GPU),
        "--uvicorn-log-level=info",
    ]

    print(cmd)
    subprocess.Popen(cmd)


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES):
    import aiohttp, json

    url = await serve.get_web_url.aio()

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Health check: {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Health check passed")

        payload = {
            "model": MODEL_NAME,
            "input": ["Climate change is causing rising sea levels worldwide."],
        }
        headers = {"Content-Type": "application/json"}

        async with session.post("/v1/embeddings", json=payload, headers=headers) as resp:
            result = await resp.json()
            embedding = result["data"][0]["embedding"]
            print(f"Embedding dim: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
    print("Done!")
