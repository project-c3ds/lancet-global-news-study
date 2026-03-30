# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "modal",
# ]
# ///

"""
Serve Lancet Climate-Health classifier on Modal with vLLM.

Deploy:
    modal deploy ft/serve_vllm.py

Test:
    modal run ft/serve_vllm.py

Then use the OpenAI client:
    from openai import OpenAI
    client = OpenAI(base_url="https://<your-workspace>--lancet-vllm-serve.modal.run/v1", api_key="not-needed")
"""

import json
from typing import Any

import aiohttp
import modal

MODEL_NAME = "iRanadheer/lancet_qwen35_4b_full_merged"
MODEL_REVISION = "c1971a54c1d057e0e5ac948ff4471ccc959e489c"

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

app = modal.App("lancet-vllm")

N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H200:{N_GPU}",
    scaledown_window=15 * MINUTES,
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

    cmd = [
        "vllm", "serve",
        MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", "4096",
        "--max-num-seqs", "512",
        "--gpu-memory-utilization", "0.95",
        "--trust-remote-code",
        "--tensor-parallel-size", str(N_GPU),
        "--uvicorn-log-level=info",
    ]

    print(cmd)
    subprocess.Popen(cmd)


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES):
    url = await serve.get_web_url.aio()

    messages = [
        {"role": "system", "content": "You are an expert multilabel classifier for newspaper articles."},
        {"role": "user", "content": "### Article:\nClimate change causes more dengue fever in Southeast Asia\n\nResearchers found rising temperatures expand mosquito habitats.\n\nLet's work this out in a step by step way to be sure we have the right answer."},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Health check: {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Health check passed")

        payload = {"messages": messages, "model": MODEL_NAME, "stream": True, "max_tokens": 2000, "temperature": 0.01}
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

        async with session.post("/v1/chat/completions", json=payload, headers=headers) as resp:
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[len("data: "):]
                chunk = json.loads(line)
                print(chunk["choices"][0]["delta"].get("content", ""), end="")
        print()
