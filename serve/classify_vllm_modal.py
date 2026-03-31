# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "modal",
# ]
# ///

"""
Serve lancet classification model on Modal with vLLM.

Deploy:
    modal deploy serve/classify_vllm_modal.py

Test:
    modal run serve/classify_vllm_modal.py

OpenAI client:
    from openai import OpenAI
    client = OpenAI(base_url="https://<workspace>--lancet-classify-serve.modal.run/v1", api_key="not-needed")
"""

import modal

MODEL_NAME = "iRanadheer/lancet_qwen35_4b_full_merged"

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

app = modal.App("lancet-classify")

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

    cmd = [
        "vllm", "serve",
        MODEL_NAME,
        "--served-model-name", "lancet-classify",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--max-num-seqs", "1024",
        "--max-num-batched-tokens", "131072",
        "--gpu-memory-utilization", "0.95",
        "--trust-remote-code",
        "--tensor-parallel-size", str(N_GPU),
        "--uvicorn-log-level=info",
    ]

    print(cmd)
    subprocess.Popen(cmd)


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES):
    import aiohttp

    url = await serve.get_web_url.aio()

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Health check: {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Health check passed")

        payload = {
            "model": "lancet-classify",
            "messages": [
                {"role": "user", "content": "Classify: Climate change causes dengue fever rise."},
            ],
            "max_tokens": 200,
            "temperature": 0.01,
        }
        headers = {"Content-Type": "application/json"}

        async with session.post("/v1/chat/completions", json=payload, headers=headers) as resp:
            result = await resp.json()
            print(result["choices"][0]["message"]["content"])
    print("Done!")
