# Deploy the Doric  model to Modal

This guide will walk you through the process of deploying the Doric model to Modal[https://modal.com/].

## Requirements

- Python 3.12+
- uv (package manager)

## Setup

```sh
uv venv .venv
uv sync
```

## Deploy the model

Deploy the model from Hugging Face to Modal:

```sh
uv add modal
modal deploy vllm_doric.py
```

## Run the chat CLI

Run the chat CLI:

```sh
uv run python chat_cli.py
```

## Launch Web UI

To interact with the deployed model via a web interface, you can use [Open WebUI](https://github.com/open-webui/open-webui).

### Get the Modal endpoint URL

After deploying, Modal will provide a URL for your endpoint. You can find it in the deployment output or by running:

```sh
modal app list
```

The endpoint URL will be in the format: `https://your-workspace-name--doric-vllm-inference-serve.modal.run`

### Run Open WebUI

Launch Open WebUI using Docker, pointing it to your Modal endpoint:

```sh
docker run -d \
    --name open-webui \
    -p 3000:8080 \
    -v open-webui:/app/backend/data \
    -e OPENAI_API_BASE_URL=https://your-workspace-name--doric-vllm-inference-serve.modal.run/v1 \
    --restart always \
    ghcr.io/open-webui/open-webui:main
```

**Note:** Replace `https://your-workspace-name--doric-vllm-inference-serve.modal.run` with your actual Modal endpoint URL.

Example:

```sh
docker run -d \
    --name open-webui \
    -p 3000:8080 \
    -v open-webui:/app/backend/data \
    -e OPENAI_API_BASE_URL=https://siquick--doric-vllm-inference-serve.modal.run/v1 \
    --restart always \
    ghcr.io/open-webui/open-webui:main
```

Once running, access the Web UI at `http://localhost:3000` in your browser.
