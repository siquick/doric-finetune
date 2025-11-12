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
