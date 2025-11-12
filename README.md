# Finetuning an LLM to answer naturally in Doric

End-to-end workflow for fine-tuning an open-source LLM to answer naturally in Doric. The dataset generation scripts live alongside the `finetune_notebook/`, which is the primary destination once new data is ready. Treat the dataset tooling as one stage in the larger “collect → transform → fine-tune” journey.

## Repository Layout

- `dataset_generation/`
  - `generate_topics_via_llm.py` – expands grouped topic lists with an LLM.
  - `generate_doric_dataset.py` – main async generator for Doric chat samples.
  - `transform_to_sharegpt.py` – reshapes generated rows into ShareGPT-style conversations.
- `datasets/` – recommended output directory for `.jsonl` datasets (create with `mkdir -p datasets`).
- `finetune_notebook/` – the main Colab/Notebook entry point for training OSS models with Unsloth/LoRA adapters; takes the ShareGPT-shaped data as input.
- `modal/` – optional remote execution helpers.
- `topics.json` / `topics.txt` – topic seeds for generation.
- `pyproject.toml`, `uv.lock` – dependency definitions consumed by `uv`.

## Requirements & Setup

- Python 3.12+
- Optional speedups: `orjson`, `rapidfuzz`, `tqdm`
- API access: set `OPENAI_API_KEY`, and optionally `MODEL`, `OPENAI_BASE_URL`
- Env loading: both generators auto-load `.env` at runtime.

Bootstrap the environment entirely with `uv`:

```bash
uv sync            # install deps declared in pyproject/uv.lock
mkdir -p datasets  # central location for emitted JSONL files
cp -p .env.template .env # copy the template to .env and add your variables - you only need to add the VLLM_URL variable if you want to deploy the model to Modal
```

## Typical Workflow

1. **Seed topics** – edit `topics.json`/`topics.txt` or run the topic expander to broaden coverage.
2. **Generate dataset** – write raw output to `datasets/doric_synth.jsonl` (tweak ratios, concurrency, etc.).
3. **Convert to ShareGPT** – create `datasets/doric_conversations_sharegpt.jsonl` for training pipelines.
4. **Fine-tune in `finetune_notebook/`** – open the notebook (Colab / local Jupyter), point it at the ShareGPT dataset (in this instance it's pulled from Hugging Face), and run the full Unsloth/LoRA workflow (data loading, packing, training, merge/export).
5. **Validate** – sanity-check held-out prompts plus any safety/adversarial probes.

## Scripts & Usage

### 1. Topic Expansion – `dataset_generation/generate_topics_via_llm.py`

Creates additional grouped prompts while filtering duplicates and banned patterns.

```bash
uv run python dataset_generation/generate_topics_via_llm.py \
  --input topics.json \
  --output topics_augmented.json \
  --per-group 40
```

Key flags:
- `--in-place` to overwrite `topics.json`
- `--model`, `OPENAI_API_KEY` control the LLM backend

### 2. Dataset Generation – `dataset_generation/generate_doric_dataset.py`

Builds two-turn chats (`user` → `assistant`) with bucket controls for core/adv/safety/multi prompts and rigorous quality filters (marker density, overlap, dedup, sentence-boundary checks).

```bash
uv run python dataset_generation/generate_doric_dataset.py \
  --topics-json topics.json \
  --out datasets/doric_synth.jsonl \
  --n-per-topic 6 --adv-ratio 0.1 --safety-ratio 0.05 --multi-ratio 0.2 \
  --max-concurrency 50
```

Notes:
- Without `OPENAI_API_KEY`, it falls back to the heuristic template backend.
- Outputs land in `datasets/` (create it once with `mkdir -p datasets`).
- Logs show rejected samples so you can tune topics or ratios.

### 3. ShareGPT Transformation – `dataset_generation/transform_to_sharegpt.py`

Converts the generator’s `{messages, meta}` rows into the ShareGPT shape expected by many fine-tuning pipelines.

```bash
uv run python dataset_generation/transform_to_sharegpt.py \
  --input datasets/doric_synth.jsonl \
  --output datasets/doric_conversations_sharegpt.jsonl
```

Add `--drop-meta` if you only want the `conversations` array.

## Finetune Notebook

`finetune_notebook/` is designed for hosted runtimes like Google Colab (Pro+ recommended) or any GPU notebook service with enough VRAM for a 7B/8x7B base model. Highlights:

- **Framework**: built around Unsloth’s fast LoRA adapters for OSS checkpoints (Mixtral, Llama, etc.). The notebook wires up Unsloth’s training loop, PEFT config, and weight merging so you can export either adapters or a fully merged model.
- **Data ingestion**: expects the ShareGPT JSONL under `datasets/`. A helper cell uploads it to Hugging Face Datasets (private repo by default) or loads it directly from local storage if running outside Colab. That makes it trivial to reuse the same data in future notebook sessions.
- **Training flow**: load base model/tokenizer, push dataset through packing/shuffling, configure LoRA ranks + learning rate schedule, run evaluation prompts inline, and optionally upload the resulting adapter/merged weights back to Hugging Face Hub.
- **Runtime tips**: enable Colab’s T4/A100 runtime, mount Drive for persistence, and watch the notebook’s VRAM budget sliders. The cells are annotated so you can swap in different base models or play with Unsloth hyperparameters.

A quick start is documented at the top of the notebook itself: open `finetune_notebook/doric_finetune.ipynb` (or similar), run the setup cell to install Unsloth + HF deps via `uv`, then follow the numbered sections (data upload, training, export).

## Modal Deployment

Use the `modal/` directory when you want to host the fine-tuned Doric model behind a lightweight inference API. Modal provisions GPUs on demand, runs `vllm_doric.py` (a vLLM app that loads the Hugging Face checkpoint), and exposes an endpoint you can hit from scripts or the bundled CLI.

Typical flow:

```bash
uv add modal                  # install Modal SDK (one-time)
cd modal
modal deploy vllm_doric.py    # builds the container + deploys the endpoint
uv run python chat_cli.py     # interactive REPL that streams replies via Modal
```

Before deploying, run `modal token new` so the CLI can authenticate. The CLI sends prompts to the deployed vLLM server and prints streamed Doric responses, making it handy for smoke tests after each fine-tune. More context lives in `modal/README.md` if you need environment variables or customization tips.

## Additional Notes

- Use `uv add <package>` then `uv sync` if you need extra libs for experiments.
- The generator enforces sentence-safe outputs and adaptive token budgets to avoid truncated Doric responses.
- Adjust `--seed` for reproducible shuffles, and `--progress-every` or `--no-progress` to tune logging.
