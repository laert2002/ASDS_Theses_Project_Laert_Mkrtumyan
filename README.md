# A Study of Memory–Computation Trade-offs in Long-Context LLM Inference

Master's thesis project. This first stage covers the foundations: the environment, tokenization, model architecture, tensor shapes, attention, and the KV cache. It uses **Llama-3.2-1B** and **Llama-3.1-8B**.

## Repository layout

```
configs/    default.yaml   - all settings (model ids, dtype, prompt, ...)
src/        experiment scripts (one per task) + utils.py (shared helpers)
results/    JSON outputs written by the scripts (machine-readable)
notebooks/  exploratory Jupyter notebooks
docs/       write-ups / explanations for each task
```

| Task | Script | Write-up |
|---|---|---|
| 1. Environment | [src/env_check.py](src/env_check.py) | this README |
| 3. Tokenization | [src/tokenization_demo.py](src/tokenization_demo.py) | [docs/tokenization.md](docs/tokenization.md) |
| 4. Model configs | [src/model_configs.py](src/model_configs.py) | [docs/model_configs.md](docs/model_configs.md) |
| 5. Forward-pass shapes | [src/forward_shapes.py](src/forward_shapes.py) | [docs/shapes.md](docs/shapes.md) |
| 6. Attention by hand | [src/attention_manual.py](src/attention_manual.py) | docstring in the script |
| 7. KV cache | [src/kv_cache_inspect.py](src/kv_cache_inspect.py) | [docs/kv_cache_structure.md](docs/kv_cache_structure.md) |

## Setup

### 1. Python environment

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
```

### 2. PyTorch with CUDA

Install PyTorch **first**, with the CUDA version your driver supports. Get the exact command from https://pytorch.org/get-started/locally/. Example for CUDA 12.4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then install the rest of the dependencies:

```bash
pip install -r requirements.txt
```

Check that the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. Hugging Face access (the Llama models are gated)

1. Log in at huggingface.co and open the model pages, [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B). Accept the license on each page and wait for approval.
2. Create a **read** access token under *Settings → Access Tokens*.
3. Log in on the machine: `hf auth login` (older versions: `huggingface-cli login`). Alternatively, set the `HF_TOKEN` environment variable. **Never commit the token to the repository.**

Models are downloaded once into the Hugging Face cache (`~/.cache/huggingface`, or `HF_HOME` if set). Llama-3.2-1B is ≈ 2.5 GB in fp16 and Llama-3.1-8B is ≈ 16 GB.

### GPU requirements

| Script | Needs GPU? | Downloads |
|---|---|---|
| `attention_manual.py` | no | nothing |
| `tokenization_demo.py` | no | tokenizer only |
| `model_configs.py` | no | `config.json` only (both models) |
| `env_check.py`, `forward_shapes.py`, `kv_cache_inspect.py` | yes (≥ 6 GB is enough for 1B) | Llama-3.2-1B weights |

Without CUDA, the scripts fall back to CPU + float32. This is slow, but useful for debugging.

## Running

Always run from the **repository root**:

```bash
python src/env_check.py          # Task 1: load 1B in fp16, generate 20 tokens
python src/tokenization_demo.py  # Task 3
python src/model_configs.py      # Task 4
python src/forward_shapes.py     # Task 5
python src/attention_manual.py   # Task 6
python src/kv_cache_inspect.py   # Task 7
```

Each script prints its results and saves a JSON file to `results/`. Several scripts also print a markdown table that you can paste into the matching `docs/` file (look for the `TODO` markers).

## Configuration

All settings are in [configs/default.yaml](configs/default.yaml): model ids, `dtype`, `device`, `seed`, prompt, and the input lengths for the attention-growth experiment. Change them there rather than in the code.

## Extending

- Shared logic (loading, devices, saving) lives in [src/utils.py](src/utils.py). Change model loading there, for example to add quantization or a different attention backend, and every script picks it up.
- Add new experiments as new scripts in `src/` that import from `utils`, and write their outputs to `results/`.
