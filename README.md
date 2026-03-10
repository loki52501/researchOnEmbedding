# autoresearch — RTX A4000 Fork

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for **RTX A4000 (16 GB VRAM)** cloud kernels, with an active research thread on **resource-efficient embedding methods**.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

## What's different in this fork

### 1. RTX A4000 support (Ampere, sm_86)

Flash Attention 3 does not run on Ampere GPUs. This fork patches `train.py` to use **PyTorch SDPA** as a drop-in replacement, with correct causal masking and sliding-window support.

A4000-tuned defaults (set via `autoresearch.ipynb`):

| Parameter | Original (H100) | This fork (A4000) |
|-----------|----------------|-------------------|
| `MAX_SEQ_LEN` | 2048 | 1024 |
| `EVAL_TOKENS` | 40 × 524288 | 10 × 524288 |
| `DEPTH` | 8 | 6 |
| `DEVICE_BATCH_SIZE` | 128 | 16 |
| `TOTAL_BATCH_SIZE` | 2^19 | 2^17 |
| `WINDOW_PATTERN` | SSSL | L |

### 2. Factorized Embeddings research thread

Active research on replacing standard `nn.Embedding(V, D)` with a low-rank factorized version `Embedding(V, r) → Linear(r, D)`, and similarly for the LM head.

**Motivation:** With DEPTH=6 and model_dim=384, embedding tables account for ~31% of all parameters and the LM head is the most expensive forward-pass operation. Factorization reduces both.

**New hyperparameter:** `EMBED_RANK` (default 64) — controls the bottleneck rank. Sweep: 32, 64, 128.

**Estimated savings at r=64:**

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| `wte` | 3.1M | 548K | 5.6× |
| `value_embeds` ×3 | 9.4M | 1.6M | 5.9× |
| `lm_head` | 3.1M | 548K | 5.6× |
| **Total** | **15.7M** | **~2.7M** | **5.8×** |

Design doc: [`docs/plans/2026-03-10-factorized-embeddings-design.md`](docs/plans/2026-03-10-factorized-embeddings-design.md)

### 3. Cloud notebook setup

`autoresearch.ipynb` — a Jupyter notebook that handles the full setup on a cloud GPU kernel (no `uv` required, pip-based):

- GPU check
- pip install of all dependencies + embedding research libraries (`einops`, `bitsandbytes`, `vector-quantize-pytorch`, `triton`, `scipy`, `scikit-learn`)
- FA3 → SDPA patch
- A4000 hyperparameter tuning
- Data preparation
- Baseline training run
- Manual experiment runner helper

---

## How it works

The repo has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

Training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation). The metric is **val_bpb** (validation bits per byte) — lower is better.

## Quick start (cloud kernel, pip-based)

```bash
# 1. Install dependencies (no uv needed)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install kernels matplotlib numpy pandas pyarrow requests rustbpe tiktoken
pip install einops bitsandbytes vector-quantize-pytorch triton scipy scikit-learn

# 2. Download data and train tokenizer (one-time, ~2 min)
python prepare.py --num-shards 10

# 3. Run a single training experiment (~5 min)
python train.py
```

Or open **`autoresearch.ipynb`** and run cells top to bottom — it handles everything including the FA3 patch and A4000 tuning.

## Quick start (uv, H100 / original setup)

```bash
# 1. Install uv project manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

## Running the agent

Spin up Claude Code in this repo, then:

```
Have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent will create a branch, establish a baseline, and loop autonomously.

## Project structure

```
prepare.py           — constants, data prep + runtime utilities (do not modify)
train.py             — model, optimizer, training loop (agent modifies this)
program.md           — agent instructions
pyproject.toml       — dependencies
autoresearch.ipynb   — cloud notebook setup (A4000 + pip)
project.md           — research tracker (active experiments, results log)
docs/plans/          — design documents for research threads
results.tsv          — experiment log (untracked by git)
```

## A4000 tuning notes

If you're running on an RTX A4000 or similar 16 GB card:

1. Run `autoresearch.ipynb` cells 3a–3c to apply patches automatically, or apply manually:
   - `prepare.py`: set `MAX_SEQ_LEN = 1024`, `EVAL_TOKENS = 10 * 524288`
   - `train.py`: set `DEPTH = 6`, `DEVICE_BATCH_SIZE = 16`, `TOTAL_BATCH_SIZE = 2**17`, `WINDOW_PATTERN = "L"`
2. FA3 is not supported on Ampere — run cell 3b to patch with PyTorch SDPA
3. Use `python train.py` instead of `uv run train.py` if uv is unavailable

## Notable forks

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (original, H100)
- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS MLX)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows RTX)

## License

MIT
