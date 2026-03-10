# autoresearch Project Tracker

## Overview
Autonomous LLM pretraining research loop on RTX A4000 (16 GB VRAM).
Agent modifies `train.py`, runs 5-min experiments, logs to `results.tsv`, keeps improvements.

---

## Active Research: Factorized Embeddings

**Goal:** Reduce embedding memory + compute by factorizing V×D tables into V×r + r×D

**Design doc:** `docs/plans/2026-03-10-factorized-embeddings-design.md`

**Status:** Code complete — ready for cloud experiments

### Tasks

- [x] Implement `FactorizedEmbedding` and `FactorizedLMHead` in `train.py`
- [x] Add `EMBED_RANK = 64` hyperparameter to `train.py`
- [x] Update `GPTConfig` with `embed_rank` field
- [x] Update `GPT.__init__` to use factorized modules
- [x] Update `init_weights` for factorized modules (+ fixed `estimate_flops` and `_precompute_rotary_embeddings` crash bugs)
- [x] Verify optimizer groups handle new params correctly
- [ ] Run baseline experiment — set `EMBED_RANK = 384` (no compression), record val_bpb
- [ ] Run r=64 experiment — primary target (~5.8× embedding savings)
- [ ] Run r=128 experiment — more capacity, ~2.4× savings
- [ ] Run r=32 experiment — most aggressive, find quality floor
- [ ] Run best_r + weight tying experiment (Phase 2)

### Results Log

| Experiment | val_bpb | peak_vram_gb | status | notes |
|------------|---------|--------------|--------|-------|
| baseline | TBD | TBD | pending | unmodified train.py |
| r=64 | TBD | TBD | pending | |
| r=128 | TBD | TBD | pending | |
| r=32 | TBD | TBD | pending | |
| r=best + tying | TBD | TBD | pending | |

---

## Environment

| Item | Value |
|------|-------|
| GPU | RTX A4000 |
| VRAM | 15.7 GB |
| CUDA | 12.1 |
| PyTorch | 2.1.1+cu121 |
| Python | 3.11 |
| Compute cap | sm_86 (Ampere) |
| FA3 | Patched → PyTorch SDPA |
| DEPTH | 6 |
| model_dim | 384 |
| vocab_size | 8192 |
| MAX_SEQ_LEN | 1024 |
| DEVICE_BATCH_SIZE | 16 |
| TOTAL_BATCH_SIZE | 2^17 |

---

## Future Research Ideas

- **Phase 2:** Weight tying — share `wte` ↔ `lm_head` factorized matrix
- **Phase 2:** Shared value embeddings — one factorized VE table across all VE layers
- **Phase 3:** int8 quantization of small lookup table (combine with factorization)
- **Phase 3:** Compositional embeddings — represent tokens as combinations of a small codebook

---

## Files

| File | Purpose |
|------|---------|
| `train.py` | Model + training loop (agent modifies this) |
| `prepare.py` | Data prep + eval harness (do not modify) |
| `program.md` | Agent instructions |
| `results.tsv` | Experiment log (untracked by git) |
| `run.log` | Latest training run output |
| `autoresearch.ipynb` | Cloud notebook setup (A4000 + pip) |
| `docs/plans/` | Design documents |
| `project.md` | This file — project tracker |
