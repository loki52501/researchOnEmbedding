# Design: Low-Rank Factorized Embeddings
**Date:** 2026-03-10
**Status:** Approved
**Goal:** Reduce embedding memory and compute cost within the autoresearch loop on RTX A4000 (16 GB VRAM)

---

## Problem

With DEPTH=6, model_dim=384, vocab=8192, embeddings account for ~31% of all parameters:

| Component | Shape | Params |
|-----------|-------|--------|
| `wte` | 8192 × 384 | 3.1M |
| `value_embeds` × 3 | 8192 × 384 each | 9.4M |
| `lm_head` | 384 × 8192 | 3.1M |
| **Total** | | **~15.7M / ~50M** |

`lm_head` is also the most expensive forward-pass operation — a large matmul over vocab at every token position.

---

## Solution: Low-Rank Factorization

Decompose every `V×D` embedding matrix into `V×r` + `r×D` where `r << D`.

### New Modules

```python
class FactorizedEmbedding(nn.Module):
    """Drop-in for nn.Embedding(V, D). Stores V×r + projects r→D."""
    def __init__(self, vocab_size, embed_dim, rank):
        self.embed = nn.Embedding(vocab_size, rank)
        self.proj  = nn.Linear(rank, embed_dim, bias=False)
    def forward(self, idx):
        return self.proj(self.embed(idx))

class FactorizedLMHead(nn.Module):
    """Drop-in for nn.Linear(D, V). Projects D→r then r→V."""
    def __init__(self, embed_dim, vocab_size, rank):
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.up   = nn.Linear(rank, vocab_size, bias=False)
    def forward(self, x):
        return self.up(self.down(x))
```

### Memory Impact at r=64

| Component | Before | After (r=64) | Reduction |
|-----------|--------|--------------|-----------|
| `wte` | 3.1M | 548K | 5.6× |
| `value_embeds` × 3 | 9.4M | 1.6M | 5.9× |
| `lm_head` | 3.1M | 548K | 5.6× |
| **Total** | **15.7M** | **~2.7M** | **5.8×** |

### Changes to `train.py`

1. Add `EMBED_RANK = 64` hyperparameter (agent sweeps: 32, 64, 128)
2. Add `embed_rank: int = 64` to `GPTConfig`
3. Add `FactorizedEmbedding` and `FactorizedLMHead` classes
4. In `GPT.__init__`: swap `nn.Embedding` → `FactorizedEmbedding`, swap `nn.Linear` (lm_head) → `FactorizedLMHead`
5. In `init_weights`: init embed table `normal_(std=1.0)`, proj `uniform_(-s, s)`, lm_head down zeros, up small normal
6. Optimizer: keep embed/proj params in AdamW group (not Muon — these are small matrices)

---

## Experiment Sequence

| Exp | Hypothesis | Expected |
|-----|-----------|---------|
| Baseline | Unmodified train.py | Reference val_bpb |
| r=64 | 5.8× embedding compression | val_bpb within 0.01 of baseline |
| r=128 | More capacity, 2.4× savings | Better val_bpb than r=64, less savings |
| r=32 | Most aggressive | val_bpb degrades, find floor |
| Best r + weight tying | Share wte ↔ lm_head factorized table | Further memory reduction |

---

## Success Criteria

- `val_bpb` within **0.01** of baseline at `r=64`
- `peak_vram_mb` measurably lower
- Code change < 40 lines net added to `train.py`
- Simplicity criterion: if r=128 matches r=64 quality with negligible memory difference, prefer r=64

---

## Future Work (Phase 2)

- Weight tying: share `wte` and `lm_head` factorized matrices
- Shared value embeddings: one factorized VE table across all VE layers
- Combine with int8 quantization of the small lookup table
