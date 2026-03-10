# Factorized Embeddings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all `nn.Embedding` and the `lm_head` linear in `train.py` with low-rank factorized versions, reducing embedding parameter count ~5.8× while keeping val_bpb within 0.01 of baseline.

**Architecture:** Decompose every `V×D` embedding table into `V×r` (small lookup) + `r→D` (linear projection), and the unembedding head `D→V` into `D→r` + `r→V`. A single new hyperparameter `EMBED_RANK` controls the bottleneck rank. All three embedding sites (wte, value_embeds, lm_head) are factorized identically.

**Tech Stack:** PyTorch `nn.Module`, existing `train.py` structure. No new dependencies. New classes are drop-in replacements.

**Reference files:**
- Modify: `train.py` (the only file we touch)
- Design doc: `docs/plans/2026-03-10-factorized-embeddings-design.md`
- Tracker: `project.md`

---

### Task 1: Add `EMBED_RANK` hyperparameter

**Files:**
- Modify: `train.py` (hyperparameters section, around line 449)

**Step 1: Open `train.py` and locate the hyperparameters block**

Look for this comment:
```python
# Model size
DEPTH = 6               # number of transformer layers
DEVICE_BATCH_SIZE = 16  # per-device batch size (reduce if OOM)
```

**Step 2: Add `EMBED_RANK` directly below `DEPTH`**

```python
# Model size
DEPTH = 6               # number of transformer layers
EMBED_RANK = 64         # bottleneck rank for factorized embeddings (try 32, 64, 128)
DEVICE_BATCH_SIZE = 16  # per-device batch size (reduce if OOM)
```

**Step 3: Verify**

```bash
grep -n "EMBED_RANK" train.py
```
Expected: one line showing `EMBED_RANK = 64`

---

### Task 2: Add `embed_rank` field to `GPTConfig`

**Files:**
- Modify: `train.py` (`GPTConfig` dataclass, around line 32)

**Step 1: Locate `GPTConfig`**

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
```

**Step 2: Add `embed_rank` field**

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    embed_rank: int = 64          # bottleneck rank for factorized embeddings
```

**Step 3: Verify**

```bash
grep -n "embed_rank" train.py
```
Expected: two lines — one in `GPTConfig`, we'll add more in later tasks.

---

### Task 3: Add `FactorizedEmbedding` and `FactorizedLMHead` classes

**Files:**
- Modify: `train.py` (insert after the `norm` helper, around line 43)

**Step 1: Locate insertion point — after the `norm` function**

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

**Step 2: Insert both classes immediately after `norm`**

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class FactorizedEmbedding(nn.Module):
    """
    Drop-in replacement for nn.Embedding(vocab_size, embed_dim).
    Stores a small V×r table and projects r→embed_dim.
    Saves memory: V*r + r*D instead of V*D params.
    """
    def __init__(self, vocab_size: int, embed_dim: int, rank: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, rank)
        self.proj  = nn.Linear(rank, embed_dim, bias=False)

    def forward(self, idx):
        return self.proj(self.embed(idx))


class FactorizedLMHead(nn.Module):
    """
    Drop-in replacement for nn.Linear(embed_dim, vocab_size) used as lm_head.
    Projects D→r→V instead of D→V directly.
    Saves memory and reduces the cost of the final large matmul.
    """
    def __init__(self, embed_dim: int, vocab_size: int, rank: int):
        super().__init__()
        self.down = nn.Linear(embed_dim, rank,       bias=False)
        self.up   = nn.Linear(rank,      vocab_size, bias=False)

    def forward(self, x):
        return self.up(self.down(x))
```

**Step 3: Verify**

```bash
grep -n "class Factorized" train.py
```
Expected: two lines — `FactorizedEmbedding` and `FactorizedLMHead`.

---

### Task 4: Update `GPT.__init__` to use factorized modules

**Files:**
- Modify: `train.py` (`GPT.__init__`, around lines 128–141)

**Step 1: Locate the three embedding instantiations in `GPT.__init__`**

```python
self.transformer = nn.ModuleDict({
    "wte": nn.Embedding(config.vocab_size, config.n_embd),
    "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
})
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
...
self.value_embeds = nn.ModuleDict({
    str(i): nn.Embedding(config.vocab_size, kv_dim)
    for i in range(config.n_layer) if has_ve(i, config.n_layer)
})
```

**Step 2: Replace all three with factorized versions**

```python
self.transformer = nn.ModuleDict({
    "wte": FactorizedEmbedding(config.vocab_size, config.n_embd, config.embed_rank),
    "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
})
self.lm_head = FactorizedLMHead(config.n_embd, config.vocab_size, config.embed_rank)
...
self.value_embeds = nn.ModuleDict({
    str(i): FactorizedEmbedding(config.vocab_size, kv_dim, config.embed_rank)
    for i in range(config.n_layer) if has_ve(i, config.n_layer)
})
```

**Step 3: Verify no bare `nn.Embedding` or vocab-sized `nn.Linear` remain**

```bash
grep -n "nn\.Embedding\|nn\.Linear.*vocab" train.py
```
Expected: zero matches (all replaced).

---

### Task 5: Update `build_model_config` to pass `EMBED_RANK`

**Files:**
- Modify: `train.py` (`build_model_config` function, around line 468)

**Step 1: Locate `build_model_config`**

```python
def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )
```

**Step 2: Pass `embed_rank`**

```python
def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
        embed_rank=EMBED_RANK,
    )
```

**Step 3: Verify**

```bash
grep -n "embed_rank" train.py
```
Expected: lines in `GPTConfig`, `build_model_config`, and the two new classes.

---

### Task 6: Update `init_weights` for factorized modules

**Files:**
- Modify: `train.py` (`GPT.init_weights`, around lines 150–179)

**Step 1: Locate the embedding init section**

```python
# Embedding and unembedding
torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
...
# Value embeddings
for ve in self.value_embeds.values():
    torch.nn.init.uniform_(ve.weight, -s, s)
```

**Step 2: Replace with factorized inits**

```python
# Embedding and unembedding (factorized)
torch.nn.init.normal_(self.transformer.wte.embed.weight, mean=0.0, std=1.0)
torch.nn.init.uniform_(self.transformer.wte.proj.weight, -s, s)
torch.nn.init.zeros_(self.lm_head.down.weight)
torch.nn.init.normal_(self.lm_head.up.weight, mean=0.0, std=0.001)
...
# Value embeddings (factorized)
for ve in self.value_embeds.values():
    torch.nn.init.normal_(ve.embed.weight, mean=0.0, std=1.0)
    torch.nn.init.uniform_(ve.proj.weight, -s, s)
```

**Step 3: Update the bf16 cast section (end of init_weights)**

```python
# Cast embeddings to bf16
self.transformer.wte.embed.to(dtype=torch.bfloat16)
self.transformer.wte.proj.to(dtype=torch.bfloat16)
for ve in self.value_embeds.values():
    ve.embed.to(dtype=torch.bfloat16)
    ve.proj.to(dtype=torch.bfloat16)
```

---

### Task 7: Update `num_scaling_params` to count factorized params correctly

**Files:**
- Modify: `train.py` (`GPT.num_scaling_params`, around line 223)

**Step 1: Locate the method**

```python
def num_scaling_params(self):
    wte = sum(p.numel() for p in self.transformer.wte.parameters())
    value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
    lm_head = sum(p.numel() for p in self.lm_head.parameters())
    ...
```

This already uses `.parameters()` recursively — no change needed. The method correctly counts all params including the new sub-modules.

**Step 1: Verify by checking it still sums correctly**

```bash
grep -n "num_scaling_params\|wte\|value_embeds\|lm_head" train.py | grep -v "^.*#"
```

No code change needed here — `parameters()` traverses sub-modules automatically.

---

### Task 8: Update `setup_optimizer` to handle factorized embedding params

**Files:**
- Modify: `train.py` (`GPT.setup_optimizer`, around line 235)

**Step 1: Locate the param group definitions**

```python
value_embeds_params = list(self.value_embeds.parameters())
embedding_params = list(self.transformer.wte.parameters())
lm_head_params = list(self.lm_head.parameters())
```

`self.value_embeds.parameters()` and `self.transformer.wte.parameters()` already recursively include `.embed.weight` and `.proj.weight` from the factorized modules.

`self.lm_head.parameters()` includes `.down.weight` and `.up.weight`.

**Step 1: Verify the assert still holds**

The assert counts all params:
```python
assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
    len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
```

`matrix_params = list(self.transformer.h.parameters())` — these are the Block params only, unchanged.

The new proj weights inside `FactorizedEmbedding` and `FactorizedLMHead` are 2D matrices. They will end up in `embedding_params` / `lm_head_params` (via `.parameters()` recursion), NOT in `matrix_params`. This is correct — keep them in AdamW, not Muon (they're small auxiliary projections).

**No code change needed** — the existing `.parameters()` calls capture the new sub-modules correctly.

**Step 2: Confirm by running a quick forward pass test (next task)**

---

### Task 9: Smoke test — verify model builds and runs without crash

**Files:**
- No file changes — run existing code

**Step 1: Create a quick smoke test script**

```python
# smoke_test.py  (run once, then delete)
import sys
sys.path.insert(0, '.')
import torch
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Minimal imports from train.py
exec(open('train.py').read().split('t_start = time.time()')[0])

config = build_model_config(DEPTH)
print(f"Config: {config}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device="cuda")
model.init_weights()

counts = model.num_scaling_params()
print("Param counts:")
for k, v in counts.items():
    print(f"  {k:24s}: {v:,}")

# Quick forward pass
x = torch.randint(0, config.vocab_size, (2, 64), device="cuda")
y = torch.randint(0, config.vocab_size, (2, 64), device="cuda")
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model(x, y)
print(f"Loss: {loss.item():.4f}  (should be ~log(vocab_size) ≈ {torch.log(torch.tensor(config.vocab_size)).item():.2f})")
print("Smoke test PASSED")
```

**Step 2: Run it**

```bash
python smoke_test.py
```

Expected output:
- Param counts printed — `wte` + `value_embeds` + `lm_head` should be dramatically smaller than before
- Loss should be near `log(8192) ≈ 9.01` (random init, untrained)
- No crash, no NaN

**Step 3: Delete smoke test**

```bash
rm smoke_test.py
```

---

### Task 10: Run baseline experiment

**Step 1: Ensure `results.tsv` exists**

```bash
[ -f results.tsv ] || echo -e "commit\tval_bpb\tmemory_gb\tstatus\tdescription" > results.tsv
cat results.tsv
```

**Step 2: Stash current factorized changes, run original train.py as baseline**

Actually — the factorized version IS the new train.py. Run it at `EMBED_RANK=64` as the first experiment. The pre-factorization baseline was already established (or run prepare.py baseline from the notebook).

If you need a clean baseline, temporarily set `EMBED_RANK` so high it's effectively unfactorized:

```python
EMBED_RANK = 384   # same as model_dim → no compression, equivalent to nn.Embedding
```

Run:
```bash
python train.py > run.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

Record in `results.tsv`:
```
<hash>    <val_bpb>    <peak_mb/1024>    keep    baseline (EMBED_RANK=384, no compression)
```

---

### Task 11: Run `EMBED_RANK=64` experiment

**Step 1: Set rank in `train.py`**

```python
EMBED_RANK = 64
```

**Step 2: Run**

```bash
python train.py > run.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

**Step 3: Evaluate**

- If `val_bpb` is within 0.01 of baseline AND `peak_vram_mb` is lower → **keep**
- If `val_bpb` is more than 0.01 worse → **discard**, note degradation

**Step 4: Record in `results.tsv` and update `project.md`**

---

### Task 12: Run `EMBED_RANK=128` experiment

**Step 1: Set rank**

```python
EMBED_RANK = 128
```

**Step 2: Run + evaluate** (same as Task 11)

Expected: slightly better val_bpb than r=64, less memory savings (2.4× instead of 5.8×).

---

### Task 13: Run `EMBED_RANK=32` experiment

**Step 1: Set rank**

```python
EMBED_RANK = 32
```

**Step 2: Run + evaluate**

Expected: most memory savings but likely val_bpb degrades. Find the quality floor.

---

### Task 14: Update `project.md` with findings

**Step 1: Fill in the Results Log table in `project.md`**

```markdown
| Experiment | val_bpb | peak_vram_gb | status | notes |
|------------|---------|--------------|--------|-------|
| baseline (r=384) | X.XXXXXX | XX.X | keep | reference |
| r=128 | X.XXXXXX | XX.X | keep/discard | ... |
| r=64  | X.XXXXXX | XX.X | keep/discard | sweet spot? |
| r=32  | X.XXXXXX | XX.X | keep/discard | quality floor |
```

**Step 2: Mark completed tasks with ✅ in `project.md`**

**Step 3: If r=64 or r=128 wins, proceed to Phase 2 (weight tying)**

Add to Future Research section:
```markdown
- **Next:** Weight tying — share wte ↔ lm_head factorized matrices at best rank found
```

---

## Quick Reference: What changed in `train.py`

| Location | Before | After |
|----------|--------|-------|
| Hyperparams | — | `EMBED_RANK = 64` |
| `GPTConfig` | — | `embed_rank: int = 64` |
| After `norm()` | — | `FactorizedEmbedding` + `FactorizedLMHead` classes |
| `GPT.__init__` wte | `nn.Embedding(V, D)` | `FactorizedEmbedding(V, D, r)` |
| `GPT.__init__` lm_head | `nn.Linear(D, V)` | `FactorizedLMHead(D, V, r)` |
| `GPT.__init__` value_embeds | `nn.Embedding(V, kv)` | `FactorizedEmbedding(V, kv, r)` |
| `build_model_config` | — | passes `embed_rank=EMBED_RANK` |
| `init_weights` wte | `.weight` | `.embed.weight` + `.proj.weight` |
| `init_weights` value_embeds | `.weight` | `.embed.weight` + `.proj.weight` |
| `init_weights` lm_head | `.weight` | `.down.weight` + `.up.weight` |
| `init_weights` bf16 cast | `.wte.to(bf16)` | `.wte.embed.to(bf16)` + `.wte.proj.to(bf16)` |
| `build_model_config` | no embed_rank | `embed_rank=EMBED_RANK` |
