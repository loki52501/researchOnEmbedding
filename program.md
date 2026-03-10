# autoresearch — Embedding Optimization for Resource-Efficient Chat Models

This is an autonomous research experiment focused on a single question:

> **What is the most resource-efficient embedding design that lets a small LLM reach good language quality — good enough for ChatGPT-like conversational use on minimal hardware?**

The constraint is real: RTX A4000, 16 GB VRAM, 5-minute training budget per experiment. Every byte of VRAM saved in the embedding layer is a byte available for more transformer depth or wider attention. The goal is not just lower val_bpb — it's lower val_bpb *per MB of embedding memory used*.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `python prepare.py` (no uv on this cloud kernel).
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU (RTX A4000, 16 GB VRAM). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). Launch it as: `python train.py` (no uv on this cloud kernel).

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, embedding design, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages beyond what is already installed (`einops`, `bitsandbytes`, `vector-quantize-pytorch`, `triton`, `scipy`, `scikit-learn` are all available).
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The primary goal: get the lowest val_bpb while minimising embedding memory.** The secondary goal is for the resulting model to be practical for chat use on constrained hardware. These two goals align — a smaller, better-trained model is more useful than a large undertrained one.

**VRAM** is a hard constraint here (16 GB total). Keep `peak_vram_mb` well under 15000. If an experiment uses dramatically more VRAM without a proportional val_bpb gain, discard it.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Research focus: Embedding optimisation

The current `train.py` already has **factorized embeddings** (`EMBED_RANK=64`): every `V×D` embedding table is decomposed into `V×r + r→D`. This is the starting point.

Work through these ideas in order. For each, ask: does it lower val_bpb? Does it reduce `peak_vram_mb`? Is the code still clean?

### Tier 1 — Sweep the rank (quick wins, run first)
- `EMBED_RANK = 128` — more capacity, still 2.4× memory savings vs unfactorized
- `EMBED_RANK = 32` — most aggressive compression, find the quality floor
- `EMBED_RANK = 96` — try between if 64 and 128 are close

### Tier 2 — Weight tying
- **Tie `wte` ↔ `lm_head`**: share the same factorized embedding matrix for input and output. Standard trick in language models — saves the full lm_head table with no quality loss. Implement by having `lm_head.up.weight = wte.embed.weight` (transposed) after init.
- **Shared value embeddings**: instead of a separate `FactorizedEmbedding` per VE layer, use one shared table projected differently per layer.

### Tier 3 — Smarter embedding initialisation
- Try **orthogonal init** for the embedding lookup table (`nn.init.orthogonal_`) — more spread in embedding space from step 0.
- Try **scaled normal init** (`std = 1/sqrt(rank)`) for the projection — keeps embedding norms stable.

### Tier 4 — Use EMBED_RANK to free up budget for depth
- If rank=64 saves ~13M params, try reinvesting those params into depth: increase `DEPTH` from 6 to 7 or 8 while keeping factorized embeddings. Does more depth + factorized embeddings beat baseline depth + full embeddings?
- Alternatively, increase `n_head` or widen the model slightly.

### Tier 5 — Activation on the projection
- Add a non-linearity between embed and proj in `FactorizedEmbedding`: `self.proj(F.silu(self.embed(idx)))`. Does a gated projection help or hurt?
- Try `F.relu` or `F.gelu` — simpler, potentially faster.

### Tier 6 — Quantised lookup table
- Store the small `V×r` lookup table in int8 (bitsandbytes `bnb.nn.Embedding8bit` or manual quantize+dequantize). Halves the lookup table memory. The projection stays in bf16.

### Tier 7 — Codebook embeddings (most experimental)
- Use `vector_quantize_pytorch` to learn a codebook of ~256–512 codes. Each of the 8192 tokens is represented as a combination of codes. This is the most radical compression but hardest to train.

### Ideas to avoid
- Do not change the tokenizer or vocabulary size — `prepare.py` is fixed.
- Do not add attention mechanisms to the embedding layer — too much complexity for marginal gain.
- Do not try MoE-style embeddings unless all Tier 1–4 ideas are exhausted.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
