# Reproduction notes — *LLMs Know More Than They Show* (Orgad et al., ICLR 2025)

Fork of [technion-cs-nlp/LLMsKnow](https://github.com/technion-cs-nlp/LLMsKnow). Author code in `src/` is left untouched except for a single-line `import wandb` → `from wandb_shim import wandb` swap; our additions live in `scripts/`, `src/wandb_shim.py`, and (later) `phase1/`, `phase2/`. This file documents how to recreate the environment and outputs for Phase 0.

## Environment

- Python 3.11.6 pinned via `.python-version`.
- Managed with [`uv`](https://github.com/astral-sh/uv). `pyproject.toml` lists all deps; lock in `uv.lock`.
- Install:

  ```bash
  uv sync
  ```

- `wandb` is replaced by `trackio` via a thin shim at `src/wandb_shim.py` (preserves the `wandb.summary[k] = v`, `wandb.Artifact`, `wandb.run.name`, `wandb.Image`, `wandb.log_artifact` surface that the author scripts use).
- Hardware used here: one RTX 3090 (24 GB). Mistral-7B-Instruct-v0.2 loads in bf16 via `device_map='auto'`.

### Batching deviation from author code

Author scripts run batch=1 throughout. On a single 3090 this underutilises the GPU heavily (we measured ~5× speedup at batch=8 on TriviaQA generation, comparable wins on hidden-state extraction). We added `--batch_size` to `generate_model_answers.py`, `extract_exact_answer.py`, and `probe.py`. The unbatched code paths are untouched and remain the default.

**Numerical caveat:** bf16 batched generation is not bit-exact vs batch=1. On a 20-sample TriviaQA smoke test:
- generated answer text matches unbatched on ~30% of samples; the rest drift to different but valid continuations (`compute_correctness` labels match 20/20).
- hidden states at MLP-out layer 15 have a per-element max-abs error ≈ 0.003-0.007 (mean magnitude ≈ 0.022) — left-padding changes the RoPE/attention reduction order.
- Critically, this noise washes out at the probe: a logistic regression trained on either feature set produces predicted probabilities with Pearson correlation 0.99999 and max absolute difference < 0.002.

We therefore treat the batched path as the production setup and accept the small drift in raw generations. The probe AUROC reported below was obtained at `batch_size=8`.

## Datasets

- TriviaQA unfiltered: extracted from `triviaqa-unfiltered.tar.gz` into `data/triviaqa-unfiltered/{unfiltered-web-train.json, unfiltered-web-dev.json}`.
- **Caveat (in author code, not our change):** `load_data_triviaqa` calls `train_test_split(data, train_size=10000, random_state=42)` on each split, so the practical max sample count per split is 10 000, not the full TriviaQA. `--n_samples all` downstream means "all of those 10 000."

## Phase 0 pipeline

Driver: `scripts/run_phase0.sh`.

Base config (fixed for all phases unless noted):

| key | value |
| --- | --- |
| model | `mistralai/Mistral-7B-Instruct-v0.2` |
| dataset | `triviaqa` (+ `triviaqa_test`) |
| n_samples (generation) | 1000 per split |
| probe extraction point | `mlp` |
| probe layer | 15 |
| probe token | `exact_answer_last_token` |
| seeds | 0 5 26 42 63 |

Run:

```bash
bash scripts/run_phase0.sh   # ~30-60 min on a 3090
```

The script chains five steps; if any step fails the run aborts (`set -e`). Outputs land in `output/` (generation, extraction) and `checkpoints/` (probe).

### Step-by-step

1. `generate_model_answers.py --dataset triviaqa --n_samples 1000` →
   - `output/mistral-7b-instruct-answers-triviaqa.csv` (questions, generated answers, correctness)
   - `output/mistral-7b-instruct-input_output_ids-triviaqa.pt` (token ids, needed by probe.py)
   - `output/mistral-7b-instruct-scores-triviaqa.pt` (~10 GB of per-token logits; only used by the logprob/p_true baselines, not by the probe — safe to delete if disk-bound)
2. Same for `--dataset triviaqa_test`.
3. `extract_exact_answer.py --dataset triviaqa --extraction_model mistralai/Mistral-7B-Instruct-v0.2` overwrites the answers CSV in place, adding `exact_answer` and `valid_exact_answer` columns.
4. Same for `triviaqa_test`.
5. `probe.py --probe_at mlp --layer 15 --token exact_answer_last_token --save_clf --seeds 0 5 26 42 63 --n_samples all` trains five logistic-regression probes on hidden states at MLP output, layer 15, exact-answer-last-token. Saves the final probe (trained on the union train set) to `checkpoints/clf_mistral-7b-instruct_triviaqa_layer-15_token-exact_answer_last_token.pkl`.

### Reuse in later phases

- Phase 1 (probe lens) loads the pickle above and projects test-set hidden states from `triviaqa_test` onto the probe's `w`.
- Phase 2 (patching) hooks the MLP output of layer 15 — same extraction point as the probe.

## Results

Phase 0 acceptance bar (TZ): AUROC ≈ 0.7-0.8 (paper Table 1, Mistral-7B-Instruct row, TriviaQA column). **Result: 0.797 ± 0.015 — within the bar.**

Settings: layer 15, token `exact_answer_last_token`, probe_at `mlp`, n=1000 per split, batch_size=8, seeds `0 5 26 42 63`, LogisticRegression.

| metric | mean ± std |
| --- | --- |
| AUROC (held-out `triviaqa_test`) | **0.797 ± 0.015** |
| AUROC (validation = 1000-train → internal split) | 0.785 ± 0.044 |
| Acc at default 0.5 threshold (test) | 0.783 ± 0.016 |
| Baseline acc — majority class (test) | 0.720 ± 0.014 |
| Δ acc vs baseline (test) | +6.3 ± 1.3 pp |
| Precision (test) | 0.755 ± 0.027 |
| Recall (test) | 0.332 ± 0.050 |
| F1 (test) | 0.459 ± 0.049 |

Notes:
- Baseline acc 0.72 on `triviaqa_test` means ≈28% of Mistral-7B-Instruct's greedy answers are wrong at n=1000.
- The probe is conservative at threshold 0.5 (high precision, low recall). The 0.797 AUROC is the threshold-free summary and the number to compare to the paper.
- Std on the held-out test is tight (1.5pp); validation std is larger (4.4pp) because the train pool is only 1000 and seed-driven splits move more.

Saved probe: `checkpoints/clf_mistral-7b-instruct_triviaqa_layer-15_token-exact_answer_last_token.pkl` — Phase 1 reuses this directly.
