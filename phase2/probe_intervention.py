"""Phase 2 extension: is the linear truthfulness probe causal?

For each TriviaQA test question we run four conditions with greedy decoding:
  baseline  : no intervention
  +alpha*w  : add normalized probe direction to MLP-out at layer L, every decode step
  -alpha*w  : subtract same direction
  +alpha*r  : add a fixed random unit direction (control)

Intervention applies only on incremental decode steps (output.shape[1] == 1),
not on prefill. The shift is added to every batch row / sequence position of the
MLP output of layer L.

Two stages:
  --stage alpha_sweep : run on a 50-question subset across a grid of alpha values,
                        pick alpha that maximizes acc(+w) - acc(-w) subject to a
                        coherence floor. Writes phase2/alpha_sweep.csv.
  --stage main        : run all four conditions on the test set at a chosen alpha.

Determinism: greedy decoding, batch=1.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from compute_correctness import compute_correctness_triviaqa  # noqa: E402
from probing_utils import (  # noqa: E402
    MODEL_FRIENDLY_NAMES,
    extract_internal_reps_specific_layer_and_token,
    load_model_and_validate_gpu,
    tokenize,
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LAYER = 15
MAX_NEW_TOKENS = 100
HIDDEN = 4096
PROBE_PATH = REPO_ROOT / "checkpoints" / "clf_mistral-7b-instruct_triviaqa_layer-15_token-exact_answer_last_token.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["alpha_sweep", "main"], required=True)
    p.add_argument("--n_sweep", type=int, default=50,
                   help="questions used in the alpha sweep")
    p.add_argument("--n_test", type=int, default=1000,
                   help="cap on test questions for the main run")
    p.add_argument("--n_for_H", type=int, default=500,
                   help="training samples used to estimate mean activation norm H")
    p.add_argument("--alphas_rel", type=float, nargs="+",
                   default=[0.01, 0.05, 0.10, 0.15, 0.20, 0.30],
                   help="alphas as fractions of H (alpha_sweep stage)")
    p.add_argument("--alpha_rel", type=float, default=None,
                   help="chosen alpha/H for the main stage (required for stage=main)")
    p.add_argument("--coherence_floor", type=float, default=0.7,
                   help="min fraction of coherent answers required to consider an alpha")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "phase2"))
    return p.parse_args()


# ----------------------------- probe + directions -----------------------------

def load_probe_direction():
    """Return (w_unit, w_norm) where w_unit is the unit-norm probe direction
    pointing toward the 'correct' class. clf.coef_ has shape (1, hidden)
    with classes_=[0, 1] (1=correct), so coef_[0] points toward correct.
    """
    with open(PROBE_PATH, "rb") as f:
        clf = pickle.load(f)
    assert tuple(clf.classes_) == (0, 1), f"unexpected classes_: {clf.classes_}"
    w = clf.coef_[0].astype(np.float32)
    assert w.shape == (HIDDEN,), w.shape
    norm = float(np.linalg.norm(w))
    return torch.from_numpy(w / norm), norm


def make_random_direction(seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(HIDDEN, generator=g, dtype=torch.float32)
    return r / r.norm()


# ----------------------------- H estimation -----------------------------

def estimate_H(model, tokenizer, n):
    """Mean L2 norm of the layer-15 MLP-output activation at the exact-answer-
    last-token position, taken over n training samples (same setup as probe
    training). Single number used to scale alpha.
    """
    train_csv = REPO_ROOT / "output" / f"{MODEL_FRIENDLY_NAMES[MODEL_NAME]}-answers-triviaqa.csv"
    ids_pt = REPO_ROOT / "output" / f"{MODEL_FRIENDLY_NAMES[MODEL_NAME]}-input_output_ids-triviaqa.pt"
    print(f"[H] loading {train_csv}")
    data = pd.read_csv(train_csv).reset_index(drop=True)
    input_output_ids = torch.load(ids_pt)

    keep_pos = data.index[
        (data["valid_exact_answer"] == 1)
        & (data["exact_answer"] != "NO ANSWER")
        & (data["exact_answer"].map(type) == str)
    ].tolist()[:n]
    valid = data.iloc[keep_pos].reset_index(drop=True)
    ids_subset = [input_output_ids[i] for i in keep_pos]

    print(f"[H] extracting MLP-out @ layer {LAYER}, exact-answer-last-token, n={len(valid)}")
    reps = extract_internal_reps_specific_layer_and_token(
        model, tokenizer,
        prompts=valid["question"].tolist(),
        input_output_ids_lst=ids_subset,
        probe_at="mlp",
        model_name=MODEL_NAME,
        layer=LAYER,
        token="exact_answer_last_token",
        exact_answers=valid["exact_answer"].tolist(),
        exact_answers_valid=valid["valid_exact_answer"].tolist(),
        batch_size=8,
    )
    X = np.asarray(reps, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1)
    H = float(norms.mean())
    print(f"[H] mean norm = {H:.3f}  (median {np.median(norms):.3f}, n={len(norms)})")
    return H


# ----------------------------- intervention -----------------------------

@torch.no_grad()
def generate_with_shift(model, tokenizer, prompt, shift):
    """Greedy generate. If `shift` is not None, add it to MLP-out at LAYER on
    every incremental decode forward (output.shape[1] == 1). Prefill is untouched.
    """
    input_ids = tokenize(prompt, tokenizer, MODEL_NAME)
    mod = model.model.layers[LAYER].mlp

    if shift is not None:
        shift_t = shift.to(model.device).to(model.dtype)

        def hook(_m, _i, output):
            if output.shape[1] == 1:
                output[:, 0, :] = output[:, 0, :] + shift_t
            return output

        h = mod.register_forward_hook(hook)
    else:
        h = None
    try:
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    finally:
        if h is not None:
            h.remove()
    return tokenizer.decode(out.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)


# ----------------------------- coherence -----------------------------

def is_coherent(text):
    """Cheap garbage filter for the alpha sweep. An answer is 'coherent' if it
    is non-empty after strip, has at least one ASCII letter, and the most-common
    character is < 50% of the string. This catches the typical large-alpha
    failure modes (empty, all whitespace, repeating tokens) without being strict
    about answer quality.
    """
    s = (text or "").strip()
    if len(s) < 2:
        return False
    if not any(c.isalpha() and ord(c) < 128 for c in s):
        return False
    from collections import Counter
    c = Counter(s)
    return c.most_common(1)[0][1] / len(s) < 0.5


# ----------------------------- per-condition pass -----------------------------

def run_condition(model, tokenizer, prompts, labels, shift, desc):
    """Greedy-generate every prompt with the given shift. Returns
    (answers, correctness_list).
    """
    answers = []
    for p in tqdm(prompts, desc=desc, leave=False):
        answers.append(generate_with_shift(model, tokenizer, p, shift))
    corr = compute_correctness_triviaqa(answers, labels)["correctness"]
    return answers, corr


# ----------------------------- stages -----------------------------

def load_test_subset(n, seed):
    test_csv = REPO_ROOT / "output" / f"{MODEL_FRIENDLY_NAMES[MODEL_NAME]}-answers-triviaqa_test.csv"
    data = pd.read_csv(test_csv).reset_index(drop=True)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(data))[:n]
    sub = data.iloc[idx].reset_index(drop=True)
    return sub, idx


def stage_alpha_sweep(args, model, tokenizer, w_unit, r_unit, H):
    sub, _ = load_test_subset(args.n_sweep, args.seed)
    prompts = sub["question"].tolist()
    labels = sub["correct_answer"].tolist()

    baseline_answers, baseline_corr = run_condition(
        model, tokenizer, prompts, labels, shift=None, desc="baseline"
    )
    base_acc = float(np.mean(baseline_corr))
    base_coh = float(np.mean([is_coherent(a) for a in baseline_answers]))
    print(f"[sweep] baseline acc={base_acc:.3f} coherence={base_coh:.3f}")

    rows = [{
        "alpha_rel": 0.0, "alpha": 0.0, "condition": "baseline",
        "accuracy": base_acc, "coherence": base_coh, "n": len(prompts),
    }]

    for ar in args.alphas_rel:
        alpha = ar * H
        for cond, vec in [("+w", w_unit * alpha),
                          ("-w", -w_unit * alpha),
                          ("+r", r_unit * alpha)]:
            ans, corr = run_condition(model, tokenizer, prompts, labels, vec, f"a={ar:.2f} {cond}")
            acc = float(np.mean(corr))
            coh = float(np.mean([is_coherent(a) for a in ans]))
            print(f"[sweep] alpha={ar:.2f}H ({alpha:.2f}) {cond}: acc={acc:.3f} coh={coh:.3f}")
            rows.append({
                "alpha_rel": ar, "alpha": alpha, "condition": cond,
                "accuracy": acc, "coherence": coh, "n": len(prompts),
            })

    out = Path(args.out_dir) / "alpha_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"[sweep] wrote {out}")

    # Pick best alpha: max acc(+w) - acc(-w) subject to coherence(+w) and
    # coherence(-w) both >= floor.
    best = None
    for ar in args.alphas_rel:
        d = df[df["alpha_rel"] == ar].set_index("condition")
        if "+w" not in d.index or "-w" not in d.index:
            continue
        if d.loc["+w", "coherence"] < args.coherence_floor:
            continue
        if d.loc["-w", "coherence"] < args.coherence_floor:
            continue
        gap = d.loc["+w", "accuracy"] - d.loc["-w", "accuracy"]
        if best is None or gap > best[1]:
            best = (ar, gap)

    if best is None:
        print(f"[sweep] no alpha met coherence floor {args.coherence_floor:.2f}; "
              f"direction may not be causal or floor too strict.")
    else:
        print(f"[sweep] suggested alpha_rel={best[0]:.3f} "
              f"(acc gap +w vs -w = {best[1]:+.3f})")
    return df, best


def stage_main(args, model, tokenizer, w_unit, r_unit, H):
    assert args.alpha_rel is not None, "stage=main requires --alpha_rel"
    alpha = args.alpha_rel * H
    print(f"[main] alpha_rel={args.alpha_rel}  alpha={alpha:.3f}  H={H:.3f}")

    sub, _ = load_test_subset(args.n_test, args.seed)
    prompts = sub["question"].tolist()
    labels = sub["correct_answer"].tolist()
    print(f"[main] n_questions={len(prompts)}")

    conditions = [
        ("baseline", None),
        ("+w",  w_unit * alpha),
        ("-w", -w_unit * alpha),
        ("+r",  r_unit * alpha),
    ]
    per_q = pd.DataFrame({"q_idx": sub.index, "question": prompts, "label": labels})
    accs = {}
    for name, vec in conditions:
        ans, corr = run_condition(model, tokenizer, prompts, labels, vec, name)
        per_q[f"answer_{name}"] = ans
        per_q[f"correct_{name}"] = corr
        accs[name] = float(np.mean(corr))
        print(f"[main] {name:>8}: acc={accs[name]:.4f}")

    per_q_path = Path(args.out_dir) / "probe_intervention_per_question.csv"
    per_q.to_csv(per_q_path, index=False)
    print(f"[main] per-question -> {per_q_path}")

    base = accs["baseline"]
    summary = pd.DataFrame([
        {"condition": "baseline", "accuracy": accs["baseline"], "delta_pp": 0.0},
        {"condition": "+alpha*w", "accuracy": accs["+w"], "delta_pp": (accs["+w"] - base) * 100},
        {"condition": "-alpha*w", "accuracy": accs["-w"], "delta_pp": (accs["-w"] - base) * 100},
        {"condition": "+alpha*r", "accuracy": accs["+r"], "delta_pp": (accs["+r"] - base) * 100},
    ])
    summary["alpha"] = alpha
    summary["alpha_rel"] = args.alpha_rel
    summary["n"] = len(prompts)
    summary_path = Path(args.out_dir) / "probe_intervention_results.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[main] summary -> {summary_path}")
    print(summary.to_string(index=False))
    print(f"\nkey numbers:")
    print(f"  delta(+w) - delta(+r) = {(accs['+w'] - accs['+r']) * 100:+.2f} pp")
    print(f"  delta(+w) - delta(-w) = {(accs['+w'] - accs['-w']) * 100:+.2f} pp")


# ----------------------------- main -----------------------------

def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("[probe] loading direction")
    w_unit, w_raw_norm = load_probe_direction()
    r_unit = make_random_direction(args.seed)
    cos = float((w_unit * r_unit).sum())
    print(f"[probe] ||w||={w_raw_norm:.3f}  cos(w, r)={cos:+.4f}")

    print("[model] loading")
    model, tokenizer = load_model_and_validate_gpu(MODEL_NAME)
    model.eval()

    H = estimate_H(model, tokenizer, args.n_for_H)

    if args.stage == "alpha_sweep":
        stage_alpha_sweep(args, model, tokenizer, w_unit, r_unit, H)
    else:
        stage_main(args, model, tokenizer, w_unit, r_unit, H)


if __name__ == "__main__":
    main()
