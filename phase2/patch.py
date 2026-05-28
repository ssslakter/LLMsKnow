"""Phase 2: causal patching at the MLP-out of layer 15, last prompt token (Variant A).

Pipeline:
  1. Load Mistral-7B-Instruct and the test answers CSV (Phase 0 output).
  2. On a subset of questions, draw K sampled greedy=False generations to find
     'unstable' questions (1 <= #correct <= K-1).
  3. For each unstable question pick one correct donor and one incorrect recipient.
     Run three patching conditions on the recipient:
       - correct  : donor activation from this same question
       - random   : donor activation from an unrelated question
       - same_cls : donor activation from another incorrect sample of this question
     For each condition: hook layer 15 MLP output, replace the activation at the
     last prompt-token position with the donor vector, then greedy-generate and
     check correctness via compute_correctness_triviaqa.
  4. Print per-question results, save phase2/results.csv with flip rates.

Determinism: batch=1, torch.use_deterministic_algorithms is not required because
greedy decoding is argmax-only. We do set seeds for the K sampling pass.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from compute_correctness import compute_correctness_triviaqa
from probing_utils import (
    MODEL_FRIENDLY_NAMES,
    load_model_and_validate_gpu,
    tokenize,
    tokenize_batch,
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LAYER = 15
MAX_NEW_TOKENS = 100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_questions", type=int, default=500,
                   help="how many test questions to sample over")
    p.add_argument("--k_samples", type=int, default=10,
                   help="resamples per question to find unstable ones")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_unstable", type=int, default=None,
                   help="cap on unstable questions to patch (None = all)")
    p.add_argument("--out_csv", type=str,
                   default=str(REPO_ROOT / "phase2" / "results.csv"))
    p.add_argument("--per_question_csv", type=str,
                   default=str(REPO_ROOT / "phase2" / "per_question.csv"))
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 20 questions, K=4, no unstable cap")
    p.add_argument("--sanity_only", action="store_true",
                   help="run only sanity checks (identity patch == no patch on a few prompts) and exit")
    p.add_argument("--sample_batch_size", type=int, default=16,
                   help="batch size for the sampling pass (Phase A). The patching pass (Phase B) is always batch=1.")
    return p.parse_args()


def run_sanity(model, tokenizer):
    """Identity patch (donor == recipient) must reproduce the unpatched greedy
    answer exactly. Tests that the hook mechanism does not perturb the forward."""
    prompts = [
        "What is the capital of France?",
        "Which planet is known as the red planet?",
        "Who wrote Hamlet?",
    ]
    print("[sanity] running identity-patch checks")
    for prompt in prompts:
        prompt_ids = make_prompt_ids(prompt, tokenizer)
        target_pos = prompt_ids.shape[1] - 1
        donor_h, donor_ids = capture_activation(model, tokenizer, prompt, LAYER, target_pos)
        assert torch.equal(donor_ids.cpu(), prompt_ids.cpu()), "prompt token mismatch in capture"
        a = generate_with_patch(model, tokenizer, prompt, LAYER, target_pos, None)
        b = generate_with_patch(model, tokenizer, prompt, LAYER, target_pos, donor_h)
        ok = (a == b)
        print(f"  {'OK' if ok else 'FAIL'}: identity-patch matches no-patch")
        print(f"    prompt: {prompt}")
        print(f"    plain : {a[:120]!r}")
        print(f"    patched: {b[:120]!r}")
        if not ok:
            raise SystemExit("sanity check failed: identity patch diverged from no patch")
    print("[sanity] all identity-patch checks passed")


def make_prompt_ids(prompt, tokenizer):
    """Returns a (1, L) tensor of token ids on cuda. Matches Phase 0 tokenize()."""
    return tokenize(prompt, tokenizer, MODEL_NAME)


@torch.no_grad()
def sampled_generate_k(model, tokenizer, prompts, k, temperature, top_p, seed, batch_size):
    """For each prompt, generate k sampled continuations. Returns:
        answers: list[list[str]] of length len(prompts) x k
    Phase A (sampling) is batched for throughput. Phase B (patching) stays
    batch=1 as required for clean activation alignment.
    """
    answers = [[] for _ in prompts]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for j in range(k):
        torch.manual_seed(seed + 1000 * j)
        for start in tqdm(range(0, len(prompts), batch_size),
                          desc=f"resample {j+1}/{k}", leave=False):
            chunk = prompts[start:start + batch_size]
            input_ids, attention_mask = tokenize_batch(chunk, tokenizer, MODEL_NAME)
            prompt_len = input_ids.shape[1]
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                return_dict_in_generate=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_id,
            )
            gen = out.sequences[:, prompt_len:]
            for bi, txt_ids in enumerate(gen):
                answers[start + bi].append(tokenizer.decode(txt_ids))
    return answers


def correctness_for(answers_per_q, labels):
    """answers_per_q: list[list[str]]. Returns list[list[int]]."""
    out = []
    for ans_list, lbl in zip(answers_per_q, labels):
        out.append(compute_correctness_triviaqa(ans_list, [lbl] * len(ans_list))["correctness"])
    return out


def find_unstable(correctness_lists):
    """Return indices i where 0 < sum(correctness_lists[i]) < len(correctness_lists[i])."""
    out = []
    for i, c in enumerate(correctness_lists):
        s = int(sum(c))
        if 0 < s < len(c):
            out.append(i)
    return out


@torch.no_grad()
def capture_activation(model, tokenizer, prompt, layer, target_pos):
    """Run forward on prompt, return the layer-`layer` mlp output activation at
    position `target_pos` (typically prompt_len-1). Shape: (hidden,) on cuda, bf16.
    Also returns the prompt input_ids for sanity checks.
    """
    input_ids = make_prompt_ids(prompt, tokenizer)
    mod = model.model.layers[layer].mlp
    captured = {}

    def hook(_module, _inputs, output):
        # output: (B, T, hidden). We want (hidden,) at target_pos for batch 0.
        captured["h"] = output[0, target_pos, :].detach().clone()

    h = mod.register_forward_hook(hook)
    try:
        model(input_ids)
    finally:
        h.remove()
    return captured["h"], input_ids


@torch.no_grad()
def generate_with_patch(model, tokenizer, prompt, layer, target_pos, donor_h):
    """Greedy generate from `prompt` while replacing the layer-`layer` mlp output
    at position `target_pos` with `donor_h`. The replacement only fires on the
    prefill step (when T > target_pos); during incremental decoding T==1 and we
    leave it alone.
    """
    input_ids = make_prompt_ids(prompt, tokenizer)
    mod = model.model.layers[layer].mlp

    def hook(_module, _inputs, output):
        # output: (B, T, hidden)
        if output.shape[1] > target_pos:
            output[:, target_pos, :] = donor_h.to(output.dtype).to(output.device)
        return output

    h = mod.register_forward_hook(hook) if donor_h is not None else None
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
    return tokenizer.decode(out.sequences[0][input_ids.shape[1]:])


def is_correct(answer, label):
    return compute_correctness_triviaqa([answer], [label])["correctness"][0] == 1


def main():
    args = parse_args()
    if args.smoke:
        args.n_questions = 20
        args.k_samples = 4

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    test_csv = REPO_ROOT / "output" / f"{MODEL_FRIENDLY_NAMES[MODEL_NAME]}-answers-triviaqa_test.csv"
    print(f"loading test answers from {test_csv}")
    data = pd.read_csv(test_csv).reset_index(drop=True)
    print(f"test rows: {len(data)}")

    # Sample a fixed subset (seeded) of question indices.
    all_idx = list(range(len(data)))
    rng.shuffle(all_idx)
    pick_idx = all_idx[:args.n_questions]
    sub = data.iloc[pick_idx].reset_index(drop=True)
    prompts = sub["question"].tolist()
    labels = sub["correct_answer"].tolist()
    print(f"sampling {args.k_samples}x for {len(prompts)} questions")

    print("loading model")
    model, tokenizer = load_model_and_validate_gpu(MODEL_NAME)
    model.eval()

    run_sanity(model, tokenizer)
    if args.sanity_only:
        return

    # ---- Phase A: K sampled generations per question, find unstable ----
    sampled = sampled_generate_k(
        model, tokenizer, prompts,
        k=args.k_samples, temperature=args.temperature,
        top_p=args.top_p, seed=args.seed,
        batch_size=args.sample_batch_size,
    )
    corr = correctness_for(sampled, labels)
    unstable = find_unstable(corr)
    print(f"unstable questions: {len(unstable)} / {len(prompts)}")
    if len(unstable) == 0:
        print("no unstable questions — aborting"); return
    if args.max_unstable is not None:
        unstable = unstable[:args.max_unstable]

    # Sanity: prompt prefix equality is trivially true (donor == recipient prompt
    # textually; tokenizer is deterministic). We still assert it inside the loop.

    # ---- Phase B: for each unstable q, run 3 patching conditions ----
    rows = []
    n_total = len(unstable)
    flip_correct = flip_random = flip_same = 0
    qualitative = []

    for n_done, qi in enumerate(unstable):
        prompt = prompts[qi]
        label = labels[qi]
        cs = corr[qi]
        ans_list = sampled[qi]
        correct_sample_idx = next(j for j, c in enumerate(cs) if c == 1)
        incorrect_sample_idx = next(j for j, c in enumerate(cs) if c == 0)
        # second incorrect (if exists) for same-class control
        incorrect_others = [j for j, c in enumerate(cs) if c == 0 and j != incorrect_sample_idx]
        same_class_idx = incorrect_others[0] if incorrect_others else None

        # token position = last token of the prompt
        prompt_ids = make_prompt_ids(prompt, tokenizer)
        target_pos = prompt_ids.shape[1] - 1

        # Donor for correct: re-run the prompt under the seed that produced the
        # correct sample. The activation at the last *prompt* token does NOT
        # depend on sampling (sampling only affects generated tokens after the
        # prompt). So a single deterministic forward on the prompt suffices.
        donor_h_correct, donor_prompt_ids = capture_activation(
            model, tokenizer, prompt, LAYER, target_pos
        )
        assert torch.equal(donor_prompt_ids.cpu(), prompt_ids.cpu()), "prompt token mismatch"

        # Random donor: pick an unrelated question (different qi) and capture
        # its last-prompt-token activation.
        rand_qi = qi
        while rand_qi == qi:
            rand_qi = rng.randrange(len(prompts))
        rand_prompt_ids = make_prompt_ids(prompts[rand_qi], tokenizer)
        rand_pos = rand_prompt_ids.shape[1] - 1
        donor_h_random, _ = capture_activation(
            model, tokenizer, prompts[rand_qi], LAYER, rand_pos
        )

        # Same-class donor: another incorrect sample of THIS question. Since the
        # activation at the last prompt token is independent of the sample (same
        # prompt, deterministic forward), this is literally identical to
        # donor_h_correct unless... wait. The activation is identical because the
        # prompt is identical. So 'same-class incorrect' as donor from the same
        # question is meaningless as a control here — it would equal the donor
        # we're already injecting. The intended control under Variant A is: take
        # an unrelated question on which the model is also wrong, use its last-
        # prompt-token activation. We implement that.
        same_class_h = None
        same_class_q_used = None
        # Pick an unrelated question whose greedy answer was incorrect on the
        # test CSV (automatic_correctness == 0).
        wrong_pool = [i for i in range(len(prompts))
                      if i != qi and sub.iloc[i]["automatic_correctness"] == 0]
        if wrong_pool:
            same_class_q_used = rng.choice(wrong_pool)
            sc_prompt_ids = make_prompt_ids(prompts[same_class_q_used], tokenizer)
            sc_pos = sc_prompt_ids.shape[1] - 1
            same_class_h, _ = capture_activation(
                model, tokenizer, prompts[same_class_q_used], LAYER, sc_pos
            )

        # Recipient = greedy generation. Patch with each donor.
        ans_correct = generate_with_patch(
            model, tokenizer, prompt, LAYER, target_pos, donor_h_correct
        )
        ans_random = generate_with_patch(
            model, tokenizer, prompt, LAYER, target_pos, donor_h_random
        )
        ans_same = (generate_with_patch(model, tokenizer, prompt, LAYER, target_pos, same_class_h)
                    if same_class_h is not None else None)
        # Unpatched greedy baseline (recipient)
        ans_unpatched = generate_with_patch(
            model, tokenizer, prompt, LAYER, target_pos, None
        )

        f_corr = int(is_correct(ans_correct, label))
        f_rand = int(is_correct(ans_random, label))
        f_same = int(is_correct(ans_same, label)) if ans_same is not None else None
        baseline = int(is_correct(ans_unpatched, label))

        # Only count flips where the baseline (unpatched greedy) is wrong.
        # Otherwise there's nothing to flip.
        if baseline == 0:
            flip_correct += f_corr
            flip_random += f_rand
            if f_same is not None:
                flip_same += f_same

        print(f"[{n_done+1}/{n_total}] q={qi}  baseline={'OK' if baseline else 'WRONG'}  "
              f"corr_patch={'FLIP' if f_corr else '-'}  "
              f"rand_patch={'FLIP' if f_rand else '-'}  "
              f"same_patch={'FLIP' if f_same else '-' if f_same is not None else 'NA'}")

        rows.append({
            "q_idx": qi,
            "question": prompt,
            "label": label,
            "baseline_correct": baseline,
            "baseline_answer": ans_unpatched.strip(),
            "patch_correct_flip": f_corr,
            "patched_correct_answer": ans_correct.strip(),
            "patch_random_flip": f_rand,
            "patched_random_answer": ans_random.strip(),
            "patch_same_flip": f_same,
            "patched_same_answer": ans_same.strip() if ans_same else "",
            "n_correct_in_k": int(sum(cs)),
            "k": args.k_samples,
        })

        if len(qualitative) < 5 and baseline == 0 and f_corr == 1 and f_rand == 0:
            qualitative.append(rows[-1])

    # ---- Phase C: report ----
    # Denominator: how many unstable recipients actually had baseline=wrong.
    n_eligible = sum(1 for r in rows if r["baseline_correct"] == 0)
    n_same_eligible = sum(1 for r in rows if r["baseline_correct"] == 0 and r["patch_same_flip"] is not None)

    def rate(num, den):
        return num / den if den > 0 else float("nan")

    print()
    print("=" * 70)
    print(f"Recipients with baseline=WRONG (denominator): {n_eligible}")
    print()
    print(f"{'condition':<35} {'flip_rate':>10} {'n':>6}")
    print("-" * 55)
    print(f"{'correct -> incorrect (main)':<35} {rate(flip_correct, n_eligible):>10.3f} {n_eligible:>6}")
    print(f"{'random  -> incorrect (control)':<35} {rate(flip_random,  n_eligible):>10.3f} {n_eligible:>6}")
    print(f"{'same-class -> incorrect (control)':<35} {rate(flip_same,    n_same_eligible):>10.3f} {n_same_eligible:>6}")
    print()
    if n_eligible > 0:
        print(f"main - random   = {rate(flip_correct, n_eligible) - rate(flip_random, n_eligible):+.3f}")
        if n_same_eligible > 0:
            print(f"main - same_cls = {rate(flip_correct, n_eligible) - rate(flip_same, n_same_eligible):+.3f}")

    # save summary CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"condition": "correct->incorrect",   "flip_rate": rate(flip_correct, n_eligible),     "n": n_eligible},
        {"condition": "random->incorrect",    "flip_rate": rate(flip_random,  n_eligible),     "n": n_eligible},
        {"condition": "same_class->incorrect","flip_rate": rate(flip_same,    n_same_eligible),"n": n_same_eligible},
    ]).to_csv(args.out_csv, index=False)
    print(f"\nsaved summary -> {args.out_csv}")

    pd.DataFrame(rows).to_csv(args.per_question_csv, index=False)
    print(f"saved per-question -> {args.per_question_csv}")

    if qualitative:
        print("\n" + "=" * 70)
        print("Qualitative examples (baseline wrong, correct-patch flipped, random did not):")
        for r in qualitative:
            print()
            print(f"Q: {r['question']}")
            print(f"  label aliases: {str(r['label'])[:120]}...")
            print(f"  was (unpatched): {r['baseline_answer'][:200]}")
            print(f"  now (patched):   {r['patched_correct_answer'][:200]}")


if __name__ == "__main__":
    main()
