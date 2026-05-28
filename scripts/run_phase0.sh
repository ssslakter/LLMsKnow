#!/usr/bin/env bash
# Phase 0: generate answers, extract exact answers, train probe on layer 15.
# Reproducing the central probing setup from Orgad et al. 2025 on Mistral-7B-Instruct + TriviaQA.
# Run from repo root: bash scripts/run_phase0.sh

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
N=1000
LAYER=15
TOKEN="exact_answer_last_token"
PROBE_AT="mlp"
SEEDS="0 5 26 42 63"
BATCH_SIZE=8

cd src

echo "==> [1/5] generate train answers"
uv run python generate_model_answers.py --model "$MODEL" --dataset triviaqa --n_samples "$N" --batch_size "$BATCH_SIZE"

echo "==> [2/5] generate test answers"
uv run python generate_model_answers.py --model "$MODEL" --dataset triviaqa_test --n_samples "$N" --batch_size "$BATCH_SIZE"

echo "==> [3/5] extract exact answer (train)"
uv run python extract_exact_answer.py --model "$MODEL" --dataset triviaqa --extraction_model "$MODEL" --batch_size "$BATCH_SIZE"

echo "==> [4/5] extract exact answer (test)"
uv run python extract_exact_answer.py --model "$MODEL" --dataset triviaqa_test --extraction_model "$MODEL" --batch_size "$BATCH_SIZE"

echo "==> [5/5] train probe (layer=$LAYER token=$TOKEN probe_at=$PROBE_AT)"
uv run python probe.py \
    --model "$MODEL" \
    --probe_at "$PROBE_AT" \
    --seeds $SEEDS \
    --n_samples all \
    --save_clf \
    --dataset triviaqa \
    --layer "$LAYER" \
    --token "$TOKEN" \
    --batch_size "$BATCH_SIZE"

echo "==> Phase 0 done."
