#!/usr/bin/env bash
set -e
python src/eval.py --config configs/baseline_lstm.yaml --ckpt experiments/baseline_lstm/best.ckpt
python src/eval.py --config configs/baseline_transformer.yaml --ckpt experiments/baseline_transformer/best.ckpt
python src/eval.py --config configs/hybrid.yaml --ckpt experiments/hybrid/best.ckpt
