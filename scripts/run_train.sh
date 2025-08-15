#!/usr/bin/env bash
set -e
python src/train.py --config configs/baseline_lstm.yaml
python src/train.py --config configs/baseline_transformer.yaml
python src/train.py --config configs/hybrid.yaml
