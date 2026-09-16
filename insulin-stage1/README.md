# Encoder Model

This directory contains the encoder training, evaluation, and inference code.

## Entry Points

- `train.py`: train the encoder or evaluate an existing checkpoint.
- `inference.py`: run the test-set inference workflow and save visualizations.
- `run_latest_topk_inference.py`: run batch evaluation and prediction export.
- `models/encoder.py`: encode patient history and retrieval memory.
- `models/decoder.py`: generate the daily glucose and insulin sequence.
- `data/`: data loading and feature preparation.
- `retrieval_model/`: retrieval components and the packaged retrieval checkpoint.

## Checkpoint

Place the Release model file at:

```text
checkpoints/topk10/model.pth
```

## Commands

Run commands from this directory. The default data paths are documented in
the repository root `README.md`.

```bash
python train.py
python train.py --test_only 1 --load_model_path ./checkpoints/topk10
python inference.py --load_model_path ./checkpoints/topk10
python run_latest_topk_inference.py --help
```
