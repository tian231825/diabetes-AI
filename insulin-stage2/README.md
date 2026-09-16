# Insulin Stage 2

This directory contains the stage-2 training, evaluation, and inference code.

Main entry points:

- `train.py`: train stage 2 or evaluate an existing checkpoint.
- `inference.py`: run the test-set inference workflow and save visualizations.
- `run_latest_topk_inference.py`: run batch evaluation and prediction export.
- `models/encoder.py`: encode patient history and retrieval memory.
- `models/decoder.py`: generate the stage-2 daily glucose and insulin sequence.
- `models/`: model definitions.
- `data/`: data loading and feature processing.
- `retrieval_model/`: retrieval components and checkpoint.

## Inference

Place the Release model file at:

```text
checkpoints/top17/model.pth
```

Run commands from this directory. The default data paths are documented in the
repository root `README.md`.

```bash
python train.py
python train.py --test_only 1 --load_model_path ./checkpoints/top17
python inference.py --load_model_path ./checkpoints/top17
python run_latest_topk_inference.py --help
```
