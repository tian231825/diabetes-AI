# Diabetes-AI

This repository contains the encoder and decoder implementations for insulin
prediction, the retrieval components used by both models, and the medication
recommendation service.

The two large model files are distributed in the GitHub Release named
`Diabetes-AI large files`.

## Contents

- `insulin-stage1/`: stage-1 training, evaluation, and inference.
- `insulin-stage2/`: stage-2 training, evaluation, and inference.
- `drug-model/`: stage2 drug recommendation logic and service entry points.
- `sample_data/`: small examples for checking the input format. It does not
  contain the study data.

## Requirements

Use Python 3.8 or a compatible environment. Install the packages listed in
`requirements.txt`. GPU execution requires a PyTorch build compatible with the
installed CUDA runtime.

## How to Use

### 1. Download the model files

Download the two `.pth` assets from the Release `Diabetes-AI large files` and
place them at the following paths. The filenames must be changed to
`model.pth`, because the entry points load that filename.

```text
insulin-stage1/checkpoints/topk10/model.pth
insulin-stage2/checkpoints/top17/model.pth
```

The retrieval checkpoints are already included under each stage's
`retrieval_model/results/` directory:

```text
retrieval_model/results/daily_moco/current/daily_moco_sim.pt
retrieval_model/results/person_pair/current/person_pair_similarity.pt
```

### 2. Prepare input data

Run each stage from its own directory. The default configuration expects the
following files in that directory:

```text
latest_data/
  data_simplified_basic.json
  data_simplified_premix.json
  data_simplified_basic_extra.json
  data_simplified_premix_extra.json
merged_v3.json
merged_v4_L.json
```

Full cohort and clinical data are not part of this repository. Pass different
locations with the `--file_path_*` and `--personality_path*` options when the
files are stored elsewhere. Use `--cohort_ids_path` or `--fixed_split_path`
only when a fixed cohort or split is required.

### 3. Run the candidate-record sample

The repository includes a small candidate-record sample and a small retrieval
pool. After placing both Release model files at the checkpoint paths above,
run the sample entry point from the corresponding stage directory:

```bash
cd insulin-stage1
python sample_inference.py

cd ../insulin-stage2
python sample_inference.py
```

The sample entry points use the bundled sample files and run the candidate
record `0001778620_1446600` against the sample retrieval pool. They print the
retrieved records together with the insulin and glucose predictions.

### Sample data notes

The `sample_data/` directory is a minimal dataset intended to demonstrate the
required data structure and run the bundled sample workflow. It is not a
replacement for the full study or clinical datasets.

Some sample insulin doses are non-repeating decimals. These values usually
come from continuous insulin delivery by an insulin pump. The calculation is:

```text
continuous insulin dose = recorded dose × effective duration (hours) / 24
```

### 4. Train

```bash
cd insulin-stage1
python train.py

cd ../insulin-stage2
python train.py
```

Training outputs are written to `--save_path`.

### 5. Test an existing model

```bash
cd insulin-stage1
python train.py --test_only 1 --load_model_path ./checkpoints/topk10

cd ../insulin-stage2
python train.py --test_only 1 --load_model_path ./checkpoints/top17
```

The test command loads `model.pth` and evaluates the test split with
teacher forcing disabled.

### 6. Run inference

The stage inference entry points load the test data, evaluate the model, and
write prediction visualizations below `inference_outputs/` inside the model
directory.

```bash
cd insulin-stage1
python inference.py --load_model_path ./checkpoints/topk10

cd ../insulin-stage2
python inference.py --load_model_path ./checkpoints/top17
```

For batch evaluation and prediction tables, use the retained helper:

```bash
cd insulin-stage1
python run_latest_topk_inference.py --help

cd ../insulin-stage2
python run_latest_topk_inference.py --help
```

The stage2 drug service entry points are `drug-model/medication_api.py` and
`drug-model/stage2_api.py`. The shared rules are in
`drug-model/medication_rules.py`. Start the required service with
Uvicorn from the directory that contains its configuration and model files.

## Data Policy

Do not place cohort, clinical, prediction, evaluation, temporary, or training
output files in this repository. Only the small format examples under
`sample_data/` are included.
