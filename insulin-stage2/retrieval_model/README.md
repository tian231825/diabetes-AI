# Retrieval Components

This directory contains the retrieval models used by the stage-2 model.

The retrieval runtime uses these packaged checkpoints:

```text
results/daily_moco/current/daily_moco_sim.pt
results/person_pair/current/person_pair_similarity.pt
```

The `shared/` package provides common data loading, feature processing, and
runtime utilities. The `models/` package contains the retrieval model code.

Retrieval training and inference entry points are located in:

```text
models/daily_moco/
models/person_pair/
```

The main stage-2 inference code loads the person-pair checkpoint
automatically. The DailyMoCo checkpoint is used by the retrieval training and
inference utilities.
