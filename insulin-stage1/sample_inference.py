"""Run one sample query against every other patient in the sample data."""

import json
from pathlib import Path

import torch
import models.model as model_module
from configs.Config import opt_config
from data.dataloader2 import DiabetesDataset, collate_fn


STAGE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = STAGE_DIR.parent / "sample_data" / "cohort" / "stage1"
CANDIDATE_ID = "0001778620_1446600"


class AllSamplePatients(DiabetesDataset):
    """Keep the processed sample cohort available as the retrieval pool."""

    def __init__(self, cfg, mode="all"):
        super().__init__(cfg, mode="train")
        self.filtered_indices = list(range(len(self.tensor_data)))
        self.filtered_data = list(self.tensor_data)
        self.n = len(self.filtered_data)
        self.start_idx = 0

def _use_sample_defaults(cfg):
    cfg.file_path_d1 = str(STAGE_DIR / "latest_data" / "data_simplified_basic.sample.json")
    cfg.file_path_d2 = str(STAGE_DIR / "latest_data" / "data_simplified_premix.sample.json")
    cfg.file_path_d1_extra = str(STAGE_DIR / "latest_data" / "data_simplified_basic_extra.sample.json")
    cfg.file_path_d2_extra = str(STAGE_DIR / "latest_data" / "data_simplified_premix_extra.sample.json")
    cfg.personality_path = str(SAMPLE_DIR / "personality_base.sample.json")
    cfg.personality_path_extra = str(SAMPLE_DIR / "personality_extra.sample.json")
    cfg.load_model_path = str(STAGE_DIR / "checkpoints" / "topk10")


def main():
    cfg = opt_config()
    _use_sample_defaults(cfg)
    model_module.DiabetesDataset = AllSamplePatients
    dataset = AllSamplePatients(cfg)
    samples = {item["check_id"]: item for item in dataset.fetch_database()}
    if CANDIDATE_ID not in samples:
        raise KeyError(f"Candidate {CANDIDATE_ID} is not present in sample data")

    pool = dataset.fetch_database()
    sample_index = next(index for index, item in enumerate(pool) if item["check_id"] == CANDIDATE_ID)
    sample = pool[sample_index]
    cfg.d_person = int(sample["d_per1"])
    cfg.d_insulin = int(sample["d_insulin"])
    cfg.d_drug = int(sample["d_drug"])
    checkpoint = Path(cfg.load_model_path) / "model.pth"
    state = torch.load(str(checkpoint), map_location="cpu")
    time_embedding = state.get("stage2.time_embedding.weight")
    if time_embedding is not None:
        cfg.max_seq_len = int(time_embedding.shape[0])
    model = model_module.TwoStageModel(cfg).to(cfg.device)
    if checkpoint.exists():
        model.load_state_dict(state)
    model.eval()
    batch = collate_fn(pool)
    batch = {key: value.to(cfg.device) if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.no_grad():
        outputs = model(batch, tf_ratio=0.0, mode="test")
    insulin_output = outputs[0][sample_index:sample_index + 1].detach().cpu().tolist()
    bg_output = outputs[1][sample_index:sample_index + 1].detach().cpu().tolist()
    topk_indices, topk_scores = model._retrieve_topk_indices(batch)
    topk_ids = [pool[int(index)]["check_id"] for index in topk_indices[sample_index]]
    output = {
        "check_id": CANDIDATE_ID,
        "knowledge_base_size": len(pool),
        "batch_size": len(pool),
        "retrieval_topk": [{"check_id": key, "score": float(score)} for key, score in zip(topk_ids, topk_scores[sample_index])],
        "insulin_prediction": insulin_output,
        "bg_prediction": bg_output,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
