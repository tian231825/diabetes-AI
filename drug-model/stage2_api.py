# -*- coding: utf-8 -*-
"""
FastAPI wrapper for insulin-stage2 TwoStageModel.

Runs on port 8001. Stage2 checkpoint with retrieval-augmented inference.
The web backend sends complete patient data; this service converts raw records
into model tensors and returns the JSON shape consumed by ModelPredictionResult.

Required payload fields:
- patient: person feature dict (c-peptide values are merged into person_value)
- blood_glucose_records: for stage=2, at least one positive stage-2 BG value;
  the first observed stage-2 BG row seeds bg2 initial state
- drug_recommendation OR treatment_records with at least one mappable drug:
  needed to build drug2 (one drug vector per predicted day, consumed in
decoder._forward_inference).

Optional / accepted for compatibility (not consumed by the model in the
current deployment, kept for shape compatibility and forward portability):
- treatment_records when drug_recommendation is supplied: insulin1 / insulin2
  are not read by the inference rollout; treatment_records is only used as a
  fallback source for drug2 when drug_recommendation is absent and at least one
  treatment drug maps to the stage2 drug vocabulary.
"""
from __future__ import annotations

import logging
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from configs.Config import opt_config
from data.dataloader2 import collate_fn
from models.model import TwoStageModel
from medication_rules import (
    DiabetesPrescriptionSystemRefactored,
    Patient,
    DrugCategory,
    DrugRecommendation,
    DrugRule,
)


# ============================================================================
# Regimen classification functions (copied from inference.py to avoid train.py dependency)
# ============================================================================

def _gaussian_consistency_score(ratio_tensor, mu, sigma, min_sigma=1e-3):
    sigma = max(float(abs(sigma)), min_sigma)
    return torch.exp(-0.5 * ((ratio_tensor - float(mu)) / sigma) ** 2)


def _load_rule_stats(cfg):
    default_stats = {
        "mu_bb": 0.45,
        "sigma_bb": 0.10,
        "mu_pm_am": 0.50,
        "sigma_pm_am": 0.10,
        "mu_pm_pm": 0.50,
        "sigma_pm_pm": 0.10,
        "n_bb_patients": 0,
        "n_pm_patients": 0,
        "source": "default_fallback",
    }
    return default_stats


def _compute_topk_regimen_prior(model, batch, horizon):
    topk_indices, topk_scores = model._retrieve_topk_indices(batch)
    weights = torch.softmax(topk_scores.float(), dim=1).cpu().numpy()

    batch_size = topk_indices.shape[0]
    prior_basic = np.zeros((batch_size, horizon), dtype=np.float64)
    prior_premix = np.zeros((batch_size, horizon), dtype=np.float64)

    for b in range(batch_size):
        for k, kb_index in enumerate(topk_indices[b].tolist()):
            ref = model._knowledge_base[kb_index]
            ref_regimen = ref.get("regimen2")
            ref_regimen_mask = ref.get("regimen2_mask")
            if ref_regimen is None or ref_regimen_mask is None:
                continue

            ref_regimen = ref_regimen.detach().cpu().numpy() if hasattr(ref_regimen, "detach") else np.asarray(ref_regimen)
            ref_regimen_mask = ref_regimen_mask.detach().cpu().numpy() if hasattr(ref_regimen_mask, "detach") else np.asarray(ref_regimen_mask)
            ref_horizon = min(horizon, ref_regimen.shape[0], ref_regimen_mask.shape[0])
            if ref_horizon <= 0:
                continue

            valid = ref_regimen_mask[:ref_horizon].sum(axis=-1) > 0
            basic = valid & (ref_regimen[:ref_horizon, 0] > 0.5)
            premix = valid & (ref_regimen[:ref_horizon, 1] > 0.5)

            w = float(weights[b, k])
            prior_basic[b, :ref_horizon] += w * basic.astype(np.float64)
            prior_premix[b, :ref_horizon] += w * premix.astype(np.float64)

    total = prior_basic + prior_premix
    zero_mask = total <= 1e-8
    prior_basic = np.where(zero_mask, 0.5, prior_basic / np.clip(total, 1e-8, None))
    prior_premix = np.where(zero_mask, 0.5, prior_premix / np.clip(total, 1e-8, None))

    device = batch["person_value"].device
    return (
        torch.as_tensor(prior_basic, dtype=torch.float32, device=device),
        torch.as_tensor(prior_premix, dtype=torch.float32, device=device),
    )


def _infer_rule_probabilistic_regimen_labels(dose_tensor, prior_basic, prior_premix, cfg, rule_stats):
    threshold = float(getattr(cfg, "effective_dose_threshold", 2.0))
    temperature = max(float(getattr(cfg, "rule_softmax_temperature", 0.2)), 1e-6)
    prior_weight = float(getattr(cfg, "rule_prior_weight", 0.35))
    eps = 1e-8

    dose_tensor = torch.clamp(dose_tensor, min=0.0)
    bb_slots = dose_tensor[..., :4]
    pm_slots = dose_tensor[..., 4:8]

    bb_trigger = (bb_slots > threshold).any(dim=-1)
    pm_am = pm_slots[..., 0] + pm_slots[..., 1]
    pm_pm = pm_slots[..., 2] + pm_slots[..., 3]
    pm_trigger = (pm_am > threshold) | (pm_pm > threshold)

    bb_total = bb_slots.sum(dim=-1)
    pm_total = pm_am + pm_pm

    r_bb = bb_slots[..., 3] / (bb_total + eps)
    r_pm_am = pm_am / (pm_total + eps)
    r_pm_pm = pm_pm / (pm_total + eps)

    bb_score = _gaussian_consistency_score(r_bb, rule_stats["mu_bb"], rule_stats["sigma_bb"])
    pm_am_score = _gaussian_consistency_score(r_pm_am, rule_stats["mu_pm_am"], rule_stats["sigma_pm_am"])
    pm_pm_score = _gaussian_consistency_score(r_pm_pm, rule_stats["mu_pm_pm"], rule_stats["sigma_pm_pm"])
    pm_score = torch.sqrt(torch.clamp(pm_am_score * pm_pm_score, min=0.0))

    huge_neg = torch.full_like(bb_score, -1e4)
    bb_logit = torch.where(
        bb_trigger,
        bb_score / temperature + prior_weight * torch.log(prior_basic + eps),
        huge_neg,
    )
    pm_logit = torch.where(
        pm_trigger,
        pm_score / temperature + prior_weight * torch.log(prior_premix + eps),
        huge_neg,
    )

    logits = torch.stack([bb_logit, pm_logit], dim=-1)
    probs = torch.softmax(logits, dim=-1)
    pred_label = torch.argmax(probs, dim=-1).long()
    no_trigger = ~(bb_trigger | pm_trigger)
    pred_label = torch.where(no_trigger, torch.full_like(pred_label, -1), pred_label)
    return pred_label, probs

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / "clinical_top18"
CHECKPOINT_FILE = CHECKPOINT_DIR / "model.pth"
SERVICE_LOG_DIR = PROJECT_DIR / "service_logs"
SERVICE_LOG_FILE = SERVICE_LOG_DIR / "inference_service.log"

DEVICE = "cpu"
TOP_K = 18
D_PERSON = 79  # 77 patient features + 2 c-peptide
PREDICTION_HORIZON = 7
D_BG = 7
D_INSULIN = 5  # [premix_flag, morning, noon, evening, long/night]
C_PEP_DIM = 2  # fasting + 2h
DRUG_MODE = 2
D_DRUG = 17

BG_FIELDS = [
    "fasting", "after_breakfast", "before_lunch", "after_lunch",
    "before_dinner", "after_dinner", "before_sleep",
]

FEATURE_NAMES = [
    "age",
    "gender",
    "height",
    "weight",
    "BMI",
    "duration_of_diabetes",
    "smoking_history",
    "drinking_history",
    "心血管病史",
    "肝炎史",
    "骨折史",
    "卒中史",
    "胰腺炎史",
    "甲状腺髓样癌史",
    "心衰史",
    "低血糖史",
    "生酮饮食",
    "胆囊切除术",
    "肺结核史",
    "收缩压",
    "舒张压",
    "heart_rate",
    "血氧饱和度",
    "呼吸",
    "体温",
    "高血压分级",
    "HbA1c",
    "HDL-C",
    "LDL-C",
    "TC",
    "TG",
    "ALT",
    "AST",
    "UA",
    "SCR",
    "TT4",
    "FT4",
    "TSH",
    "β-羟丁酸",
    "尿白细胞",
    "AST/ALT",
    "eGFR",
    "complications-糖尿病性周围神经病变",
    "complications-糖尿病性视网膜病变",
    "complications-糖尿病性周围血管病变",
    "complications-糖尿病性肾病",
    "complications-高血压",
    "complications-冠状动脉粥样硬化性心脏病",
    "complications-心力衰竭",
    "complications-失代偿性心力衰竭",
    "complications-慢性肾脏病",
    "complications-肾衰竭",
    "complications-肾透析",
    "complications-肾结石",
    "complications-蛋白尿",
    "complications-高脂血症",
    "complications-脂肪肝",
    "complications-代谢综合征",
    "complications-胃轻瘫",
    "complications-胃炎",
    "complications-胰腺炎",
    "complications-胃肠道不良反应",
    "complications-前列腺增生",
    "complications-前列腺炎",
    "complications-骨质疏松症",
    "complications-肝功能不全",
    "complications-酮症",
    "complications-酮症酸中毒",
    "complications-高渗高血糖综合征",
    "complications-甲状腺髓样癌",
    "complications-多发性内分泌腺瘤病2型",
    "糖尿病周围血管病分级",
    "肾积水",
    "CKD-N期",
    "血脂异常",
    "肥胖情况",
    "脂质代谢异常标志",
]

PATIENT_KEY_ALIASES = {
    "HDL-C": ("HDL_C", "HDL-C", "hdl_c"),
    "LDL-C": ("LDL_C", "LDL-C", "ldl_c"),
    "β-羟丁酸": ("beta_hydroxybutyrate",),
    "尿白细胞": ("urine_leukocyte", "尿白细胞"),
    "AST/ALT": ("AST_ALT_ratio", "ast_alt_ratio"),
    "高血压分级": ("hypertension_grade", "高血压分级"),
    "糖尿病周围血管病分级": ("vascular_disease_grade", "diabetic_peripheral_vascular_disease_grade", "糖尿病周围血管病分级"),
    "肾积水": ("hydronephrosis", "肾积水"),
}

PATIENT_SUMMARY_SOURCE_FILES = (
    "merged.json",
    "merged_v3.json",
    "merged_v4.json",
    "merged_v4_L.json",
)
PATIENT_SUMMARY_EXTRA_FILE = "feature_extra.json"
SUMMARY_BASIC_FIELDS = (
    ("age", "age"),
    ("gender", "gender"),
    ("height", "height"),
    ("weight", "weight"),
    ("BMI", "BMI"),
    ("duration_of_diabetes", "duration_of_diabetes"),
)
SUMMARY_VITAL_FIELDS = (
    ("收缩压", "Systolic Blood Pressure (SBP)"),
    ("舒张压", "Diastolic Blood Pressure (DBP)"),
    ("heart_rate", "Heart Rate (HR)"),
    ("血氧饱和度", "Oxygen Saturation (SpO2)"),
    ("呼吸", "Respiratory Rate (RR)"),
    ("体温", "Temperature (Temp)"),
)
SUMMARY_LAB_FIELDS = (
    ("HbA1c", "HbA1c"),
    ("HDL-C", "HDL-C"),
    ("LDL-C", "LDL-C"),
    ("TC", "TC"),
    ("TG", "TG"),
    ("ALT", "ALT"),
    ("AST", "AST"),
    ("UA", "UA"),
    ("SCR", "SCR"),
    ("TT4", "TT4"),
    ("FT4", "FT4"),
    ("TSH", "TSH"),
    ("β-羟丁酸", "BHB"),
    ("尿白细胞", "Urine WBC"),
    ("AST/ALT", "AST/ALT"),
    ("eGFR", "eGFR"),
)
SUMMARY_C_PEP_FIELDS = (
    ("C肽(空腹)", "C_pep"),
    ("C肽(2小时)", "C_pep_2h"),
)
SUMMARY_HISTORY_FIELDS = (
    ("smoking_history", "Smoking History"),
    ("drinking_history", "Alcohol History"),
    ("心血管病史", "Cardiovascular Disease History"),
    ("肝炎史", "Hepatitis History"),
    ("骨折史", "Fracture History"),
    ("卒中史", "Stroke History"),
    ("胰腺炎史", "Pancreatitis History"),
    ("甲状腺髓样癌史", "Medullary Thyroid Cancer History"),
    ("心衰史", "Heart Failure History"),
    ("低血糖史", "Hypoglycemia History"),
    ("生酮饮食", "Ketogenic Diet"),
    ("胆囊切除术", "Cholecystectomy"),
    ("胆囊切除术史", "Cholecystectomy"),
    ("肺结核史", "Tuberculosis History"),
    ("肾积水", "Hydronephrosis"),
)
SUMMARY_DIAGNOSIS_FIELDS = (
    ("糖尿病性周围神经病变", "Diabetic Peripheral Neuropathy"),
    ("糖尿病性视网膜病变", "Diabetic Retinopathy"),
    ("糖尿病性周围血管病变", "Diabetic Peripheral Vascular Disease"),
    ("糖尿病性肾病", "Diabetic Nephropathy"),
    ("高血压", "Hypertension"),
    ("冠状动脉粥样硬化性心脏病", "ASCVD"),
    ("心力衰竭", "Heart Failure"),
    ("失代偿性心力衰竭", "Decompensated Heart Failure"),
    ("慢性肾脏病", "Chronic Kidney Disease"),
    ("肾衰竭", "Renal Failure"),
    ("肾透析", "Dialysis"),
    ("肾结石", "Nephrolithiasis"),
    ("蛋白尿", "Proteinuria"),
    ("高脂血症", "Hyperlipidemia"),
    ("脂肪肝", "Fatty Liver Disease"),
    ("代谢综合征", "Metabolic Syndrome"),
    ("胃轻瘫", "Gastroparesis"),
    ("胃炎", "Gastritis"),
    ("胰腺炎", "Pancreatitis"),
    ("胃肠道不良反应", "GI Adverse Events"),
    ("前列腺增生", "Benign Prostatic Hyperplasia"),
    ("前列腺炎", "Prostatitis"),
    ("骨质疏松症", "Osteoporosis"),
    ("肝功能不全", "Hepatic Insufficiency"),
    ("酮症", "Ketosis"),
    ("酮症酸中毒", "Diabetic Ketoacidosis"),
    ("高渗高血糖综合征", "Hyperosmolar Hyperglycemic State"),
    ("甲状腺髓样癌", "Medullary Thyroid Cancer"),
    ("多发性内分泌腺瘤病2型", "MEN-2"),
    ("周围血管病分级", "Peripheral Vascular Disease Grade"),
)
SUMMARY_INDICATOR_FIELDS = (
    ("CKD-N期", "CKD Stage"),
    ("血脂异常", "Dyslipidemia"),
    ("肥胖情况", "Obesity"),
    ("脂质代谢异常标志", "Abnormal Lipid Metabolism Marker"),
    ("高血压分级", "Hypertension Grade"),
)
_PATIENT_SUMMARY_CACHE: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None

FEATURE_SCHEMA_VERSION = "stage2-person-v1-d79"
FEATURE_SCHEMA_DIM = D_PERSON
FEATURE_SCHEMA_HASH = hashlib.sha256(
    json.dumps(FEATURE_NAMES + ["C_pep", "C_pep_2h"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
).hexdigest()

DRUG_NAME_TO_INDEX = {
    "盐酸二甲双胍片": 0, "二甲双胍缓释片": 0, "二甲双胍": 0,
    "司美格鲁肽注射液": 1, "司美格鲁肽": 1,
    "度拉糖肽注射液": 2, "度拉糖肽": 2,
    "利拉鲁肽注射液": 3, "利拉鲁肽": 3,
    "利司那肽注射液": 4, "利司那肽": 4,
    "磷酸西格列汀片": 5, "西格列汀": 5,
    "利格列汀片": 6, "利格列汀": 6,
    "脯氨酸恒格列净片": 7, "恒格列净": 7,
    "艾托格列净片": 8, "艾托格列净": 8,
    "达格列净片": 9, "达格列净": 9,
    "恩格列净片": 10, "恩格列净": 10,
    "卡格列净": 11,
    "吡格列酮片": 12, "吡格列酮": 12,
    "西格列他钠片": 13, "西格列他钠": 13,
    "格列吡嗪缓释片": 14, "格列吡嗪": 14, "格列齐特": 14,
    "格列美脲片": 15, "格列美脲": 15,
    "格列喹酮片": 16, "格列喹酮": 16,
    "阿卡波糖片": 17, "阿卡波糖胶囊": 17, "阿卡波糖": 17,
    "伏格列波糖片": 18, "伏格列波糖": 18,
    "桑枝总生物碱片": 19,
    "米格列醇片": 20, "米格列醇": 20,
    "多格列艾汀片": 21, "多格列艾汀": 21,
    "瑞格列奈片": 22, "瑞格列奈": 22,
    "那格列奈片": 23, "那格列奈": 23,
}

# Map rule categories to the short names used by the API and model input.
RULE_DRUG_TO_CLEAN_NAME = {
    "二甲双胍": "二甲双胍",
    "SGLT2抑制剂-达格列净": "达格列净",
    "SGLT2抑制剂-恩格列净": "恩格列净",
    "SGLT2抑制剂-卡格列净": "卡格列净",
    "SGLT2抑制剂-艾托格列净": "艾托格列净",
    "SGLT2抑制剂-恒格列净": "恒格列净",
    "SGLT2抑制剂-加格列净": "加格列净",
    "GLP-1受体激动剂(贝那鲁肽/艾塞那肽/利司那肽/利拉鲁肽/司美格鲁肽)": "司美格鲁肽",
    "DPP4抑制剂-西格列汀": "西格列汀",
    "DPP4抑制剂-沙格列汀": "沙格列汀",
    "DPP4抑制剂-维格列汀": "维格列汀",
    "DPP4抑制剂-利格列汀": "利格列汀",
    "DPP4抑制剂-阿格列汀": "阿格列汀",
    "DPP4抑制剂-瑞格列汀": "瑞格列汀",
    "DPP4抑制剂-考格列汀": "考格列汀",
    "磺脲类-格列本脲": "格列本脲",
    "磺脲类-格列吡嗪": "格列吡嗪",
    "磺脲类-格列齐特": "格列齐特",
    "磺脲类-格列喹酮": "格列喹酮",
    "磺脲类-格列美脲": "格列美脲",
    "格列奈类": "瑞格列奈",
    "α-糖苷酶抑制剂(阿卡波糖,伏格列波糖,米格列醇,桑枝总生物碱片)": "阿卡波糖",
    "噻唑烷二酮类-吡格列酮": "吡格列酮",
    "噻唑烷二酮类-罗格列酮": "罗格列酮",
    "PPAR全激动剂-西格列他那纳": "西格列他钠",
    "葡萄糖激酶激活剂(多格列艾汀)": "多格列艾汀",
}


DRUG_NAME_TO_INDEX["Glinide"] = 22
RULE_DRUG_TO_CLEAN_NAME[DrugCategory.GLN.value] = "Glinide"

SERVICE_LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(SERVICE_LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

app = FastAPI(title="Insulin Stage2 Inference API", version="stage2-top18")


def _build_config():
    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        cfg = opt_config()
    finally:
        sys.argv = old_argv

    cfg.device = DEVICE
    cfg.top_k = TOP_K
    cfg.d_person = D_PERSON
    cfg.d_person1 = D_PERSON
    cfg.d_bg = D_BG
    cfg.d_insulin = D_INSULIN
    cfg.max_T1 = PREDICTION_HORIZON
    cfg.max_seq_len = PREDICTION_HORIZON
    cfg.load_model_path = str(CHECKPOINT_DIR)
    cfg.num_workers = 0
    cfg.drug_mode = DRUG_MODE
    cfg.d_drug = D_DRUG
    return cfg


class InferenceRuntime:
    def __init__(self) -> None:
        self.cfg = _build_config()
        self.device = torch.device(DEVICE)
        self.lock = threading.Lock()
        self.model = self._load_model()

    def _load_model(self) -> TwoStageModel:
        if not CHECKPOINT_FILE.exists():
            raise RuntimeError(f"checkpoint not found: {CHECKPOINT_FILE}")

        logging.info("Loading stage2 model from %s", CHECKPOINT_FILE)
        model = TwoStageModel(self.cfg).to(self.device)
        state = torch.load(str(CHECKPOINT_FILE), map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        logging.info("Stage2 model loaded successfully")
        return model

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        _validate_feature_schema(payload)
        stage = int(payload.get("stage") or 0)
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")

        patient = payload.get("patient") or {}
        bg_records = _records_for_stage(payload.get("blood_glucose_records"), stage)
        treatment_records = _records_for_stage(payload.get("treatment_records"), stage)
        drug_rec = _normalize_drug_recommendation(payload.get("drug_recommendation"))
        drug2_override = _normalize_drug2(payload.get("drug2"), PREDICTION_HORIZON)
        if stage == 2 and not _has_observed_bg(bg_records):
            raise ValueError("stage=2 has no blood glucose records (initial BG required)")

        # Stage-2 decoding uses the observed glucose state and daily drug input.
        # For stage1, drug2 can be a zero tensor (drug recommendation is an output).
        # For stage2, we need drug_recommendation or treatment_records.
        has_treatment_drug = _has_mappable_treatment_drug(treatment_records)
        if stage == 2 and drug2_override is None and not drug_rec and not has_treatment_drug:
            raise ValueError(
                f"stage={stage} requires drug2, non-empty drug_recommendation, or mappable treatment_records drug "
                "to build drug2"
            )

        check_id = str(
            payload.get("request_id")
            or patient.get("patient_id")
            or patient.get("case_number")
            or "web_request"
        )
        sample = _build_sample(check_id, patient, bg_records, treatment_records, self.cfg, drug_rec, stage, drug2_override)
        batch = collate_fn([sample])
        batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

        with self.lock, torch.no_grad():
            outputs = self.model(batch, tf_ratio=0.0, mode="infer")

        insulin_tensor = outputs[0][0]  # (T, 8)
        insulin = insulin_tensor.detach().cpu().tolist()
        bg = outputs[1][0].detach().cpu().tolist()
        predicted_length = min(len(bg), PREDICTION_HORIZON)
        if len(outputs) > 2:
            raw_len = outputs[2][0].detach().cpu().item()
            predicted_length = max(1, min(PREDICTION_HORIZON, int(round(float(raw_len)))))

        # Compute regimen labels (basic=0 / premix=1 / none=-1)
        regimen_labels = []
        regimen_probs_list = []
        try:
            if D_INSULIN == 5:
                pred = insulin_tensor[:predicted_length]
                flag_prob = torch.clamp(pred[:, 0], min=0.0, max=1.0)
                dose_active = pred[:, 1:].amax(dim=-1) > float(getattr(self.cfg, "effective_dose_threshold", 2.0))
                pred_label = torch.where(
                    dose_active,
                    torch.where(flag_prob >= 0.5, torch.ones_like(flag_prob, dtype=torch.long), torch.zeros_like(flag_prob, dtype=torch.long)),
                    torch.full_like(flag_prob, -1, dtype=torch.long),
                )
                probs = torch.stack([1.0 - flag_prob, flag_prob], dim=-1)
                regimen_labels = pred_label.detach().cpu().tolist()
                regimen_probs_list = probs.detach().cpu().tolist()
            else:
                rule_stats = _load_rule_stats(self.cfg)
                prior_basic, prior_premix = _compute_topk_regimen_prior(self.model, batch, predicted_length)
                pred_label, probs = _infer_rule_probabilistic_regimen_labels(
                    insulin_tensor[:predicted_length].unsqueeze(0),
                    prior_basic[:, :predicted_length],
                    prior_premix[:, :predicted_length],
                    self.cfg,
                    rule_stats,
                )
                regimen_labels = pred_label[0].detach().cpu().tolist()
                regimen_probs_list = probs[0].detach().cpu().tolist()
        except Exception as e:
            logging.warning("Regimen label inference failed: %s", e)
            regimen_labels = [-1] * predicted_length
            regimen_probs_list = [[0.5, 0.5]] * predicted_length

        return {
            "ok": True,
            "stage": stage,
            "request_check_id": check_id,
            "normalized_check_id": check_id,
            "mapped_check_id": None,
            "predicted_length": predicted_length,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_dim": FEATURE_SCHEMA_DIM,
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "bg_prediction": bg[:predicted_length],
            "insulin_prediction": insulin[:predicted_length],
            "regimen_label": regimen_labels,
            "regimen_probs": regimen_probs_list,
            "service_log_file": str(SERVICE_LOG_FILE),
        }

    def retrieval_cards(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        _validate_feature_schema(payload)
        stage = int(payload.get("stage") or 2)
        if stage != 2:
            raise ValueError("stage2 retrieval-cards only supports stage=2")

        patient = payload.get("patient") or {}
        bg_records = _records_for_stage(payload.get("blood_glucose_records"), stage)
        treatment_records = _records_for_stage(payload.get("treatment_records"), stage)
        drug_rec = _normalize_drug_recommendation(payload.get("drug_recommendation"))
        drug2_override = _normalize_drug2(payload.get("drug2"), PREDICTION_HORIZON)
        if not _has_observed_bg(bg_records):
            raise ValueError("stage=2 has no blood glucose records (initial BG required)")

        check_id = str(
            payload.get("request_id")
            or patient.get("patient_id")
            or patient.get("case_number")
            or "web_request"
        )
        sample = _build_sample(check_id, patient, bg_records, treatment_records, self.cfg, drug_rec, stage, drug2_override)
        batch = collate_fn([sample])
        batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

        with self.lock, torch.no_grad():
            topk_indices, topk_scores = self.model._retrieve_topk_indices(batch)

        scores = topk_scores[0].detach().cpu().float()
        temperature = max(float(getattr(self.model.cfg, "memory_fusion_temperature", 1.0)), 1e-6)
        weights = torch.softmax(scores / temperature, dim=0).tolist()

        def _as_int(value: Any, default: int) -> int:
            if torch.is_tensor(value):
                return int(value.detach().cpu().item())
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _rows(sample_dict: Dict[str, Any], key: str, length_key: str) -> List[List[Optional[float]]]:
            value = sample_dict.get(key)
            if value is None:
                return []
            if torch.is_tensor(value):
                arr = value.detach().cpu()
            else:
                arr = torch.tensor(value)
            length = _as_int(sample_dict.get(length_key), arr.shape[0])
            length = max(0, min(length, arr.shape[0]))
            return [
                [None if cell is None else round(float(cell), 2) for cell in row]
                for row in arr[:length].tolist()
            ]

        reference_cases = []
        for rank, kb_index in enumerate(topk_indices[0].detach().cpu().tolist(), start=1):
            retrieved_sample = self.model._knowledge_base[int(kb_index)]
            reference_cases.append({
                "rank": rank,
                "check_id": retrieved_sample.get("check_id"),
                "similarity_score": round(float(scores[rank - 1]), 6),
                "similarity_weight": round(float(weights[rank - 1]), 6),
                "patient_summary": _patient_summary(retrieved_sample.get("check_id"), stage),
                "stage1_bg_records": _rows(retrieved_sample, "bg1", "len1"),
                "stage1_insulin_records": _rows(retrieved_sample, "insulin1", "len1"),
                "stage2_bg_records": _rows(retrieved_sample, "bg2", "len2"),
                "stage2_insulin_records": _rows(retrieved_sample, "insulin2", "len2"),
                "bg_records": _rows(retrieved_sample, "bg2", "len2"),
                "insulin_records": _rows(retrieved_sample, "insulin2", "len2"),
            })

        insulin_columns = (
            ["预混标志", "早餐短效", "午餐短效", "晚餐短效", "当日长效"]
            if D_INSULIN == 5
            else [f"胰岛素{i + 1}" for i in range(D_INSULIN)]
        )
        return {
            "ok": True,
            "stage": stage,
            "request_check_id": check_id,
            "top_k": len(reference_cases),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_dim": FEATURE_SCHEMA_DIM,
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "bg_columns": ["空腹", "早餐后", "午餐前", "午餐后", "晚餐前", "晚餐后", "睡前"],
            "insulin_columns": insulin_columns,
            "reference_cases": reference_cases,
        }


def _runtime() -> InferenceRuntime:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        runtime = InferenceRuntime()
        app.state.runtime = runtime
    return runtime


@app.on_event("startup")
def startup_event() -> None:
    _runtime()


@app.get("/health")
def health() -> Dict[str, Any]:
    runtime = getattr(app.state, "runtime", None)
    return {
        "ok": runtime is not None and runtime.loaded,
        "model_loaded": runtime is not None and runtime.loaded,
        "version": app.version,
        "device": DEVICE,
        "top_k": TOP_K,
        "d_person": D_PERSON,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_dim": FEATURE_SCHEMA_DIM,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "checkpoint_path": str(CHECKPOINT_FILE),
    }


@app.post("/predict")
async def predict(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    _check_token(authorization)
    payload = await request.json()
    try:
        result = _runtime().predict(payload)
        return JSONResponse(result)
    except ValueError as exc:
        logging.warning("Bad predict request: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        logging.exception("Predict failed")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/retrieval-cards")
async def retrieval_cards(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    _check_token(authorization)
    payload = await request.json()
    try:
        result = _runtime().retrieval_cards(payload)
        return JSONResponse(result)
    except ValueError as exc:
        logging.warning("Bad retrieval-cards request: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        logging.exception("Retrieval-cards failed")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


_drug_system: Optional[DiabetesPrescriptionSystemRefactored] = None


def _get_drug_system() -> DiabetesPrescriptionSystemRefactored:
    global _drug_system
    if _drug_system is None:
        _drug_system = DiabetesPrescriptionSystemRefactored()
    return _drug_system


def _build_rule_patient(payload: Dict[str, Any]) -> Patient:
    """Convert an API payload to the internal patient representation."""
    patient = payload.get("patient") or {}
    stage = int(payload.get("stage") or 1)
    raw_records = payload.get("blood_glucose_records") or []
    bg_records = _records_for_stage(raw_records, stage)

    def _f(key: str) -> Optional[float]:
        v = patient.get(key)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _flag(key: str) -> bool:
        v = patient.get(key)
        if v is None or v == "" or v == 0:
            return False
        if isinstance(v, str):
            if v.strip().lower() in {"false", "no", "n", "0", "none", "null"}:
                return False
        try:
            return float(v) > 0
        except (TypeError, ValueError):
            return bool(v)

    age = _f("age") or 50.0
    bmi = _f("BMI")
    if bmi is None:
        height = _f("height")
        weight = _f("weight")
        if height and weight:
            height_m = height / 100 if height > 3 else height
            if height_m > 0:
                bmi = weight / (height_m * height_m)
    bmi = bmi or 24.0
    hba1c = _f("HbA1c") or 8.0
    duration = _f("duration_of_diabetes") or 1.0
    egfr = _f("eGFR") or 90.0
    alt_val = _f("ALT")
    ast_val = _f("AST")
    c_pep = _f("C_pep")
    c_pep_2h = _f("C_pep_2h")
    systolic = _f("systolic_pressure")
    diastolic = _f("diastolic_pressure")

    # Blood glucose
    paired_pre_vals = []
    paired_post_vals = []
    paired_delta_vals = []
    daily_high_delta_counts = []
    meal_pairs = (
        (("fasting",), "after_breakfast"),
        (("before_lunch",), "after_lunch"),
        (("before_dinner",), "after_dinner"),
    )
    for rec in bg_records:
        high_delta_meals_this_day = 0
        for pre_keys, post_key in meal_pairs:
            pre_meal_vals = []
            for k in pre_keys:
                v = rec.get(k)
                if v is not None:
                    try:
                        pre_meal_vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
            post_v = rec.get(post_key)
            try:
                post_meal_val = float(post_v) if post_v is not None else None
            except (ValueError, TypeError):
                post_meal_val = None
            if pre_meal_vals and post_meal_val is not None:
                pre_meal_val = sum(pre_meal_vals) / len(pre_meal_vals)
                delta_meal = post_meal_val - pre_meal_val
                paired_pre_vals.append(pre_meal_val)
                paired_post_vals.append(post_meal_val)
                paired_delta_vals.append(delta_meal)
                if delta_meal > 6:
                    high_delta_meals_this_day += 1
        daily_high_delta_counts.append(high_delta_meals_this_day)
    agi_postprandial_repeated_spike = False
    for i in range(len(daily_high_delta_counts)):
        if sum(daily_high_delta_counts[i:i + 3]) >= 2:
            agi_postprandial_repeated_spike = True
            break
    avg_pre = sum(paired_pre_vals) / len(paired_pre_vals) if paired_pre_vals else None
    avg_post = sum(paired_post_vals) / len(paired_post_vals) if paired_post_vals else None
    delta_bg = 7.0 if agi_postprandial_repeated_spike else None

    # Elevated liver: ALT or AST >= 2.5x ULN (~100)
    elevated_liver = 1.0 if (alt_val is not None and alt_val >= 100) or (ast_val is not None and ast_val >= 100) else 0.0

    # Complications
    complications = set()
    if _flag("ascvd"):
        complications.add("ASCVD")
    if _flag("heart_failure"):
        complications.add("HF")
    if _flag("decompensated_heart_failure"):
        complications.add("失代偿性HF")
    if _flag("chronic_kidney_disease"):
        complications.add("CKD")
    if _flag("fatty_liver"):
        complications.add("脂肪肝")
    if _flag("hyperlipidemia"):
        complications.add("高脂血症")
    if _flag("hypertension"):
        complications.add("高血压")
    if _flag("gastroparesis"):
        complications.add("胃轻瘫")
    if _flag("gastritis"):
        complications.add("胃炎")
    if _flag("retinopathy"):
        complications.add("眼病")
    if _flag("metabolic_syndrome"):
        complications.add("代谢综合征")
    if _flag("proteinuria"):
        complications.add("albuminuria")
    if _flag("pancreatitis") or _flag("pancreatitis_history"):
        complications.add("胰腺炎")
    if _flag("ketosis"):
        complications.add("酮症")
    if _flag("ketoacidosis"):
        complications.add("酮症酸中毒")
    if _flag("hhs"):
        complications.add("高渗高血糖综合征")
    if _flag("prostatitis"):
        complications.add("前列腺炎")
    if _flag("kidney_stone"):
        complications.add("肾结石")

    # Risk factors
    risk_factors = set()
    if _flag("smoking_history"):
        risk_factors.add("smoking")
    if _flag("dyslipidemia_flag"):
        risk_factors.add("dyslipidemia")
    if _flag("hypertension") or (systolic is not None and systolic >= 140):
        risk_factors.add("hypertension")
    if _flag("hypoglycemia"):
        risk_factors.add("hypoglycemia")
    if _flag("gi_adverse_events"):
        risk_factors.add("gastrointestinal")
    if _flag("hepatic_insufficiency") or (alt_val is not None and alt_val >= 100):
        risk_factors.add("肝功能不全")
    if _flag("renal_failure") or (egfr is not None and egfr < 30):
        risk_factors.add("肾功能不全")
    if _flag("osteoporosis"):
        risk_factors.add("骨质疏松")
    if _flag("prostatic_hyperplasia"):
        risk_factors.add("前列腺增生")
    if _flag("ketogenic_diet"):
        risk_factors.add("生酮饮食")
    if _flag("mtc") or _flag("medullary_thyroid_cancer_history"):
        risk_factors.add("甲状腺髓样癌")
    if _flag("men2"):
        risk_factors.add("多发性内分泌腺瘤病2型")

    # History
    history = set()
    if _flag("heart_failure_history"):
        history.add("心衰史")
    if _flag("fracture_history"):
        history.add("骨折史")
    if _flag("stroke_history"):
        history.add("卒中病史")
    if _flag("pancreatitis_history") or _flag("pancreatitis"):
        history.add("胰腺炎")
    if _flag("mtc") or _flag("medullary_thyroid_cancer_history"):
        history.add("甲状腺髓样癌")
    if _flag("men2"):
        history.add("多发性内分泌腺瘤病2型")

    # Hypoglycemia risk
    hypoglycemia_risk = _flag("hypoglycemia") or (egfr is not None and egfr < 45) or _flag("gi_adverse_events")

    return Patient(
        age=int(age),
        bmi=bmi,
        hba1c=hba1c,
        duration=duration,
        C_peptide_half=c_pep or 0.0,
        C_peptide_2=c_pep_2h or (c_pep or 0.0),
        C_peptide_3=0.0,
        systolic_blood_pressure=systolic or 120.0,
        diastolic_blood_pressure=diastolic or 80.0,
        Elevated_liver=elevated_liver,
        diabetes_type=2,
        delta_blood_glucose_after_meal=delta_bg,
        complications=complications,
        risk_factors=risk_factors,
        history=history,
        blood_glucose={
            "preprandial": avg_pre or 6.0,
            "postprandial": avg_post or 9.0,
        },
        family_history=set(),
        ALT=alt_val,
        AST=ast_val,
        UACR=None,
        UAER=None,
        renal_function=egfr,
        hypoglycemia_risk=hypoglycemia_risk,
    )


@app.post("/drug-recommend")
async def drug_recommend(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    _check_token(authorization)
    payload = await request.json()
    try:
        rule_patient = _build_rule_patient(payload)
        system = _get_drug_system()
        S1, S2, S3 = system.evaluate_patient(rule_patient)

        # Build rule descriptions dict
        rules_desc = {}
        for rule in system.rules:
            rules_desc[rule.rule_id] = rule.description

        def _clean_name(dr: DrugRecommendation):
            return RULE_DRUG_TO_CLEAN_NAME.get(dr.drug.value)

        def _build_trace_entry(dr: DrugRecommendation):
            clean = _clean_name(dr)
            if clean is None:
                return None
            return {
                "drug": clean,
                "trace": {
                    "recommend": sorted(list(dr.recommend_rules)),
                    "consider": sorted(list(dr.consider_rules)),
                    "caution": sorted(list(dr.caution_rules)),
                    "forbid": sorted(list(dr.forbid_rules)),
                },
            }

        from medication_rules import _drug_sort_key, _drug_sort_key_consider, _drug_sort_key_forbid

        def _collect_names(drug_set, sort_key=None):
            seen = set()
            result = []
            for dr in sorted(drug_set, key=sort_key or (lambda d: d.drug.value)):
                clean = _clean_name(dr)
                if clean and clean not in seen:
                    seen.add(clean)
                    result.append(clean)
            return result

        def _collect_traces(drug_set, sort_key=None):
            seen = set()
            result = []
            for dr in sorted(drug_set, key=sort_key or (lambda d: d.drug.value)):
                entry = _build_trace_entry(dr)
                if entry and entry["drug"] not in seen:
                    seen.add(entry["drug"])
                    result.append(entry)
            return result

        return JSONResponse({
            "ok": True,
            "S1": _collect_names(S1, _drug_sort_key),
            "S2": _collect_names(S2, _drug_sort_key_consider),
            "S3": _collect_names(S3, _drug_sort_key_forbid),
            "traces": {
                "S1": _collect_traces(S1, _drug_sort_key),
                "S2": _collect_traces(S2, _drug_sort_key_consider),
                "S3": _collect_traces(S3, _drug_sort_key_forbid),
            },
            "rules": rules_desc,
        })
    except Exception as exc:
        logging.exception("Drug recommendation failed")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


def _check_token(authorization: Optional[str]) -> None:
    token = os.getenv("INFERENCE_API_TOKEN", "").strip()
    if not token:
        return
    expected = f"Bearer {token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid authorization token")


def _validate_feature_schema(payload: Dict[str, Any]) -> None:
    version = payload.get("feature_schema_version")
    if version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"feature_schema_version mismatch: expected {FEATURE_SCHEMA_VERSION}, got {version!r}"
        )
    dim = payload.get("feature_schema_dim")
    try:
        dim_value = int(dim)
    except (TypeError, ValueError):
        raise ValueError(
            f"feature_schema_dim mismatch: expected {FEATURE_SCHEMA_DIM}, got {dim!r}"
        )
    if dim_value != FEATURE_SCHEMA_DIM:
        raise ValueError(
            f"feature_schema_dim mismatch: expected {FEATURE_SCHEMA_DIM}, got {dim_value}"
        )


def _normalize_drug_recommendation(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        names = [str(item).strip() for item in value if str(item).strip()]
        return names if names else None
    text = str(value).strip()
    return [text] if text else None


def _has_mappable_treatment_drug(records: List[Dict[str, Any]]) -> bool:
    for record in records:
        drug_name = str(record.get("drug") or "").strip()
        if drug_name and _drug_index(drug_name) is not None:
            return True
    return False




# Feature preprocessing and normalization.

import math

# Map a value to (0, 1) using a reference range.
def _sigmoid_normalize(value, low, high):
    if value is None:
        return None
    normalized = (value - low) / (high - low)
    return 1.0 / (1.0 + math.exp(-(normalized - 0.5) * 10.0))

# Classify kidney function from eGFR.
def _classify_ckd_by_egfr(egfr):
    if egfr is None or egfr <= 0:
        return 0.0
    if egfr >= 90: return 0.0
    if egfr >= 60: return 1.0
    if egfr >= 45: return 2.0
    if egfr >= 30: return 2.5
    if egfr >= 15: return 3.0
    return 4.0

# Check whether LDL is above the reference threshold.
def _check_ldl_abnormal(ldl):
    if ldl is None: return 0.0
    return 1.0 if ldl > 3.4 else 0.0

# Classify obesity from BMI.
def _classify_obesity_by_bmi(bmi):
    if bmi is None: return -1.0
    if bmi < 18.5: return 0.0
    if bmi < 24: return 1.0
    if bmi < 28: return 2.0
    return 3.0

# Check whether triglycerides are above the reference threshold.
def _check_tg_abnormal(tg):
    if tg is None: return 0.0
    return 1.0 if tg > 1.7 else 0.0

def _blood_pressure_grade(systolic: Optional[float], diastolic: Optional[float]) -> Optional[float]:
    if systolic is None or diastolic is None:
        return None
    if systolic < 120 and diastolic < 80:
        return 0.0
    if systolic < 140 and diastolic < 90:
        return 1.0
    if systolic >= 180 or diastolic >= 110:
        return 4.0
    if systolic >= 160 or diastolic >= 100:
        return 3.0
    return 2.0

def _urine_leukocyte_value(value: Any) -> Optional[float]:
    numeric = _as_float(value)
    if numeric is not None:
        return numeric
    if not isinstance(value, str):
        return None
    mapping = {
        "-": 0.0,
        "+-": 0.5,
        "±": 0.5,
        "1+": 1.0,
        "+": 1.0,
        "2+": 2.0,
        "++": 2.0,
        "3+": 3.0,
        "+++": 3.0,
    }
    return mapping.get(value.strip())

# Features that need sigmoid preprocessing before Z-score
# Maps FEATURE_NAME -> (low, high) reference range
_SIGMOID_FEATURES = {
    "甘油三酯": (0.56, 1.7),
    "收缩压": (90, 120),
    "舒张压": (60, 80),
    "heart_rate": (60, 100),
    "血氧饱和度": (95, 100),
    "呼吸": (12, 20),
    "体温": (36, 37.5),
    "首次空腹血糖": (3.9, 6.1),
    "HbA1c": (4.0, 6.5),
    "HDL-C": (1.0, 1.7),
    "LDL-C": (0, 3.4),
    "TC": (0, 5.2),
    "TG": (0.56, 1.7),
    "ALT": (0, 40),
    "AST": (0, 40),
    "UA": (208, 428),
    "SCR": (44, 106),
    "TT4": (65.0, 165.0),
    "FT4": (10.0, 25.0),
    "TSH": (0.4, 4.5),
    "β-羟丁酸": (20, 300),
    "AST/ALT": (0.8, 1.5),
    "eGFR": (1, 200),
}

# Additional field-name mappings for API payloads.
_EXTRA_FIELD_MAPPINGS = {
    "首次空腹血糖": "fasting_glucose",
    "甘油三酯": "TG",
}

_ZERO_IS_MISSING_FEATURES = {
    "age",
    "height",
    "weight",
    "BMI",
    "HbA1c",
    "HDL-C",
    "LDL-C",
    "TC",
    "TG",
    "ALT",
    "AST",
    "UA",
    "SCR",
    "TT4",
    "FT4",
    "TSH",
    "β-羟丁酸",
    "AST/ALT",
    "eGFR",
    "收缩压",
    "舒张压",
    "heart_rate",
    "血氧饱和度",
    "呼吸",
    "体温",
}

# Z-score normalization stats for the clinical top18 checkpoint.
# Order matches FEATURE_NAMES exactly (77 features).
_PERSON_MEAN = [
    0.573759, 0.580055, 162.966960, 65.196831, 24.609247, 1.681864,
    0.329721, 0.250387, 0.012756, 0.020101, 0.035562, 0.039041,
    0.009664, 0.000000, 0.001933, 0.009277, 0.000000, 0.044840,
    0.004639, 0.949621, 0.759157, 0.605964, 0.596664, 0.936195,
    0.391050, 1.336969, 0.995324, 0.259730, 0.796474, 0.920923,
    0.768769, 0.545005, 0.477445, 0.467962, 0.326650, 0.207427,
    0.067536, 0.234052, 0.261690, 0.521832, 0.251671, 0.483061,
    0.057596, 0.127174, 0.768844, 0.364128, 0.129880, 0.214534,
    0.003479, 0.000773, 0.028218, 0.000387, 0.000000, 0.085427,
    0.054117, 0.299575, 0.367221, 0.009277, 0.001933, 0.098956,
    0.006958, 0.004252, 0.132973, 0.000387, 0.217627, 0.076923,
    0.101662, 0.013529, 0.000000, 0.000000, 0.000000, 1.463085,
    0.035949, 0.400153, 0.243440, 1.627778, 0.000000,
]

_PERSON_STD = [
    0.136680, 0.493550, 8.812213, 12.891615, 8.374475, 1.061105,
    0.470112, 0.433236, 0.112220, 0.140344, 0.185196, 0.193693,
    0.097828, 1.000000, 0.043920, 0.095870, 1.000000, 0.206952,
    0.067949, 0.169690, 0.360764, 0.362618, 0.286557, 0.062336,
    0.293825, 0.862481, 0.045187, 0.367670, 0.271128, 0.134275,
    0.361904, 0.357190, 0.313662, 0.406079, 0.368277, 0.243431,
    0.116756, 0.320889, 0.384247, 0.945224, 0.363743, 0.220172,
    0.232977, 0.333168, 0.421572, 0.989506, 0.575347, 0.410499,
    0.058880, 0.027794, 0.165595, 0.019657, 1.000000, 0.279516,
    0.226248, 0.458072, 0.482047, 0.095870, 0.043920, 0.298603,
    0.083123, 0.065069, 0.339545, 0.019657, 0.412632, 0.266469,
    0.302204, 0.115526, 1.000000, 1.000000, 1.000000, 1.032001,
    0.186163, 0.761541, 0.429159, 0.776351, 1.000000,
]

# C-peptide Z-score stats (from training data)
_CPEP_MEAN = [468.3352, 1406.1772]
_CPEP_STD = [384.1784, 984.0095]


def _records_for_stage(records: Any, stage: int) -> List[Dict[str, Any]]:
    if not isinstance(records, list):
        return []
    filtered = []
    for record in records:
        if not isinstance(record, dict):
            continue
        record_type = record.get("type")
        if record_type is None or int(record_type) == stage:
            filtered.append(record)
    return sorted(filtered, key=lambda item: str(item.get("record_date") or ""))


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fr:
            data = json.load(fr)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logging.warning("Failed to load patient summary source %s: %s", path, exc)
        return {}


def _patient_summary_sources() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    global _PATIENT_SUMMARY_CACHE
    if _PATIENT_SUMMARY_CACHE is not None:
        return _PATIENT_SUMMARY_CACHE

    patient_data: Dict[str, Any] = {}
    for filename in PATIENT_SUMMARY_SOURCE_FILES:
        patient_data.update(_load_json_dict(PROJECT_DIR / filename))
    extra_data = _load_json_dict(PROJECT_DIR / PATIENT_SUMMARY_EXTRA_FILE)
    _PATIENT_SUMMARY_CACHE = (patient_data, extra_data)
    logging.info(
        "Loaded patient summary sources: patients=%d, extra=%d",
        len(patient_data),
        len(extra_data),
    )
    return _PATIENT_SUMMARY_CACHE


def _summary_leaf_value(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("value", "检验项值", "测量值", "result", "结果"):
            if key in value:
                return _summary_leaf_value(value.get(key))
        return None
    if isinstance(value, list):
        for item in value:
            leaf = _summary_leaf_value(item)
            if not _summary_value_missing(leaf):
                return leaf
        return None
    return value


def _summary_value_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        clean = value.strip().lower()
        return clean in {"", "-", "--", "none", "null", "nan", "no", "未查", "未检测", "未见"}
    return False


def _raw_section(raw: Dict[str, Any], names: Iterable[str]) -> Dict[str, Any]:
    for name in names:
        section = raw.get(name)
        if isinstance(section, dict):
            return section
    return {}


def _raw_summary_value(raw: Dict[str, Any], key: str, extra: Optional[Dict[str, Any]] = None) -> Any:
    extra = extra or {}
    direct = _summary_leaf_value(raw.get(key))
    if not _summary_value_missing(direct):
        return direct

    course = _raw_section(raw, ("首次病程", "入院指标", "生命体征"))
    course_value = _summary_leaf_value(course.get(key))
    if not _summary_value_missing(course_value):
        return course_value

    lab = _raw_section(raw, ("检验", "检验指标"))
    lab_value = _summary_leaf_value(lab.get(key))
    if not _summary_value_missing(lab_value):
        return lab_value

    extra_value = _summary_leaf_value(extra.get(key))
    if not _summary_value_missing(extra_value):
        return extra_value
    return None


def _summary_number(value: Any) -> Optional[float]:
    return _as_float(_summary_leaf_value(value))


def _format_summary_value(value: Any) -> str:
    value = _summary_leaf_value(value)
    if _summary_value_missing(value):
        return "-"
    numeric = _summary_number(value)
    if numeric is not None:
        if abs(numeric - round(numeric)) < 1e-9:
            return str(int(round(numeric)))
        return f"{numeric:.3f}".rstrip("0").rstrip(".")
    return str(value).strip()


def _summary_item(label: str, value: Any) -> Optional[Dict[str, str]]:
    if _summary_value_missing(value):
        return None
    return {"label": label, "value": _format_summary_value(value)}


def _positive_summary_value(value: Any, allow_nonzero_numeric: bool = False, allow_positive_text: bool = False) -> bool:
    value = _summary_leaf_value(value)
    if _summary_value_missing(value):
        return False
    numeric = _summary_number(value)
    if numeric is not None:
        if allow_nonzero_numeric:
            return numeric > 0
        return abs(numeric - 1.0) < 1e-9
    if isinstance(value, str):
        clean = value.strip().lower()
        if clean in {"yes", "true", "1", "有", "是", "阳性", "positive"}:
            return True
        if clean in {"no", "false", "0", "无", "否", "阴性", "manualcheck", "none", "null", "nan"}:
            return False
        return allow_positive_text
    return bool(value)


def _lab_summary_value(raw: Dict[str, Any], extra: Dict[str, Any], key: str) -> Any:
    if key == "尿白细胞":
        return extra.get("尿白细胞")
    if key == "AST/ALT":
        direct = _raw_summary_value(raw, key, extra)
        if not _summary_value_missing(direct):
            return direct
        ast = _summary_number(_raw_summary_value(raw, "AST", extra))
        alt = _summary_number(_raw_summary_value(raw, "ALT", extra))
        if ast is not None and alt not in (None, 0):
            return ast / alt
        return None
    if key == "eGFR":
        direct = _raw_summary_value(raw, key, extra)
        if not _summary_value_missing(direct):
            return direct
        scr = _summary_number(_raw_summary_value(raw, "SCR", extra))
        age = _summary_number(_raw_summary_value(raw, "age", extra))
        gender = _raw_summary_value(raw, "gender", extra)
        return _calculate_egfr_2021(scr, age, _summary_is_male(gender))
    return _raw_summary_value(raw, key, extra)


def _summary_is_male(gender: Any) -> Optional[bool]:
    if gender is None:
        return None
    numeric = _summary_number(gender)
    if numeric is not None:
        return numeric > 0
    if isinstance(gender, str):
        clean = gender.strip().lower()
        if clean in {"男", "male", "m", "1", "true", "yes"}:
            return True
        if clean in {"女", "female", "f", "0", "false", "no"}:
            return False
    return None


def _calculate_egfr_2021(scr: Optional[float], age: Optional[float], is_male: Optional[bool]) -> Optional[float]:
    if scr is None or age is None or is_male is None or scr <= 0 or age <= 0:
        return None
    scr_mg_dl = scr / 88.4
    kappa = 0.9 if is_male else 0.7
    alpha = -0.302 if is_male else -0.241
    sex_factor = 1.0 if is_male else 1.012
    egfr = 142 * min(scr_mg_dl / kappa, 1) ** alpha * max(scr_mg_dl / kappa, 1) ** -1.200 * 0.9938 ** age * sex_factor
    return max(0.0, min(float(egfr), 200.0))


def _indicator_summary_items(raw: Dict[str, Any], extra: Dict[str, Any]) -> List[Dict[str, str]]:
    ldl = _summary_number(_raw_summary_value(raw, "LDL-C", extra))
    tg = _summary_number(_raw_summary_value(raw, "TG", extra))
    bmi = _summary_number(_raw_summary_value(raw, "BMI", extra))
    egfr = _summary_number(_lab_summary_value(raw, extra, "eGFR"))
    systolic = _summary_number(_raw_summary_value(raw, "收缩压", extra))
    diastolic = _summary_number(_raw_summary_value(raw, "舒张压", extra))

    values = {
        "CKD-N期": _classify_ckd_by_egfr(egfr) if egfr is not None else None,
        "血脂异常": _check_ldl_abnormal(ldl) if ldl is not None else None,
        "肥胖情况": _classify_obesity_by_bmi(bmi) if bmi is not None else None,
        "脂质代谢异常标志": _check_tg_abnormal(tg) if tg is not None else None,
        "高血压分级": _blood_pressure_grade(systolic, diastolic),
    }
    items: List[Dict[str, str]] = []
    for key, label in SUMMARY_INDICATOR_FIELDS:
        item = _summary_item(label, values.get(key))
        if item:
            items.append(item)
    return items


def _patient_summary(check_id: Any, stage: int) -> Dict[str, Any]:
    patient_data, extra_data = _patient_summary_sources()
    raw = patient_data.get(str(check_id))
    if not isinstance(raw, dict):
        return {"sections": [], "text": ""}
    extra = extra_data.get(str(check_id))
    if not isinstance(extra, dict):
        extra = {}

    sections: List[Dict[str, Any]] = []

    def add_section(title: str, items: List[Dict[str, str]]) -> None:
        if items:
            sections.append({"title": title, "items": items})

    if stage == 2:
        c_pep_items = []
        for key, label in SUMMARY_C_PEP_FIELDS:
            item = _summary_item(label, _raw_summary_value(raw, key, extra))
            if item:
                c_pep_items.append(item)
        add_section("C肽", c_pep_items)

    basic_items = []
    for key, label in SUMMARY_BASIC_FIELDS:
        item = _summary_item(label, _raw_summary_value(raw, key, extra))
        if item:
            basic_items.append(item)
    add_section("基础信息", basic_items)

    lab_items = []
    for key, label in SUMMARY_LAB_FIELDS:
        item = _summary_item(label, _lab_summary_value(raw, extra, key))
        if item:
            lab_items.append(item)
    add_section("检验指标", lab_items)

    history_items = []
    lifestyle = _raw_section(raw, ("lifestyle_factors",))
    personal_history = _raw_section(raw, ("personal_history",))
    for key, label in SUMMARY_HISTORY_FIELDS:
        value = lifestyle.get(key) if key in lifestyle else personal_history.get(key)
        if key == "肾积水":
            value = extra.get("肾积水")
        if _positive_summary_value(value):
            history_items.append({"label": label, "value": "1"})
    add_section("病史特征", history_items)

    diagnosis_items = []
    diagnoses = _raw_section(raw, ("诊断",))
    for key, label in SUMMARY_DIAGNOSIS_FIELDS:
        if key == "周围血管病分级":
            value = extra.get("糖尿病周围血管病")
            if _positive_summary_value(value, allow_nonzero_numeric=True):
                diagnosis_items.append({"label": label, "value": _format_summary_value(value)})
            continue
        value = diagnoses.get(key)
        if _positive_summary_value(value, allow_nonzero_numeric=True, allow_positive_text=True):
            diagnosis_items.append({"label": label, "value": "1" if str(value).strip().lower() == "yes" else _format_summary_value(value)})
    add_section("诊断特征", diagnosis_items)

    vital_items = []
    for key, label in SUMMARY_VITAL_FIELDS:
        item = _summary_item(label, _raw_summary_value(raw, key, extra))
        if item:
            vital_items.append(item)
    add_section("生命体征", vital_items)

    add_section("指标判定特征", _indicator_summary_items(raw, extra))

    text_parts = []
    for section in sections:
        pairs = [f"{item['label']}={item['value']}" for item in section.get("items", [])]
        if pairs:
            text_parts.append(f"{section.get('title')}: " + ", ".join(pairs))
    return {"sections": sections, "text": "；".join(text_parts)}


def _first_present(data: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _zero_as_missing(feature: str, value: Optional[float]) -> Optional[float]:
    if value is not None and feature in _ZERO_IS_MISSING_FEATURES and abs(value) <= 1e-12:
        return None
    return value


def _as_positive_float(value: Any) -> Optional[float]:
    parsed = _as_float(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _duration_to_log_years(value: Any) -> Optional[float]:
    numeric = _as_float(value)
    if numeric is None:
        return None
    return math.log1p(max(0.0, numeric))


def _preprocess_person_feature(feature: str, value: float) -> Optional[float]:
    if value is None:
        return None
    if feature == "age":
        return value / 100.0 if abs(value) > 1.5 else value
    if feature == "duration_of_diabetes":
        return _duration_to_log_years(value)
    if feature in _SIGMOID_FEATURES:
        low, high = _SIGMOID_FEATURES[feature]
        return _sigmoid_normalize(value, low, high)
    return value


def _feature_value(patient: Dict[str, Any], feature: str) -> Optional[float]:
    if feature == "gender":
        raw = _first_present(patient, ("gender",))
        numeric = _as_float(raw)
        if numeric is not None:
            return 1.0 if numeric != 0 else 0.0
        if isinstance(raw, str):
            clean = raw.strip().lower()
            if clean in ("男", "male", "m", "1", "true", "yes"):
                return 1.0
            if clean in ("女", "female", "f", "0", "false", "no"):
                return 0.0
        return None

    # Extra field name mappings for Java payload
    if feature in _EXTRA_FIELD_MAPPINGS:
        val = _as_float(patient.get(_EXTRA_FIELD_MAPPINGS[feature]))
        if val is not None:
            return _zero_as_missing(feature, val)

    if feature == "尿白细胞":
        return _urine_leukocyte_value(_first_present(patient, PATIENT_KEY_ALIASES.get(feature, (feature,))))
    if feature == "高血压分级":
        direct = _as_float(_first_present(patient, PATIENT_KEY_ALIASES.get(feature, (feature,))))
        if direct is not None:
            return direct
        systolic = _as_float(_first_present(patient, ("systolic_pressure", "收缩压")))
        diastolic = _as_float(_first_present(patient, ("diastolic_pressure", "舒张压")))
        return _blood_pressure_grade(systolic, diastolic)

    aliases = PATIENT_KEY_ALIASES.get(feature, (feature,))
    direct = _as_float(_first_present(patient, aliases))
    if direct is not None:
        return _zero_as_missing(feature, direct)

    if feature == "BMI":
        height = _as_float(patient.get("height"))
        weight = _as_float(patient.get("weight"))
        if height and weight:
            height_m = height / 100.0
            if height_m > 0:
                return weight / (height_m * height_m)
    if feature == "AST/ALT":
        ast = _as_float(patient.get("AST"))
        alt = _as_float(patient.get("ALT"))
        if ast is not None and alt and alt != 0:
            return ast / alt

    mapped = {
        "心血管病史": "cardiovascular_history",
        "肝炎史": "hepatitis_history",
        "心衰史": "heart_failure_history",
        "低血糖史": "hypoglycemia",
        "胆囊切除术": "cholecystectomy",
        "肺结核史": "tuberculosis_history",
        "收缩压": "systolic_pressure",
        "舒张压": "diastolic_pressure",
        "骨折史": "fracture_history",
        "卒中史": "stroke_history",
        "胰腺炎史": "pancreatitis_history",
        "甲状腺髓样癌史": "medullary_thyroid_cancer_history",
        "生酮饮食": "ketogenic_diet",
        "血氧饱和度": "spo2",
        "呼吸": "respiratory_rate",
        "体温": "temperature",
        "complications-糖尿病性周围神经病变": "diabetic_peripheral_neuropathy",
        "complications-糖尿病性视网膜病变": "retinopathy",
        "complications-糖尿病性周围血管病变": "diabetic_peripheral_vascular_disease",
        "complications-糖尿病性肾病": "diabetic_nephropathy_risk",
        "complications-高血压": "hypertension",
        "complications-冠状动脉粥样硬化性心脏病": "ascvd",
        "complications-心力衰竭": "heart_failure",
        "complications-失代偿性心力衰竭": "decompensated_heart_failure",
        "complications-慢性肾脏病": "chronic_kidney_disease",
        "complications-肾衰竭": "renal_failure",
        "complications-肾透析": "dialysis",
        "complications-肾结石": "kidney_stone",
        "complications-蛋白尿": "proteinuria",
        "complications-高脂血症": "hyperlipidemia",
        "complications-脂肪肝": "fatty_liver",
        "complications-代谢综合征": "metabolic_syndrome",
        "complications-胃轻瘫": "gastroparesis",
        "complications-胃炎": "gastritis",
        "complications-胰腺炎": "pancreatitis",
        "complications-胃肠道不良反应": "gi_adverse_events",
        "complications-前列腺增生": "prostatic_hyperplasia",
        "complications-前列腺炎": "prostatitis",
        "complications-骨质疏松症": "osteoporosis",
        "complications-肝功能不全": "hepatic_insufficiency",
        "complications-高渗高血糖综合征": "hhs",
        "complications-甲状腺髓样癌": "mtc",
        "complications-多发性内分泌腺瘤病2型": "men2",
        "CKD-N期": "diabetic_nephropathy_stage",
        "血脂异常": "dyslipidemia_flag",
        "肥胖情况": "obesity_grade",
        "脂质代谢异常标志": "lipid_metabolism_abnormal",
    }
    if feature in mapped:
        value = _as_float(patient.get(mapped[feature]))
        if value is not None:
            return _zero_as_missing(feature, value)
        if feature == "complications-糖尿病性肾病":
            return _as_float(patient.get("diabetic_nephropathy_stage"))
        return None

    if feature == "complications-酮症":
        ketosis = _as_float(patient.get("ketosis"))
        if ketosis is not None:
            return _zero_as_missing(feature, ketosis)
    if feature == "complications-酮症酸中毒":
        ketoacidosis = _as_float(patient.get("ketoacidosis"))
        if ketoacidosis is not None:
            return _zero_as_missing(feature, ketoacidosis)

    return None


def _person_tensor(patient: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    values: List[float] = []
    masks: List[float] = []
    for idx, feature in enumerate(FEATURE_NAMES):
        value = _feature_value(patient, feature)
        value = _preprocess_person_feature(feature, value)
        if value is None:
            values.append(0.0)
            masks.append(0.0)
        else:
            mean = _PERSON_MEAN[idx] if idx < len(_PERSON_MEAN) else 0.0
            std = _PERSON_STD[idx] if idx < len(_PERSON_STD) and _PERSON_STD[idx] else 1.0
            values.append((float(value) - mean) / std)
            masks.append(1.0)
    return (
        torch.tensor(values, dtype=torch.float32),
        torch.tensor(masks, dtype=torch.float32),
    )


def _matrix_with_mask(records: List[Dict[str, Any]], fields: List[str], limit: int) -> Tuple[torch.Tensor, torch.Tensor]:
    values: List[List[float]] = []
    masks: List[List[float]] = []
    for record in records[:limit]:
        row: List[float] = []
        mask_row: List[float] = []
        for field in fields:
            value = _as_positive_float(record.get(field))
            if value is None:
                row.append(0.0)
                mask_row.append(0.0)
            else:
                row.append(value)
                mask_row.append(1.0)
        values.append(row)
        masks.append(mask_row)
    while len(values) < limit:
        values.append([0.0] * len(fields))
        masks.append([0.0] * len(fields))
    return torch.tensor(values, dtype=torch.float32), torch.tensor(masks, dtype=torch.float32)


def _has_observed_bg(records: List[Dict[str, Any]]) -> bool:
    for record in records:
        if not isinstance(record, dict):
            continue
        for field in BG_FIELDS:
            if _as_positive_float(record.get(field)) is not None:
                return True
    return False


def _sum_fields(record: Dict[str, Any], fields: Iterable[str]) -> Tuple[float, float]:
    total = 0.0
    mask = 0.0
    for field in fields:
        value = _as_float(record.get(field))
        if value is not None:
            total += value
            mask = 1.0
    return total, mask


def _insulin_matrix(records: List[Dict[str, Any]], limit: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build insulin tensor matching the deployed checkpoint format."""
    values: List[List[float]] = []
    masks: List[List[float]] = []
    for record in records[:limit]:
        if D_INSULIN == 5:
            basic_morning = _as_float(record.get("subcutaneous_morning")) or 0.0
            basic_noon = _as_float(record.get("subcutaneous_noon")) or 0.0
            basic_evening = _as_float(record.get("subcutaneous_evening")) or 0.0
            basic_night = _as_float(record.get("subcutaneous_before_sleep")) or 0.0
            premix_morning_long = _as_float(record.get("subcutaneous_premix_morning")) or 0.0
            premix_morning_short = _as_float(record.get("subcutaneous_premix_morning_short")) or 0.0
            premix_evening_long = _as_float(record.get("subcutaneous_premix_evening")) or 0.0
            premix_evening_short = _as_float(record.get("subcutaneous_premix_evening_short")) or 0.0
            premix_sum = premix_morning_long + premix_morning_short + premix_evening_long + premix_evening_short
            basic_sum = basic_morning + basic_noon + basic_evening + basic_night
            flag_premix = 1.0 if premix_sum > basic_sum and premix_sum > 0 else 0.0
            row = [
                flag_premix,
                basic_morning + premix_morning_short,
                basic_noon,
                basic_evening + premix_evening_short,
                basic_night + premix_morning_long + premix_evening_long,
            ]
            mask_row = [1.0 if abs(value) > 1e-12 else 0.0 for value in row]
            values.append(row)
            masks.append(mask_row)
            continue

        row = [0.0] * D_INSULIN
        mask_row = [0.0] * D_INSULIN

        def _set(idx, field):
            v = _as_float(record.get(field))
            if v is not None:
                row[idx] = v
                mask_row[idx] = 1.0

        # basic SC
        _set(0, "subcutaneous_morning")
        _set(1, "subcutaneous_noon")
        _set(2, "subcutaneous_evening")
        _set(3, "subcutaneous_before_sleep")
        # premix SC: map premix_morning → dim 4 (long), dim 5 (short stays 0)
        _set(4, "subcutaneous_premix_morning")
        _set(5, "subcutaneous_premix_morning_short")
        # premix_evening → dim 6 (long), dim 7 (short stays 0)
        _set(6, "subcutaneous_premix_evening")
        _set(7, "subcutaneous_premix_evening_short")

        values.append(row)
        masks.append(mask_row)

    while len(values) < limit:
        values.append([0.0] * D_INSULIN)
        masks.append([0.0] * D_INSULIN)
    return torch.tensor(values, dtype=torch.float32), torch.tensor(masks, dtype=torch.float32)


def _c_pep_tensor(patient: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (2,) tensors: [fasting_c_peptide, 2h_c_peptide] — Z-score normalized"""
    fasting = _as_positive_float(_first_present(patient, ("C_pep", "c_pep")))
    post_2h = _as_positive_float(_first_present(patient, ("C_pep_2h", "c_pep_2h")))

    # Apply Z-score normalization (matches training pipeline)
    norm_fasting = (fasting - _CPEP_MEAN[0]) / _CPEP_STD[0] if fasting is not None else 0.0
    norm_post_2h = (post_2h - _CPEP_MEAN[1]) / _CPEP_STD[1] if post_2h is not None else 0.0

    values = [norm_fasting, norm_post_2h]
    masks = [1.0 if fasting is not None else 0.0, 1.0 if post_2h is not None else 0.0]
    return torch.tensor(values, dtype=torch.float32), torch.tensor(masks, dtype=torch.float32)


def _drug_index(drug_name: str) -> Optional[int]:
    if not drug_name:
        return None
    name = drug_name.strip()
    if name in DRUG_NAME_TO_INDEX:
        return DRUG_NAME_TO_INDEX[name]
    for known_name, idx in DRUG_NAME_TO_INDEX.items():
        if known_name in name or name in known_name:
            return idx
    return None


def _drug_tensor(treatment_records: List[Dict[str, Any]], limit: int) -> torch.Tensor:
    from collections import defaultdict
    days: Dict[str, List[str]] = defaultdict(list)
    for record in treatment_records:
        drug_name = str(record.get("drug") or "").strip()
        if not drug_name:
            continue
        date_str = str(record.get("record_date") or "")[:10]
        if not date_str:
            continue
        days[date_str].append(drug_name)

    sorted_dates = sorted(days.keys())[:limit]
    out_rows: List[torch.Tensor] = []

    for date_str in sorted_dates:
        vec_23 = torch.zeros(24)
        for drug_name in days[date_str]:
            idx = _drug_index(drug_name)
            if idx is not None:
                vec_23[idx] = 1.0

        if DRUG_MODE == 1:
            out = vec_23
        elif DRUG_MODE == 2:
            out = torch.tensor([
                vec_23[0],
                vec_23[1:5].max(),
                vec_23[5],
                vec_23[6],
                vec_23[7],
                vec_23[8],
                vec_23[9],
                vec_23[10],
                vec_23[11],
                vec_23[12],
                vec_23[13],
                vec_23[14],
                vec_23[15],
                vec_23[16],
                vec_23[17:21].max(),
                vec_23[21],
                vec_23[22:24].max(),
            ])
        else:
            out = torch.cat([
                vec_23[0:1],
                vec_23[1:5].max(dim=0, keepdim=True).values,
                vec_23[5:7],
                vec_23[7:12].max(dim=0, keepdim=True).values,
                vec_23[12:13],
                vec_23[13:14],
                vec_23[14:17].max(dim=0, keepdim=True).values,
                vec_23[17:21].max(dim=0, keepdim=True).values,
                vec_23[21:22],
                vec_23[22:24].max(dim=0, keepdim=True).values,
            ])
        out_rows.append(out)

    if not out_rows:
        return torch.zeros((limit, D_DRUG), dtype=torch.float32)

    while len(out_rows) < limit:
        out_rows.append(torch.zeros(D_DRUG))

    return torch.stack(out_rows, dim=0)[:limit]


def _drug_tensor_from_recommendation(drug_names: List[str], limit: int) -> torch.Tensor:
    """Build (limit, D_DRUG) drug tensor from rule engine S1 drug names.

    Maps drug names through DRUG_NAME_TO_INDEX (24-dim) then DRUG_MODE=2 compression to 17-dim.
    The same drug plan repeats for all `limit` days.
    """
    vec_23 = torch.zeros(24)
    for name in drug_names:
        idx = _drug_index(name)
        if idx is not None:
            vec_23[idx] = 1.0

    if DRUG_MODE == 1:
        out = vec_23
    elif DRUG_MODE == 2:
        out = torch.tensor([
            vec_23[0],
            vec_23[1:5].max(),
            vec_23[5],
            vec_23[6],
            vec_23[7],
            vec_23[8],
            vec_23[9],
            vec_23[10],
            vec_23[11],
            vec_23[12],
            vec_23[13],
            vec_23[14],
            vec_23[15],
            vec_23[16],
            vec_23[17:21].max(),
            vec_23[21],
            vec_23[22:24].max(),
        ])
    else:
        out = torch.cat([
            vec_23[0:1],
            vec_23[1:5].max(dim=0, keepdim=True).values,
            vec_23[5:7],
            vec_23[7:12].max(dim=0, keepdim=True).values,
            vec_23[12:13],
            vec_23[13:14],
            vec_23[14:17].max(dim=0, keepdim=True).values,
            vec_23[17:21].max(dim=0, keepdim=True).values,
            vec_23[21:22],
            vec_23[22:24].max(dim=0, keepdim=True).values,
        ])
    return out.unsqueeze(0).expand(limit, -1)


def _normalize_drug2(value: Any, limit: int) -> Optional[torch.Tensor]:
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except Exception as exc:
        raise ValueError(f"drug2 must be numeric: {exc}") from exc

    if tensor.ndim == 1:
        if tensor.shape[0] != D_DRUG:
            raise ValueError(f"drug2 vector dim mismatch: expected {D_DRUG}, got {tensor.shape[0]}")
        return tensor.unsqueeze(0).expand(limit, -1).clone()

    if tensor.ndim == 2:
        if tensor.shape[1] != D_DRUG:
            raise ValueError(f"drug2 feature dim mismatch: expected {D_DRUG}, got {tensor.shape[1]}")
        if tensor.shape[0] == 1:
            return tensor.expand(limit, -1).clone()
        if tensor.shape[0] != limit:
            raise ValueError(f"drug2 sequence length must be 1 or {limit}, got {tensor.shape[0]}")
        return tensor.clone()

    raise ValueError(f"drug2 must be 1D or 2D, got shape {tuple(tensor.shape)}")


def _build_sample(
    check_id: str,
    patient: Dict[str, Any],
    bg_records: List[Dict[str, Any]],
    treatment_records: List[Dict[str, Any]],
    cfg: Any,
    drug_recommendation: Optional[List[str]] = None,
    stage: int = 2,
    drug2_override: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    person_value, person_mask = _person_tensor(patient)
    c_pep_value, c_pep_mask = _c_pep_tensor(patient)
    # Merge c-peptide into person features to match stage2 checkpoint (d_person=79)
    person_value = torch.cat([person_value, c_pep_value], dim=0)
    person_mask = torch.cat([person_mask, c_pep_mask], dim=0)
    bg, bg_mask = _matrix_with_mask(bg_records, BG_FIELDS, PREDICTION_HORIZON)
    insulin, insulin_mask = _insulin_matrix(treatment_records, PREDICTION_HORIZON)
    zero_bg = torch.zeros((PREDICTION_HORIZON, D_BG), dtype=torch.float32)
    zero_insulin = torch.zeros((PREDICTION_HORIZON, D_INSULIN), dtype=torch.float32)
    if stage == 2:
        bg1 = zero_bg.clone()
        bg1_mask = torch.zeros_like(zero_bg)
        bg2 = bg
        bg2_mask = bg_mask
    else:
        bg1 = bg
        bg1_mask = bg_mask
        bg2 = zero_bg.clone()
        bg2_mask = torch.zeros_like(zero_bg)
    drug_dim = int(getattr(cfg, "d_drug", 0) or D_DRUG)
    if drug_dim <= 0:
        drug_dim = D_DRUG

    # drug2 IS read at every predicted day in the inference loop, so this
    # tensor's values matter. Prefer rule-engine recommendation when provided;
    # otherwise fall back to one-hot from historical treatment records.
    if drug2_override is not None:
        drug2 = drug2_override
    elif drug_recommendation and len(drug_recommendation) > 0:
        drug2 = _drug_tensor_from_recommendation(drug_recommendation, PREDICTION_HORIZON)
    else:
        drug2 = _drug_tensor(treatment_records, PREDICTION_HORIZON)

    return {
        "check_id": check_id,
        "person_value": person_value,
        "person_mask": person_mask,
        "bg1": bg1,
        "bg1_mask": bg1_mask,
        "insulin1": insulin,
        "insulin1_mask": insulin_mask,
        "len1": PREDICTION_HORIZON,
        "bg2": bg2,
        "bg2_mask": bg2_mask,
        "insulin2": zero_insulin.clone(),
        "insulin2_mask": torch.zeros_like(zero_insulin),
        "drug2": drug2,
        "len2": PREDICTION_HORIZON,
        "real_lengths": torch.tensor(PREDICTION_HORIZON, dtype=torch.long),
        "d_insulin": D_INSULIN,
        "d_per1": D_PERSON,
        "d_drug": drug_dim,
        "data_mode": "test",
    }
