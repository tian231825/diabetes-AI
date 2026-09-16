# -*- encoding: utf-8 -*-
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from configs.Config import opt_config
from data.dataloader2 import DiabetesDataset, collate_fn
from data.MedicalDBPreprocessor import MedicalDBPreprocessor
from models.model import TwoStageModel
from utils.logger import set_seed
from utils.loss import CombinedLoss
from train import compute_batch_loss
import os
from PIL import Image, ImageDraw, ImageFont


def bootstrap_mean_ci(values, n_boot=2000, ci=95, seed=42):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        v = float(values[0])
        return v, v

    rng = np.random.default_rng(seed)
    n = values.size
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (100 - ci) / 2.0
    return (
        float(np.percentile(boot_means, alpha)),
        float(np.percentile(boot_means, 100 - alpha)),
    )


def format_ci(low, high):
    if not np.isfinite(low) or not np.isfinite(high):
        return "[nan, nan]"
    return f"[{low:.4f}, {high:.4f}]"


def configure_visualization_fonts():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    candidate_font_files = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf",
    ]
    candidate_font_names = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ]

    selected_font = None
    for font_file in candidate_font_files:
        if os.path.exists(font_file):
            try:
                fm.fontManager.addfont(font_file)
                selected_font = fm.FontProperties(fname=font_file).get_name()
                break
            except Exception:
                pass

    if selected_font is None:
        available_fonts = {f.name for f in fm.fontManager.ttflist}
        for font_name in candidate_font_names:
            if font_name in available_fonts:
                selected_font = font_name
                break

    if selected_font:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [selected_font]
    plt.rcParams["axes.unicode_minus"] = False
    return selected_font


CARD_PAGE_BG = (250, 248, 243)
CARD_HEADER_BG = (222, 233, 244)
CARD_SECTION_BG = (234, 240, 230)
CARD_GRID = (170, 178, 184)
CARD_TEXT = (32, 39, 45)
CARD_MUTED = (98, 109, 118)
CARD_ALT_BG = (245, 248, 251)
CARD_WHITE = (255, 255, 255)


def _load_card_font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _measure_multiline(draw, text, font, spacing):
    if not text:
        return font.size
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    return bbox[3] - bbox[1]


def _draw_card_table(draw, left, top, title, headers, rows, cell_width, header_height, row_height, title_font, header_font, body_font, note=""):
    table_width = cell_width * len(headers)
    draw.rounded_rectangle((left, top, left + table_width, top + 42), radius=10, fill=CARD_SECTION_BG)
    draw.text((left + 14, top + 9), title, font=title_font, fill=CARD_TEXT)
    y = top + 54
    if note:
        draw.text((left + 2, y), note, font=body_font, fill=CARD_MUTED)
        y += 28

    for col_idx, header in enumerate(headers):
        x0 = left + col_idx * cell_width
        x1 = x0 + cell_width
        draw.rectangle((x0, y, x1, y + header_height), fill=CARD_HEADER_BG, outline=CARD_GRID, width=1)
        bbox = draw.textbbox((0, 0), header, font=header_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text((x0 + (cell_width - text_w) / 2, y + (header_height - text_h) / 2 - 1), header, font=header_font, fill=CARD_TEXT)
    y += header_height

    spacing = 4
    for row_idx, row in enumerate(rows):
        bg = CARD_WHITE if row_idx % 2 == 0 else CARD_ALT_BG
        actual_row_height = row_height
        for col_idx, cell in enumerate(row):
            if col_idx == 0:
                continue
            actual_row_height = max(actual_row_height, _measure_multiline(draw, cell, body_font, spacing) + 20)
        for col_idx, cell in enumerate(row):
            x0 = left + col_idx * cell_width
            x1 = x0 + cell_width
            draw.rectangle((x0, y, x1, y + actual_row_height), fill=bg, outline=CARD_GRID, width=1)
            font = header_font if col_idx == 0 else body_font
            fill = CARD_TEXT if col_idx == 0 else CARD_MUTED
            if col_idx == 0:
                bbox = draw.textbbox((0, 0), cell, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                draw.text((x0 + (cell_width - text_w) / 2, y + (actual_row_height - text_h) / 2 - 1), cell, font=font, fill=fill)
            else:
                draw.multiline_text((x0 + 10, y + 10), cell, font=font, fill=fill, spacing=spacing, align="left")
        y += actual_row_height
    return y


def _format_pair_cell(gt_value, pred_value):
    return f"{float(gt_value):.1f}\n{float(pred_value):.1f}"


def _build_pair_rows(target_array, pred_array, row_labels):
    day_count = min(target_array.shape[0], pred_array.shape[0])
    headers = ["Day"] + row_labels[:target_array.shape[1]]
    rows = []
    for day_idx in range(day_count):
        row = [f"D{day_idx + 1}"]
        for col_idx in range(target_array.shape[1]):
            row.append(_format_pair_cell(target_array[day_idx, col_idx], pred_array[day_idx, col_idx]))
        rows.append(row)
    return headers, rows


def save_prediction_case_card(sample, predictions, save_path):
    target_bg = sample["bg1"].numpy()
    target_insulin = sample["insulin1"].numpy()
    pred_bg = predictions["bg_prediction"][0]
    pred_insulin = predictions["insulin_prediction"][0]

    bg_days = min(target_bg.shape[0], pred_bg.shape[0])
    insulin_days = min(target_insulin.shape[0], pred_insulin.shape[0])
    target_bg = target_bg[:bg_days]
    pred_bg = pred_bg[:bg_days]
    target_insulin = target_insulin[:insulin_days]
    pred_insulin = pred_insulin[:insulin_days]

    bg_labels = ["BG_0", "BG_1", "BG_2", "BG_3", "BG_4", "BG_5", "BG_6"]
    if target_insulin.shape[1] == 6:
        insulin_labels = ["Breakfast", "Lunch", "Dinner", "Night", "Micro", "IV"]
    elif target_insulin.shape[1] == 13:
        insulin_labels = ["Flag_Basic", "Flag_Premix", "Flag_None", "Shared_0", "Shared_1", "Shared_2", "Shared_3", "Shared_4", "Shared_5", "Shared_6", "Shared_7", "IV", "Micro"]
    elif target_insulin.shape[1] == 17:
        insulin_labels = ["Flag_Basic", "Flag_Premix", "Flag_None", "Basic_B", "Basic_L", "Basic_D", "Basic_N", "Premix_B_L", "Premix_B_S", "Premix_D_L", "Premix_D_S", "Pump_B", "Pump_L", "Pump_D", "Pump_N", "IV", "Micro"]
    else:
        insulin_labels = [f"Ins_{i}" for i in range(target_insulin.shape[1])]

    bg_headers, bg_rows = _build_pair_rows(target_bg, pred_bg, bg_labels)
    insulin_headers, insulin_rows = _build_pair_rows(target_insulin, pred_insulin, insulin_labels)

    title_font = _load_card_font(22, bold=True)
    header_font = _load_card_font(18, bold=True)
    body_font = _load_card_font(16, bold=False)
    note_font = _load_card_font(15, bold=False)

    cell_width = 118
    header_height = 38
    base_row_height = 52
    left = 28
    top = 28
    width = max(len(bg_headers), len(insulin_headers)) * cell_width + left * 2
    temp_img = Image.new("RGB", (width, 2400), CARD_PAGE_BG)
    temp_draw = ImageDraw.Draw(temp_img)
    y = top
    temp_draw.rounded_rectangle((left, y, width - left, y + 56), radius=14, fill=CARD_HEADER_BG)
    temp_draw.text((left + 18, y + 12), f"Case {sample['check_id']}", font=title_font, fill=CARD_TEXT)
    y += 76
    y = _draw_card_table(temp_draw, left, y, "Blood Glucose", bg_headers, bg_rows, cell_width, header_height, base_row_height, title_font, header_font, body_font, note="Each cell: ground truth on top, prediction below")
    y += 26
    y = _draw_card_table(temp_draw, left, y, "Insulin", insulin_headers, insulin_rows, cell_width, header_height, base_row_height, title_font, header_font, body_font)
    y += 28
    temp_draw.text((left, y), "Cell format: GT / Pred (stacked vertically)", font=note_font, fill=CARD_MUTED)
    final_height = y + 40
    card = temp_img.crop((0, 0, width, final_height))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    card.save(save_path)


def extract_retrieval_topk_details(model, batch):
    with torch.no_grad():
        topk_indices, topk_scores = model._retrieve_topk_indices(batch)
        batch_size = topk_indices.shape[0]
        topk = topk_indices.shape[1]
        device = batch["person_value"].device
        reference_batch = model._build_reference_batch(topk_indices)
        reference_batch = model._move_batch_to_device(reference_batch, device)
        reference_outputs = model._encode_reference_batch(reference_batch, batch_size, topk)
        reference_encoder = model.stage1 if hasattr(model, "stage1") else model.reference_encoder
        current_person_emb = reference_encoder.encode_person(
            batch["person_value"], batch["person_mask"]
        )
        refer_person_emb = reference_outputs["person_emb"]
        refer_real_lengths = reference_outputs["real_lengths"]
        refer_bg_last = model._gather_last_valid_state(reference_outputs["bg_states"], refer_real_lengths)
        refer_tr_last = model._gather_last_valid_state(reference_outputs["tr_states"], refer_real_lengths)
        refer_bg_mean = model._masked_mean_state(reference_outputs["bg_states"], refer_real_lengths)
        refer_tr_mean = model._masked_mean_state(reference_outputs["tr_states"], refer_real_lengths)
        refer_bg_summary = model.bg_state_summary(torch.cat([refer_bg_last, refer_bg_mean], dim=-1))
        refer_tr_summary = model.tr_state_summary(torch.cat([refer_tr_last, refer_tr_mean], dim=-1))
        map_bg = model.bg_Mapper(refer_bg_summary, refer_person_emb, current_person_emb)
        map_tr = model.tr_Mapper(refer_tr_summary, refer_person_emb, current_person_emb)

    topk_scores_np = topk_scores.detach().cpu().numpy()
    temperature = max(float(getattr(model.cfg, "memory_fusion_temperature", 1.0)), 1e-6)
    topk_weights_np = torch.softmax(topk_scores.float() / temperature, dim=1).detach().cpu().numpy()
    map_bg_norm_np = torch.norm(map_bg, dim=-1).detach().cpu().numpy()
    map_tr_norm_np = torch.norm(map_tr, dim=-1).detach().cpu().numpy()

    details = []
    for b in range(topk_indices.shape[0]):
        raw_contrib = topk_weights_np[b] * ((map_bg_norm_np[b] + map_tr_norm_np[b]) / 2.0)
        if raw_contrib.sum() > 0:
            contribution_strength = raw_contrib / raw_contrib.sum()
        else:
            contribution_strength = np.full_like(raw_contrib, 1.0 / max(len(raw_contrib), 1))

        sample_details = []
        for rank, kb_index in enumerate(topk_indices[b].detach().cpu().tolist(), start=1):
            retrieved_sample = model._knowledge_base[kb_index]
            sample_details.append(
                {
                    "rank": rank,
                    "check_id": retrieved_sample["check_id"],
                    "similarity_score": float(topk_scores_np[b, rank - 1]),
                    "similarity_weight": float(topk_weights_np[b, rank - 1]),
                    "contribution_strength": float(contribution_strength[rank - 1]),
                }
            )
        details.append(sample_details)
    return details


def print_retrieval_topk_details(sample_ids, retrieval_details):
    for sample_idx, items in enumerate(retrieval_details):
        sample_id = sample_ids[sample_idx] if sample_idx < len(sample_ids) else f"sample_{sample_idx}"
        print(f"[Retrieval TopK] query_id={sample_id}")
        for item in items:
            print(
                f"  top{item['rank']:>2}: id={item['check_id']} | "
                f"similarity={item['similarity_score']:.6f} | "
                f"weight={item['similarity_weight']:.6f} | "
                f"contribution_strength={item['contribution_strength']:.6f}"
            )


def inference_single_sample(cfg, sample_data):
    """
    Run inference on a single preprocessed sample dict.
    """
    device = cfg.device
    model = TwoStageModel(cfg).to(device)
    
    if cfg.load_model_path and os.path.exists(cfg.load_model_path):
        model.load_state_dict(torch.load(os.path.join(cfg.load_model_path, "model.pth"), map_location=device))
    model.eval()

    # Wrap sample in list and collate
    batch = collate_fn([sample_data])
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(batch, tf_ratio=0.0, mode="test")

    retrieval_details = extract_retrieval_topk_details(model, batch)
    print_retrieval_topk_details(batch.get("check_id", []), retrieval_details)

    insulin_pred = outputs[0]
    bg_pred = outputs[1]
    # discharge_pred = outputs[2]
    # actual_lengths = outputs[3] if len(outputs) > 3 else None
    print(insulin_pred.cpu().numpy(),bg_pred.cpu().numpy())
    insulin_pred_np = insulin_pred.cpu().numpy()
    insulin_daily_pred = insulin_pred_np[..., :4]
    insulin_micro_pred = insulin_pred_np[..., 4:5]
    insulin_iv_pred = insulin_pred_np[..., 5:6]
    return {
        "insulin_prediction": insulin_pred_np,
        "insulin_daily_prediction": insulin_daily_pred,
        "insulin_micro_prediction": insulin_micro_pred,
        "insulin_iv_prediction": insulin_iv_pred,
        "bg_prediction": bg_pred.cpu().numpy(),
        "retrieval_topk": retrieval_details[0] if retrieval_details else [],
        # "discharge_prediction": discharge_pred.cpu().numpy(),
        # "predicted_lengths": actual_lengths.cpu().numpy() if actual_lengths is not None else None
    }

def evaluate_test_loss(cfg, test_dataset):
    device = cfg.device
    model = TwoStageModel(cfg).to(device)

    if cfg.load_model_path and os.path.exists(cfg.load_model_path):
        model.load_state_dict(torch.load(os.path.join(cfg.load_model_path, "model.pth"), map_location=device))
        print(f"Loaded model from {cfg.load_model_path}")
    else:
        print("No valid load_model_path found, using random weights.")

    model.eval()
    criterion = CombinedLoss(cfg)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    total_loss = 0.0
    insulin_loss_total = 0.0
    bg_loss_total = 0.0
    metric_sums = {
        "insulin_dose_mae": 0.0,
        "insulin_dose_mape": 0.0,
        "insulin_dose_rmse": 0.0,
        "insulin_dose_r2": 0.0,
        "bg_insulin_mae_sum": 0.0,
        "bg_mae": 0.0,
        "bg_rmse": 0.0,
        "bg_r2": 0.0,
        "daily_insulin_acc": 0.0,
        "daily_insulin_precision": 0.0,
        "daily_insulin_mae": 0.0,
        "daily_insulin_mape": 0.0,
        "daily_insulin_rmse": 0.0,
        "daily_insulin_r2": 0.0,
        "micro_acc": 0.0,
        "micro_precision": 0.0,
        "micro_mae": 0.0,
        "micro_mape": 0.0,
        "micro_rmse": 0.0,
        "micro_r2": 0.0,
        "iv_acc": 0.0,
        "iv_precision": 0.0,
        "iv_mae": 0.0,
        "iv_mape": 0.0,
        "iv_rmse": 0.0,
        "iv_r2": 0.0,
    }
    metric_histories = {key: [] for key in metric_sums.keys()}

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch, tf_ratio=0.0, mode="test")
            batch_loss, loss_dict = compute_batch_loss(batch, outputs, criterion, cfg, mode="test")
            total_loss += batch_loss.item()
            insulin_loss_total += loss_dict.get("insulin_diversity_base", 0.0)
            bg_loss_total += loss_dict.get("bg_diversity_base", 0.0)
            for key in metric_sums.keys():
                value = loss_dict.get(key, 0.0)
                metric_sums[key] += value
                metric_histories[key].append(value)

    avg_total_loss = total_loss / len(test_loader)
    avg_insulin_loss = insulin_loss_total / len(test_loader)
    avg_bg_loss = bg_loss_total / len(test_loader)
    metric_avgs = {key: metric_sums[key] / len(test_loader) for key in metric_sums.keys()}
    metric_cis = {
        key: bootstrap_mean_ci(metric_histories[key], seed=cfg.seed)
        for key in metric_histories.keys()
    }

    print("\n" + "=" * 60)
    print("Test Dataset Evaluation Completed")
    print(f"Average Test Loss: {avg_total_loss:.4f}")
    print(f"  - Insulin Loss: {avg_insulin_loss:.4f}")
    print(f"  - BG Loss: {avg_bg_loss:.4f}")
    print(f"  - Insulin Dose MAE/MAPE/RMSE/R2: {metric_avgs['insulin_dose_mae']:.4f} / {metric_avgs['insulin_dose_mape']:.2f}% / {metric_avgs['insulin_dose_rmse']:.4f} / {metric_avgs['insulin_dose_r2']:.4f}")
    print(f"    Insulin Dose 95%CI: MAE {format_ci(*metric_cis['insulin_dose_mae'])} | MAPE {format_ci(*metric_cis['insulin_dose_mape'])} | RMSE {format_ci(*metric_cis['insulin_dose_rmse'])} | R2 {format_ci(*metric_cis['insulin_dose_r2'])}")
    print(f"  - BG+Insulin MAE Sum: {metric_avgs['bg_insulin_mae_sum']:.4f} 95%CI {format_ci(*metric_cis['bg_insulin_mae_sum'])}")
    print(f"  - BG MAE/RMSE/R2: {metric_avgs['bg_mae']:.4f} / {metric_avgs['bg_rmse']:.4f} / {metric_avgs['bg_r2']:.4f}")
    print(f"    BG 95%CI: MAE {format_ci(*metric_cis['bg_mae'])} | RMSE {format_ci(*metric_cis['bg_rmse'])} | R2 {format_ci(*metric_cis['bg_r2'])}")
    print(f"  - Daily-Insulin MAE/MAPE/RMSE/R2/Recall/Precision: {metric_avgs['daily_insulin_mae']:.4f} / {metric_avgs['daily_insulin_mape']:.2f}% / {metric_avgs['daily_insulin_rmse']:.4f} / {metric_avgs['daily_insulin_r2']:.4f} / {metric_avgs['daily_insulin_acc']:.4f} / {metric_avgs['daily_insulin_precision']:.4f}")
    print(f"    Daily-Insulin 95%CI: MAE {format_ci(*metric_cis['daily_insulin_mae'])} | MAPE {format_ci(*metric_cis['daily_insulin_mape'])} | RMSE {format_ci(*metric_cis['daily_insulin_rmse'])} | R2 {format_ci(*metric_cis['daily_insulin_r2'])} | Recall {format_ci(*metric_cis['daily_insulin_acc'])} | Precision {format_ci(*metric_cis['daily_insulin_precision'])}")
    print(f"  - Micro MAE/MAPE/RMSE/R2/Recall/Precision: {metric_avgs['micro_mae']:.4f} / {metric_avgs['micro_mape']:.2f}% / {metric_avgs['micro_rmse']:.4f} / {metric_avgs['micro_r2']:.4f} / {metric_avgs['micro_acc']:.4f} / {metric_avgs['micro_precision']:.4f}")
    print(f"    Micro 95%CI: MAE {format_ci(*metric_cis['micro_mae'])} | MAPE {format_ci(*metric_cis['micro_mape'])} | RMSE {format_ci(*metric_cis['micro_rmse'])} | R2 {format_ci(*metric_cis['micro_r2'])} | Recall {format_ci(*metric_cis['micro_acc'])} | Precision {format_ci(*metric_cis['micro_precision'])}")
    print(f"  - IV MAE/MAPE/RMSE/R2/Recall/Precision: {metric_avgs['iv_mae']:.4f} / {metric_avgs['iv_mape']:.2f}% / {metric_avgs['iv_rmse']:.4f} / {metric_avgs['iv_r2']:.4f} / {metric_avgs['iv_acc']:.4f} / {metric_avgs['iv_precision']:.4f}")
    print(f"    IV 95%CI: MAE {format_ci(*metric_cis['iv_mae'])} | MAPE {format_ci(*metric_cis['iv_mape'])} | RMSE {format_ci(*metric_cis['iv_rmse'])} | R2 {format_ci(*metric_cis['iv_r2'])} | Recall {format_ci(*metric_cis['iv_acc'])} | Precision {format_ci(*metric_cis['iv_precision'])}")
    print("=" * 60 + "\n")

    result = {
        "avg_test_loss": avg_total_loss,
        "avg_insulin_loss": avg_insulin_loss,
        "avg_bg_loss": avg_bg_loss,
        "insulin_dose_mae": metric_avgs["insulin_dose_mae"],
        "insulin_dose_mape": metric_avgs["insulin_dose_mape"],
        "insulin_dose_rmse": metric_avgs["insulin_dose_rmse"],
        "insulin_dose_r2": metric_avgs["insulin_dose_r2"],
        "bg_insulin_mae_sum": metric_avgs["bg_insulin_mae_sum"],
        "bg_mae": metric_avgs["bg_mae"],
        "bg_rmse": metric_avgs["bg_rmse"],
        "bg_r2": metric_avgs["bg_r2"],
        "daily_insulin_acc": metric_avgs["daily_insulin_acc"],
        "daily_insulin_precision": metric_avgs["daily_insulin_precision"],
        "daily_insulin_mae": metric_avgs["daily_insulin_mae"],
        "daily_insulin_mape": metric_avgs["daily_insulin_mape"],
        "daily_insulin_rmse": metric_avgs["daily_insulin_rmse"],
        "daily_insulin_r2": metric_avgs["daily_insulin_r2"],
        "micro_acc": metric_avgs["micro_acc"],
        "micro_precision": metric_avgs["micro_precision"],
        "micro_mae": metric_avgs["micro_mae"],
        "micro_mape": metric_avgs["micro_mape"],
        "micro_rmse": metric_avgs["micro_rmse"],
        "micro_r2": metric_avgs["micro_r2"],
        "iv_acc": metric_avgs["iv_acc"],
        "iv_precision": metric_avgs["iv_precision"],
        "iv_mae": metric_avgs["iv_mae"],
        "iv_mape": metric_avgs["iv_mape"],
        "iv_rmse": metric_avgs["iv_rmse"],
        "iv_r2": metric_avgs["iv_r2"],
    }
    for key, (ci_low, ci_high) in metric_cis.items():
        result[f"{key}_ci95_low"] = ci_low
        result[f"{key}_ci95_high"] = ci_high
    return result

def visualize_predictions(sample, predictions, save_path):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    configure_visualization_fonts()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    target_bg = sample["bg1"].numpy()           # (days, 7)
    target_insulin = sample["insulin1"].numpy()
    total_days = sample["bg1"].shape[0]
    
    # Generated full sequence: day 1 observed + day 2..N autoregressive prediction
    gen_bg = predictions['bg_prediction'][0]
    gen_insulin = predictions['insulin_prediction'][0]
    print(predictions['bg_prediction'].shape, gen_bg.shape)
    bg_days = min(total_days, target_bg.shape[0], gen_bg.shape[0])
    insulin_days = min(total_days, target_insulin.shape[0], gen_insulin.shape[0])
    
    time_points_bg = ['BG_0', 'BG_1', 'BG_2', 'BG_3', 'BG_4', 'BG_5', 'BG_6']
    if target_insulin.shape[1] == 6:
        # 6-D stage1 format: 4 daily insulin slots + micro-pump + IV
        time_points_ins = [
            'Breakfast_Total', 'Lunch_Total', 'Dinner_Total', 'Night_Total',
            'Micro_Total', 'IV_Total'
        ]
    elif target_insulin.shape[1] == 13:
        time_points_ins = [
            'Flag_Basic', 'Flag_Premix', 'Flag_None',
            'Shared_0', 'Shared_1', 'Shared_2', 'Shared_3', 'Shared_4',
            'Shared_5', 'Shared_6', 'Shared_7', 'IV_Total', 'Micro_Total'
        ]
    elif target_insulin.shape[1] == 17:
        time_points_ins = [
            'Flag_Basic', 'Flag_Premix', 'Flag_None',
            'Basic_SC_B', 'Basic_SC_L', 'Basic_SC_D', 'Basic_SC_N',
            'Premix_SC_B_Long', 'Premix_SC_B_Short', 'Premix_SC_D_Long', 'Premix_SC_D_Short',
            'Pump_B', 'Pump_L', 'Pump_D', 'Pump_N',
            'IV_Total', 'Micro_Total'
        ]
    else:
        time_points_ins = [f'Insulin_{i}' for i in range(target_insulin.shape[1])]
    days_bg = np.arange(1, bg_days + 1)
    days_ins = np.arange(1, insulin_days + 1)
    
    bg_offsets = np.linspace(-0.3, 0.3, 7)
    ins_offsets = np.linspace(-0.3, 0.3, len(time_points_ins))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Blood glucose subplot
    for tp_idx, tp_name in enumerate(time_points_bg):
        x = days_bg + bg_offsets[tp_idx]
        ax1.scatter(x, target_bg[:bg_days, tp_idx],
                    marker='o', facecolors='none',
                    edgecolors=f'C{tp_idx}', linewidths=1.5,
                    alpha=0.4, label=f'{tp_name}_GT')
        ax1.scatter(x, gen_bg[:bg_days, tp_idx],
                    marker='o', c=f'C{tp_idx}', alpha=0.8,
                    label=f'{tp_name}_Pred')
    
    ax1.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, label='Autoregressive start')
    ax1.set_title(f"Patient {sample['check_id']} - Autoregressive BG Prediction")
    ax1.set_ylabel('Blood Glucose (mmol/L)')
    ax1.set_xticks(days_bg)
    ax1.set_xticklabels([])
    ax1.legend(ncol=4, fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Insulin subplot
    for ins_idx, ins_name in enumerate(time_points_ins):
        x = days_ins + ins_offsets[ins_idx]
        ax2.scatter(x, target_insulin[:insulin_days, ins_idx],
                    marker='*', facecolors='none',
                    edgecolors=f'C{ins_idx}', linewidths=1.5,
                    alpha=0.4, label=f'{ins_name}_GT')
        ax2.scatter(x, gen_insulin[:insulin_days, ins_idx],
                    marker='o', c=f'C{ins_idx}', alpha=0.8,
                    label=f'{ins_name}_Pred')
    
    ax2.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, label='Autoregressive start')
    ax2.set_title(f"Patient {sample['check_id']} - Autoregressive Insulin Prediction")
    ax2.set_xlabel('Hospital Day')
    ax2.set_ylabel('Insulin Dose (IU)')
    ax2.set_xticks(days_ins)
    ax2.legend(ncol=4, fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    
if __name__ == "__main__":
    # Example usage setup
    cfg = opt_config()
    set_seed(cfg.seed)

    # Load a sample (e.g., from test set)
    test_dataset = DiabetesDataset(cfg, mode="test")
    
    
    cfg.d_insulin = test_dataset[0]["d_insulin"]
    # cfg.d_insulin_route_stage = train_dataset[0]["d_insulin_route_stage"]
    cfg.d_person = test_dataset[0]["d_per1"]
    cfg.d_drug = test_dataset[0]["d_drug"]
    if cfg.cut_time != 0: 
        cfg.max_T1 = cfg.cut_time
    evaluate_test_loss(cfg, test_dataset)

    output_dir = os.path.join(cfg.load_model_path if cfg.load_model_path else cfg.save_path, "inference_outputs")
    os.makedirs(output_dir, exist_ok=True)

    num_visualizations = min(10, len(test_dataset))
    for i in range(num_visualizations):
        sample = test_dataset[i]
        # Run inference
        result = inference_single_sample(cfg, sample)
        save_path = os.path.join(output_dir, f"{sample['check_id']}_autoregressive.png")
        visualize_predictions(sample, result, save_path)
        print(f"Saved visualization to: {save_path}")
        card_save_path = os.path.join(output_dir, f"{sample['check_id']}_prediction_card.png")
        save_prediction_case_card(sample, result, card_save_path)
        print(f"Saved case card to: {card_save_path}")

