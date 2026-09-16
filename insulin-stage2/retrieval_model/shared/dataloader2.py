import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from shared.patient_feature_v2 import PatientFeatureExtractor
from shared.excluded_case_ids import EXCLUDED_CASE_IDS
import logging
import random


# =========================
# 2. Dataset
# =========================
class DiabetesDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.insulin_transfer_4 = bool(getattr(cfg, "insulin_transfer_4", 1))
        self.stage2_transfer_4_anomaly_counts = {}
        self.file_path_d1 = cfg.file_path_d1
        self.file_path_d2 = cfg.file_path_d2
        self.file_path_d1_extra = getattr(cfg, "file_path_d1_extra", "")
        self.file_path_d2_extra = getattr(cfg, "file_path_d2_extra", "")
        with open(self.file_path_d1, 'r', encoding='utf-8') as fr:
            self.data_basic = json.load(fr)
        with open(self.file_path_d2, 'r', encoding='utf-8') as fr:
            self.data_premix = json.load(fr)
        self.data_basic_extra = {}
        self.data_premix_extra = {}
        if self.file_path_d1_extra:
            with open(self.file_path_d1_extra, 'r', encoding='utf-8') as fr:
                self.data_basic_extra = json.load(fr)
        if self.file_path_d2_extra:
            with open(self.file_path_d2_extra, 'r', encoding='utf-8') as fr:
                self.data_premix_extra = json.load(fr)

        random.seed(getattr(cfg, "split_seed", cfg.seed))
        self.data_merged = self.data_basic.copy()
        self.data_merged.update(self.data_premix)
        self.data_merged.update(self.data_basic_extra)
        self.data_merged.update(self.data_premix_extra)
        shuffled_keys = list(self.data_merged.keys())
        random.shuffle(shuffled_keys)  # 打乱键的列表（两个原字典的键混合乱序）
        self.data = {k: self.data_merged[k] for k in shuffled_keys}  # 按乱序键重建字典
        self.KEYS_MAIN = ["早餐", "中餐", "晚餐", "Qn", "Qd"]
        self.KEYS_TEMPORAL = ["1", "2", "3"]
        self.log_stage2_pump_stats_from_data(self.data, "before_cut_filter")
        self.data = self.truncate_stage2_days(self.data, cfg)
        self.log_stage2_pump_stats_from_data(self.data, "after_stage2_day_truncate")
        self.data = self.filter_stage2_excluded_routes(self.data)
        self.log_stage2_pump_stats_from_data(self.data, "after_cut_excluded_routes_removed")
        self.data = self.filter_excluded_case_ids(self.data)
        cohort_ids_path = str(getattr(cfg, "cohort_ids_path", "") or "").strip()
        if cohort_ids_path:
            with open(cohort_ids_path, "r", encoding="utf-8") as fr:
                ordered_ids = [str(item) for item in json.load(fr)]
            available = {str(key): key for key in self.data}
            missing = [item for item in ordered_ids if item not in available]
            if missing:
                raise ValueError(
                    f"Common cohort contains {len(missing)} IDs unavailable after S2 filtering; "
                    f"examples: {missing[:8]}"
                )
            self.data = {available[item]: self.data[available[item]] for item in ordered_ids}
            logging.info("Ordered common-cohort filtering kept %d cases from %s", len(self.data), cohort_ids_path)
        self.n_total = len(self.data)

        # personality
        self.personality_path = cfg.personality_path
        self.personality_path_extra = getattr(cfg, "personality_path_extra", "")
        self.personality_path_extra2 = getattr(cfg, "personality_path_extra2", "")
        with open(self.personality_path, 'r', encoding='utf-8') as fr:
            self.personality_data = json.load(fr)
        if self.personality_path_extra:
            with open(self.personality_path_extra, 'r', encoding='utf-8') as fr:
                self.personality_data.update(json.load(fr))
        if self.personality_path_extra2:
            with open(self.personality_path_extra2, 'r', encoding='utf-8') as fr:
                self.personality_data.update(json.load(fr))

        p_data_dict = {}
        
        for k, v in self.data.items():
            if k in self.personality_data:
                p_data_dict[k] = self.personality_data[k]
        self.p_data_dict = p_data_dict
        self.patient_data_dict, self.feature_names = self.extract_personality(self.p_data_dict)
        self.c_pep_mean, self.c_pep_std = self.compute_c_peptide_normalization_stats(self.data)


        self.id_data_list = []
        self.personality_data_list = []
        self.ids = []
        self.tensor_data = []
        for key, value in self.data.items():
            value["id"] = key
            self.ids.append(key)
            self.id_data_list.append(value)  # 用户data
            self.personality_data_list.append(self.patient_data_dict[key])  # 全部用户的personality

        self.database_with_patient = self.construct_data(cfg)
        self.tensor_data = [self.database_with_patient[key] for key in self.ids]  # tensor级别的用户输入/输出model
        self.n_total = len(self.tensor_data)
        if self.insulin_transfer_4 and self.stage2_transfer_4_anomaly_counts:
            logging.warning(
                "Stage2 insulin_transfer_4 anomaly summary | %s",
                ", ".join(
                    f"{tag}={count}" for tag, count in sorted(self.stage2_transfer_4_anomaly_counts.items())
                ),
            )
        self.log_stage2_regimen_stats()
        self.log_stage2_pump_stats()
        fixed_split_path = str(getattr(cfg, "fixed_split_path", "") or "").strip()
        if fixed_split_path:
            with open(fixed_split_path, "r", encoding="utf-8") as fr:
                split_payload = json.load(fr)
            selected_ids = [str(item) for item in split_payload.get(mode, [])]
            id_to_index = {str(check_id): idx for idx, check_id in enumerate(self.ids)}
            missing = [check_id for check_id in selected_ids if check_id not in id_to_index]
            if missing:
                raise ValueError(
                    f"Fixed split {mode} contains {len(missing)} IDs unavailable in dataset; "
                    f"examples: {missing[:8]}"
                )
            self.filtered_indices = [id_to_index[check_id] for check_id in selected_ids]
            self.filtered_data = [self.tensor_data[idx] for idx in self.filtered_indices]
            self.n = len(self.filtered_data)
            self.start_idx = 0
            logging.info(
                "Fixed split summary | mode=%s | total=%d | selected=%d | path=%s",
                mode,
                len(self.tensor_data),
                self.n,
                fixed_split_path,
            )
            print(
                f"Fixed split summary | mode={mode} | total={len(self.tensor_data)} | "
                f"selected={self.n} | path={fixed_split_path}"
            )
        else:
            data_partition = getattr(cfg, "data_partition", "all")
            teacher_data_ratio = float(getattr(cfg, "teacher_data_ratio", 0.5))
            teacher_data_ratio = min(max(teacher_data_ratio, 0.0), 1.0)

            if data_partition in {"teacher", "similarity"}:
                self.train_ratio = 0.8
                self.val_ratio = 0.2
                self.test_ratio = 0.0
            else:
                self.train_ratio = 0.7
                self.val_ratio = 0.1
                self.test_ratio = 0.2

            all_indices = list(range(len(self.tensor_data)))
            split_rng = random.Random(getattr(cfg, "split_seed", cfg.seed))
            split_rng.shuffle(all_indices)

            if data_partition == "teacher":
                n_teacher = int(len(all_indices) * teacher_data_ratio)
                partition_indices = all_indices[:n_teacher]
            elif data_partition == "similarity":
                n_teacher = int(len(all_indices) * teacher_data_ratio)
                partition_indices = all_indices[n_teacher:]
            elif data_partition == "all":
                partition_indices = all_indices
            else:
                raise ValueError(f"Unsupported data_partition: {data_partition}")

            partition_total = len(partition_indices)
            n_train = int(partition_total * self.train_ratio)
            n_val = int(partition_total * self.val_ratio)
            n_test = partition_total - n_train - n_val

            if mode == "train":
                selected_indices = partition_indices[:n_train]
            elif mode == "val":
                selected_indices = partition_indices[n_train:n_train + n_val]
            elif mode == "test":
                selected_indices = partition_indices[n_train + n_val:]
            else:
                raise ValueError(f"Unsupported split: {mode}")

            self.filtered_indices = selected_indices
            self.filtered_data = [self.tensor_data[i] for i in self.filtered_indices]
            self.n = len(self.filtered_data)
            self.start_idx = 0
            logging.info(
                "Dataset split summary | mode=%s | total=%d | partition=%s | teacher_ratio=%.2f | "
                "partition_total=%d | train=%d | val=%d | test=%d | selected=%d",
                mode,
                len(self.tensor_data),
                data_partition,
                teacher_data_ratio,
                partition_total,
                n_train,
                n_val,
                n_test,
                self.n,
            )
            print(
                f"Dataset split summary | mode={mode} | total={len(self.tensor_data)} | "
                f"partition={data_partition} | teacher_ratio={teacher_data_ratio:.2f} | "
                f"partition_total={partition_total} | train={n_train} | val={n_val} | test={n_test} | selected={self.n}"
            )
        
        # # self.id_data_list = []
        # self.personality_data_list = []
        # self.ids = []
        # # self.tensor_data = []
        # for key, value in self.data.items():
        # #     value["id"] = key
        #     self.ids.append(key)
        # #     self.id_data_list.append(value)  # 用户data
        #     self.personality_data_list.append(self.patient_data_dict[key])  # 全部用户的personality

        # self.database_with_patient = self.construct_data(cfg)
        # # print(len(self.database_with_patient))
        
        # # 构建所有患者的张量数据（假设 construct_data 已准备好）
        # self.tensor_data = []
        # for key in self.ids:
        #     self.tensor_data.append(self.database_with_patient[key])

        # # 根据 data_mode 分离训练数据和 val/test 混合数据
        # train_indices = []
        # test_indices = []
        # for i, key in enumerate(self.ids):
        #     data_mode = self.database_with_patient[key]["data_mode"]
        #     if data_mode == "train/val":
        #         train_indices.append(i)
        #     elif data_mode == "test":   # 注意标记必须完全匹配
        #         test_indices.append(i)
        #     else:
        #         raise ValueError(f"Unknown data_mode: {data_mode}")
        
        # # 将索引对应的 tensor 数据收集起来（也可以直接保留索引，按需取数）
        # test_data = [self.tensor_data[i] for i in test_indices]
        # train_data = [self.tensor_data[i] for i in train_indices]
    
    @staticmethod
    def truncate_stage2_days(dict_, cfg):
        max_days = int(getattr(cfg, "stage2_max_days", 10))
        if max_days <= 0:
            return dict_
        truncated = 0
        for values in dict_.values():
            stage2 = values.get("c_pep_after", {})
            if not isinstance(stage2, dict):
                continue
            for field_name, field_value in list(stage2.items()):
                if isinstance(field_value, list) and len(field_value) > max_days:
                    stage2[field_name] = field_value[:max_days]
                    truncated += 1
        if truncated:
            logging.info("Stage2 day truncation kept the first %d days for %d list fields.", max_days, truncated)
        return dict_

    @staticmethod
    def cut_time(dict_, cfg):
        new_dict = {}
        for key, values in dict_.items():
            t = len(values["c_pep_after"]['血糖'])
            if t > cfg.cut_time:
                continue
            else:
                new_dict[key] = values
        return new_dict
        
    def __len__(self):
        return self.n

    def filter_stage2_excluded_routes(self, data_dict):
        filtered = {}
        removed_ids = []
        removed_iv = 0
        removed_micro = 0
        removed_same_slot_pump_subq = 0
        removed_same_day_pump_subq_mixed = 0
        removed_qd_qn_subq_short = 0
        removed_regimen_mapping_conflict = 0
        removed_invalid_stage2_day = 0
        removed_bg_missing_ratio = 0
        bg_missing_ratio_examples = []
        bg_missing_threshold = float(getattr(self.cfg, "stage2_bg_missing_threshold", 0.6))
        for key, value in data_dict.items():
            insulin_list = self._get_stage2_insulin_list(value)
            has_iv, has_micro, has_same_slot_pump_subq, _ = self._sequence_has_stage2_excluded_routes(insulin_list)
            has_same_day_pump_subq_mixed = bool(getattr(self.cfg, "remove_same_day_pump_subq_mixed", 0)) and self._sequence_has_stage2_same_day_pump_subq_mixed(insulin_list)
            has_qd_qn_subq_short = self._sequence_has_stage2_qd_qn_subq_short(insulin_list)
            has_regimen_mapping_conflict = False
            has_invalid_stage2_day = self._sample_has_stage2_invalid_day(value)
            bg_missing_ratio = self._sample_stage2_bg_missing_ratio(value)
            has_excessive_bg_missing = bg_missing_ratio > bg_missing_threshold
            type_list = value.get("c_pep_after", {}).get("胰岛素类型", [])
            if insulin_list:
                has_regimen_mapping_conflict = self._sequence_has_stage2_regimen_mapping_conflict(
                    insulin_list, type_list, sequence_id=f"{key}/stage2_filter"
                )

            should_remove = (
                has_iv or has_micro or has_same_slot_pump_subq or has_same_day_pump_subq_mixed or
                has_qd_qn_subq_short or has_regimen_mapping_conflict or
                has_invalid_stage2_day or has_excessive_bg_missing
            )
            if should_remove:
                removed_ids.append(key)
                removed_iv += int(has_iv)
                removed_micro += int(has_micro)
                removed_same_slot_pump_subq += int(has_same_slot_pump_subq)
                removed_qd_qn_subq_short += int(has_qd_qn_subq_short)
                removed_same_day_pump_subq_mixed += int(has_same_day_pump_subq_mixed)
                removed_regimen_mapping_conflict += int(has_regimen_mapping_conflict)
                removed_invalid_stage2_day += int(has_invalid_stage2_day)
                removed_bg_missing_ratio += int(has_excessive_bg_missing)
                if has_excessive_bg_missing and len(bg_missing_ratio_examples) < 8:
                    bg_missing_ratio_examples.append((key, bg_missing_ratio))
                continue
            filtered[key] = value
        if removed_ids:
            logging.info(
                "Stage2 excluded-route filtering removed %d samples (iv=%d, micro=%d, same_slot_pump_subq=%d, same_day_pump_subq_mixed=%d, qd_qn_subq_short=%d, regimen_mapping_conflict=%d, invalid_stage2_day=%d, bg_missing_ratio_gt_%.0f=%d), remaining %d samples.",
                len(removed_ids),
                removed_iv,
                removed_micro,
                removed_same_slot_pump_subq,
                removed_same_day_pump_subq_mixed,
                removed_qd_qn_subq_short,
                removed_regimen_mapping_conflict,
                removed_invalid_stage2_day,
                bg_missing_threshold * 100.0,
                removed_bg_missing_ratio,
                len(filtered),
            )
        if bg_missing_ratio_examples:
            example_text = ", ".join(
                f"{sample_id}:{ratio:.1%}" for sample_id, ratio in bg_missing_ratio_examples
            )
            logging.warning(
                "Stage2 bg completeness error | filtered %d samples with blood-glucose missing ratio > %.0f%%. Examples: %s",
                removed_bg_missing_ratio,
                bg_missing_threshold * 100.0,
                example_text,
            )
        return filtered

    def filter_excluded_case_ids(self, data_dict):
        excluded_ids = set(getattr(self.cfg, "excluded_case_ids", [])) | set(EXCLUDED_CASE_IDS)
        if not excluded_ids:
            return data_dict

        filtered = {key: value for key, value in data_dict.items() if key not in excluded_ids}
        removed_ids = [key for key in data_dict.keys() if key in excluded_ids]
        if removed_ids:
            logging.info(
                "Stage2 retrieval excluded-case filtering removed %d samples, remaining %d samples. Removed IDs: %s",
                len(removed_ids),
                len(filtered),
                removed_ids,
            )
        return filtered

    def move_leading_stage2_iv_micro_days_to_stage1(self, data_dict):
        adjusted = {}
        moved_cases = 0
        moved_days = 0
        for key, value in data_dict.items():
            sample = json.loads(json.dumps(value, ensure_ascii=False))
            stage1 = sample.get("c_pep_before", {})
            stage2 = sample.get("c_pep_after", {})
            sample["_has_original_s1_history"] = self._stage_has_observed_bg(stage1)
            if not isinstance(stage1, dict) or not isinstance(stage2, dict):
                adjusted[key] = sample
                continue

            bg2 = stage2.get("血糖", [])
            insulin2 = stage2.get("胰岛素医嘱执行", [])
            drug2 = stage2.get("非胰岛素降糖药医嘱执行", [])
            if not (isinstance(bg2, list) and isinstance(insulin2, list) and isinstance(drug2, list)):
                adjusted[key] = sample
                continue

            if not (len(bg2) == len(insulin2) == len(drug2)):
                raise ValueError(
                    f"{key}/stage2_move length mismatch: bg={len(bg2)}, insulin={len(insulin2)}, drug={len(drug2)}"
                )
            n_days = len(bg2)
            leading_days = 0
            for day_idx in range(n_days):
                day_insulin = insulin2[day_idx] if isinstance(insulin2[day_idx], dict) else {}
                if self._stage2_day_has_iv_or_micro(day_insulin):
                    leading_days += 1
                else:
                    break
            if leading_days <= 0:
                adjusted[key] = sample
                continue

            for field_name in ("血糖", "非胰岛素降糖药医嘱执行"):
                before_values = stage1.get(field_name, [])
                after_values = stage2.get(field_name, [])
                if not isinstance(before_values, list) or not isinstance(after_values, list):
                    continue
                stage1[field_name] = before_values + after_values[:leading_days]
                stage2[field_name] = after_values[leading_days:]

            stage1_insulin = stage1.get("胰岛素医嘱执行", [])
            if not isinstance(stage1_insulin, list):
                stage1_insulin = []
            moved_insulin_days = [
                self._convert_stage2_day_to_stage1_day(day)
                for day in insulin2[:leading_days]
            ]
            stage1["胰岛素医嘱执行"] = stage1_insulin + moved_insulin_days
            stage2["胰岛素医嘱执行"] = insulin2[leading_days:]

            type1 = stage1.get("胰岛素类型", [])
            if isinstance(type1, list):
                stage1["胰岛素类型"] = type1 + ["basic"] * leading_days

            type2 = stage2.get("胰岛素类型", [])
            if isinstance(type2, list):
                stage2["胰岛素类型"] = type2[leading_days:]

            sample["c_pep_before"] = stage1
            sample["c_pep_after"] = stage2
            adjusted[key] = sample
            moved_cases += 1
            moved_days += leading_days

        if moved_cases:
            logging.info(
                "Stage2 leading IV/micro days moved to stage1 for %d cases, %d total days.",
                moved_cases,
                moved_days,
            )
        return adjusted

    @staticmethod
    def _stage_has_observed_bg(stage_data):
        if not isinstance(stage_data, dict):
            return False
        for day in stage_data.get("血糖", []) or []:
            if isinstance(day, list) and any(value is not None for value in day):
                return True
        return False

    def _stage2_day_has_iv_or_micro(self, day):
        for slot in list(self.KEYS_MAIN) + ["0", "1", "2", "3"]:
            parsed = self._parse_daily_entry(day.get(slot, []))
            if parsed[4] > 0 or parsed[5] > 0:
                return True
        return False

    def _convert_stage2_day_to_stage1_day(self, day):
        converted = {slot: {} for slot in self.KEYS_MAIN + ["0"] + self.KEYS_TEMPORAL}
        breakfast = self._parse_daily_entry(day.get("早餐", []))
        lunch = self._parse_daily_entry(day.get("中餐", day.get("午餐", [])))
        dinner = self._parse_daily_entry(day.get("晚餐", []))
        qn = self._parse_daily_entry(day.get("Qn", []))
        qd = self._parse_daily_entry(day.get("Qd", []))

        self._set_positive_value(converted["早餐"], "皮下短效", breakfast[1] + breakfast[3])
        self._set_positive_value(converted["中餐"], "皮下短效", lunch[1] + lunch[3])
        self._set_positive_value(converted["晚餐"], "皮下短效", dinner[1] + dinner[3])
        self._set_positive_value(
            converted["Qn"],
            "皮下长效",
            breakfast[0] + breakfast[2] + dinner[0] + dinner[2] + qn[0] + qn[2] + qd[0] + qd[2],
        )

        for src_slot, dst_slot in [
            ("早餐", "早餐"),
            ("中餐", "中餐"),
            ("午餐", "中餐"),
            ("晚餐", "晚餐"),
            ("Qn", "Qn"),
            ("Qd", "Qn"),
            ("0", "0"),
            ("1", "1"),
            ("2", "2"),
            ("3", "3"),
        ]:
            parsed = self._parse_daily_entry(day.get(src_slot, []))
            self._set_positive_value(converted[dst_slot], "静脉短效", parsed[4])
            self._set_positive_value(converted[dst_slot], "微泵短效", parsed[5])
        return converted

    @staticmethod
    def _set_positive_value(slot_entry, name, value):
        if abs(value) > 1e-8:
            slot_entry[name] = float(value)

    def _get_stage2_insulin_list(self, value):
        stage2_block = value.get("c_pep_after", {})
        slot_keys = list(self.KEYS_MAIN) + ["0", "1", "2", "3"]
        for candidate in stage2_block.values():
            if not (isinstance(candidate, list) and candidate and isinstance(candidate[0], dict)):
                continue
            first_day = candidate[0]
            if any(slot in first_day for slot in slot_keys):
                return candidate
        return []

    def _sequence_has_stage2_excluded_routes(self, insulin_list):
        has_iv = False
        has_micro = False
        has_same_slot_pump_subq = False
        has_pump_only = False
        for day in insulin_list:
            for slot in list(self.KEYS_MAIN) + ["0", "1", "2", "3"]:
                parsed = self._parse_daily_entry(day.get(slot, []))
                slot_subq_total = parsed[0] + parsed[1]
                slot_pump_total = parsed[2] + parsed[3]
                has_iv = has_iv or (parsed[4] > 0)
                has_micro = has_micro or (parsed[5] > 0)
                if slot_pump_total > 0 and slot_subq_total > 0:
                    has_same_slot_pump_subq = True
                elif slot_pump_total > 0 and slot_subq_total == 0:
                    has_pump_only = True
            if has_iv or has_micro or has_same_slot_pump_subq:
                return has_iv, has_micro, has_same_slot_pump_subq, has_pump_only
        return has_iv, has_micro, has_same_slot_pump_subq, has_pump_only

    def _sequence_has_stage2_same_day_pump_subq_mixed(self, insulin_list):
        for day in insulin_list:
            day_subq_total = 0.0
            day_pump_total = 0.0
            for slot in list(self.KEYS_MAIN) + ["0", "1", "2", "3"]:
                parsed = self._parse_daily_entry(day.get(slot, []))
                day_subq_total += parsed[0] + parsed[1]
                day_pump_total += parsed[2] + parsed[3]
            if day_subq_total > 0 and day_pump_total > 0:
                return True
        return False

    def _sequence_has_stage2_qd_qn_subq_short(self, insulin_list):
        eps = 1e-8
        for day in insulin_list:
            qd = self._parse_daily_entry(day.get("Qd", []))
            qn = self._parse_daily_entry(day.get("Qn", []))
            qd_is_premix = abs(qd[0]) > eps and abs(qd[1]) > eps
            qn_is_premix = abs(qn[0]) > eps and abs(qn[1]) > eps
            if abs(qd[1]) > eps and not qd_is_premix:
                return True
            if abs(qn[1]) > eps and not qn_is_premix:
                return True
        return False

    def _sequence_has_stage2_premix_pump_conflict(self, insulin_list, type_list, sequence_id=""):
        components = self._build_stage2_raw_route_components(insulin_list, type_list, sequence_id=sequence_id)
        for comp in components:
            if comp["raw_pump_total"] > 0 and comp["raw_premix_like"]:
                return True
        return False

    def _sequence_has_stage2_regimen_mapping_conflict(self, insulin_list, type_list, sequence_id=""):
        ctx = self._build_stage2_regimen_context(insulin_list, type_list, sequence_id=sequence_id)
        normalized_types = ctx["types"]
        final_lunch = ctx["lunch"]
        for i in range(len(insulin_list)):
            if normalized_types[i] != "premix":
                continue
            lunch_subq = abs(final_lunch[i][0]) + abs(final_lunch[i][1])
            if lunch_subq > 0:
                logging.info(
                    "Stage2 regimen mapping conflict | %s day=%d | premix day still has lunch subcutaneous insulin after mapping",
                    sequence_id,
                    i,
                )
                return True
        return False

    def _sample_has_stage2_invalid_day(self, value):
        stage2 = value.get("c_pep_after", {})
        if not isinstance(stage2, dict):
            return False
        bg_list = stage2.get("血糖", []) or []
        insulin_list = stage2.get("胰岛素医嘱执行", []) or []
        drug_list = stage2.get("非胰岛素降糖药医嘱执行", []) or []
        if not (len(bg_list) == len(insulin_list) == len(drug_list)):
            raise ValueError(
                f"stage2 length mismatch: bg={len(bg_list)}, insulin={len(insulin_list)}, drug={len(drug_list)}"
            )
        n_days = len(bg_list)
        for i in range(n_days):
            bg_day = bg_list[i]
            insulin_day = insulin_list[i]
            drug_day = drug_list[i]
            bg_empty = isinstance(bg_day, list) and all(item is None for item in bg_day)
            insulin_total = 0.0
            if isinstance(insulin_day, dict):
                for slot in list(self.KEYS_MAIN) + ["0", "1", "2", "3"]:
                    parsed = self._parse_daily_entry(insulin_day.get(slot, []))
                    insulin_total += sum(parsed)
            drug_total = 0.0
            if isinstance(drug_day, dict):
                drug_total = float(self.merge_time_slots(drug_day).sum().item())
            if bg_empty and insulin_total == 0.0 and drug_total == 0.0:
                return True
        return False

    def _sample_stage2_bg_missing_ratio(self, value):
        stage2 = value.get("c_pep_after", {})
        if not isinstance(stage2, dict):
            return 0.0
        raw_ratio = stage2.get("__raw_stage2_bg_missing_ratio")
        if isinstance(raw_ratio, (int, float)):
            return float(raw_ratio)
        bg_list = stage2.get("血糖", []) or []
        if len(bg_list) == 0:
            return 0.0
        total = 0
        missing = 0
        for day in bg_list:
            if not isinstance(day, list):
                continue
            total += len(day)
            missing += sum(item is None for item in day)
        if total == 0:
            return 0.0
        return missing / float(total)

    @staticmethod
    def _has_s1_history(sample):
        value = sample.get("has_s1_history", 1.0)
        if torch.is_tensor(value):
            return bool(value.detach().cpu().item() > 0)
        return bool(value)

    @staticmethod
    def _has_stage1_day_after_merge(stage1_block):
        for value in stage1_block.values():
            if isinstance(value, list) and (not value or all(isinstance(day, list) for day in value)):
                return len(value) > 0
        return False

    def fetch_database(self):
        return list(self.tensor_data)

    def fetch_personality(self):
        return self.personality_data_list

    def fetch_ids(self):
        return self.ids

    @staticmethod
    def extract_personality(personality_data_dict):
        # 1. 创建特征提取器
        extractor = PatientFeatureExtractor()
        # 2. 提取所有患者的特征向量
        patient_vectors, feature_names = extractor.create_patient_personality_vectors(personality_data_dict)

        logging.info(f"Extracted vectors for {len(patient_vectors)} patients")
        logging.info(f"Vector dimension: {len(next(iter(patient_vectors.values())))}")

        # 3. 获取特征名称
        logging.info(f"Feature names: {feature_names}")

        # 4. 对向量进行标准化
        # print(patient_vectors)
        normalized_vectors = extractor.normalize_vectors(patient_vectors)
        return normalized_vectors, feature_names

    @staticmethod
    def process_bg_with_mask(bg_list, empty_feature_dim=7, empty_placeholder=False):
        if len(bg_list) == 0 and empty_placeholder:
            return (
                torch.zeros((1, empty_feature_dim), dtype=torch.float32),
                torch.zeros((1, empty_feature_dim), dtype=torch.float32),
            )

        bg_value = []
        bg_mask = []

        for day in bg_list:
            val_row = []
            mask_row = []

            for v in day:
                if v is None:
                    val_row.append(0.0)
                    mask_row.append(0.0)
                else:
                    val_row.append(float(v))
                    mask_row.append(1.0)

            bg_value.append(val_row)
            bg_mask.append(mask_row)

        return torch.tensor(bg_value, dtype=torch.float32), torch.tensor(bg_mask, dtype=torch.float32)

    @staticmethod
    def mask_last_day_future_insulin_by_bg(bg_mask, insulin_mask):
        """
        仅对最后一天生效：
        根据最后一天各餐次血糖是否仍在记录，自动决定哪些胰岛素位参与 loss。
        如果某个餐次之后血糖都为空，说明患者可能已在该时段后出院，
        则更后的胰岛素不参与 loss。

        bg 顺序: 早餐前, 早餐后, 午餐前, 午餐后, 晚餐前, 晚餐后, 睡前
        insulin 顺序:
            basic  = 早餐, 午餐, 晚餐, 睡前
            premix = 早餐长, 早餐短, 晚餐长, 晚餐短
        """
        if (
            bg_mask is None
            or insulin_mask is None
            or not torch.is_tensor(bg_mask)
            or not torch.is_tensor(insulin_mask)
            or bg_mask.size(0) == 0
            or insulin_mask.size(0) == 0
        ):
            return insulin_mask

        adjusted_mask = insulin_mask.clone()
        last_bg_mask = bg_mask[-1]
        last_insulin_mask = adjusted_mask[-1].clone()
        new_last_mask = torch.zeros_like(last_insulin_mask)

        if last_insulin_mask.numel() == 5:
            has_breakfast = bool(torch.any(last_bg_mask[0:2] > 0.5).item())
            has_lunch = bool(torch.any(last_bg_mask[2:4] > 0.5).item())
            has_dinner = bool(torch.any(last_bg_mask[4:6] > 0.5).item())
            has_night = bool((last_bg_mask[6] > 0.5).item())

            new_last_mask[0] = 0.0
            if has_breakfast:
                new_last_mask[1] = last_insulin_mask[1]
            if has_lunch:
                new_last_mask[2] = last_insulin_mask[2]
            if has_dinner:
                new_last_mask[3] = last_insulin_mask[3]
            if has_breakfast or has_lunch or has_dinner or has_night:
                new_last_mask[4] = last_insulin_mask[4]

            adjusted_mask[-1] = new_last_mask
            return adjusted_mask

        has_breakfast = bool(torch.any(last_bg_mask[0:2] > 0.5).item())
        has_lunch = bool(torch.any(last_bg_mask[2:4] > 0.5).item())
        has_dinner = bool(torch.any(last_bg_mask[4:6] > 0.5).item())
        has_night = bool((last_bg_mask[6] > 0.5).item())

        if has_breakfast:
            new_last_mask[0] = last_insulin_mask[0]
            new_last_mask[4:6] = last_insulin_mask[4:6]
        if has_lunch:
            new_last_mask[1] = last_insulin_mask[1]
        if has_dinner:
            new_last_mask[2] = last_insulin_mask[2]
            new_last_mask[6:8] = last_insulin_mask[6:8]
        if has_night:
            new_last_mask[3] = last_insulin_mask[3]

        adjusted_mask[-1] = new_last_mask
        return adjusted_mask

    def extract_insulin_by_route(self, item, route_divide=0, stage=1, insulin_type=None):
        """
        根据是否分离注射方式和阶段提取胰岛素值
        
        参数:
            item: dict - 胰岛素数据字典
            route_divide: int - 是否分离注射方式 (0: 不分离, 1: 分离)
            stage: int - 阶段 (1: 三维, 2: 二维)
            insulin_type: str - 指定提取的胰岛素类型 ("长效" 或 "短效")，None表示全部
            
        返回:
            如果 route_divide=0: float (所有注射方式的总和)
            如果 route_divide=1: list (不同注射方式的值)
        """
        if not isinstance(item, dict):
            if route_divide == 0:
                return 0.0
            else:
                return self._get_zero_vector(stage)

        # 如果 route_divide=0，直接返回总和
        if route_divide == 0:
            total = 0.0
            for key, value in item.items():
                # 如果指定了胰岛素类型，只提取对应的值
                if insulin_type is None:
                    total += value
                elif insulin_type == "长效" and "长效胰岛素" in key:
                    total += value
                elif insulin_type == "短效" and "短效胰岛素" in key:
                    total += value
            return total

        # 如果 route_divide=1，分离不同注射方式
        else:
            # 根据stage初始化向量
            result = self._get_zero_vector(stage)

            for key, value in item.items():
                # 检查是否是需要的胰岛素类型
                if insulin_type is not None:
                    if insulin_type == "长效" and "长效胰岛素" not in key:
                        continue
                    elif insulin_type == "短效" and "短效胰岛素" not in key:
                        continue

                # 根据键名分类到不同的注射方式
                if stage == 1:
                    # 三维：皮下注射、胰岛素泵、微泵
                    if "皮下注射" in key or "静脉注射" in key:
                        result[0] += value
                    elif "胰岛素泵注射" in key:
                        result[1] += value
                    elif "微泵注射" in key:
                        result[2] += value
                elif stage == 2:
                    # 二维：皮下注射、胰岛素泵
                    if "皮下注射" in key or "静脉注射" in key:
                        result[0] += value
                    elif "胰岛素泵注射" in key:
                        result[1] += value
                    # stage=2忽略微泵注射

            return result

    def process_pre_mix_insulin_one_day(self, day_data, prev_k3_value, insulin_mode_merge, route_divide, stage=1):
        """
        处理单天预混胰岛素数据

        参数:
            day_data: dict - 单天的胰岛素数据
            prev_k3_value: 前一天的"3"的值，可能是：
                          - None: 第一天
                          - float: 基础胰岛素的不分阶段版本
                          - list: 基础胰岛素的分离注射方式版本
                          - dict: 预混胰岛素返回的格式 {"长效": ..., "短效": ...}
            insulin_mode_merge: int - 胰岛素合并模式
            route_divide: int - 是否分离注射方式 (0: 不分离, 1: 分离)
            stage: int - 阶段 (1: 三维, 2: 二维)

        返回:
            main_vec: list - 当天的主向量 [早长, 早短, 晚长, 晚短]
            current_k3_value: dict - 当天的"3"的值，格式为{"长效": ..., "短效": ...}
        """
        # 1. 检查是否有不允许的键（现在只检查没有不允许的键）
        # 中餐现在允许存在，Qn也允许存在，但Qn有特殊处理

        # 2. 处理早餐胰岛素
        breakfast_data = day_data.get("早餐", [])

        if route_divide == 0:
            # 不分离注射方式
            morning_long = self.extract_insulin_by_route(
                breakfast_data, route_divide=0, stage=stage, insulin_type="长效"
            )
            morning_short = self.extract_insulin_by_route(
                breakfast_data, route_divide=0, stage=stage, insulin_type="短效"
            )
        else:
            # 分离注射方式
            morning_long = self.extract_insulin_by_route(
                breakfast_data, route_divide=1, stage=stage, insulin_type="长效"
            )
            morning_short = self.extract_insulin_by_route(
                breakfast_data, route_divide=1, stage=stage, insulin_type="短效"
            )

        # 3. 处理晚餐胰岛素
        dinner_data = day_data.get("晚餐", [])

        if route_divide == 0:
            evening_long = self.extract_insulin_by_route(
                dinner_data, route_divide=0, stage=stage, insulin_type="长效"
            )
            evening_short = self.extract_insulin_by_route(
                dinner_data, route_divide=0, stage=stage, insulin_type="短效"
            )
        else:
            evening_long = self.extract_insulin_by_route(
                dinner_data, route_divide=1, stage=stage, insulin_type="长效"
            )
            evening_short = self.extract_insulin_by_route(
                dinner_data, route_divide=1, stage=stage, insulin_type="短效"
            )

        # 4. 处理中餐（如果有的话）- 归到晚餐
        lunch_data = day_data.get("中餐", [])
        if isinstance(lunch_data, dict) and any(lunch_data.values()):
            # 提取中餐中的长效和短效
            if route_divide == 0:
                lunch_long = self.extract_insulin_by_route(
                    lunch_data, route_divide=0, stage=stage, insulin_type="长效"
                )
                lunch_short = self.extract_insulin_by_route(
                    lunch_data, route_divide=0, stage=stage, insulin_type="短效"
                )
                # 长效归到晚餐长效，短效归到晚餐短效
                evening_long += lunch_long
                evening_short += lunch_short
            else:
                lunch_long = self.extract_insulin_by_route(
                    lunch_data, route_divide=1, stage=stage, insulin_type="长效"
                )
                lunch_short = self.extract_insulin_by_route(
                    lunch_data, route_divide=1, stage=stage, insulin_type="短效"
                )
                # 向量相加
                if isinstance(evening_long, list) and isinstance(lunch_long, list):
                    if len(evening_long) == len(lunch_long):
                        evening_long = [evening_long[i] + lunch_long[i] for i in range(len(evening_long))]

                if isinstance(evening_short, list) and isinstance(lunch_short, list):
                    if len(evening_short) == len(lunch_short):
                        evening_short = [evening_short[i] + lunch_short[i] for i in range(len(evening_short))]

        # 5. 处理Qd（如果有的话）- 归到早餐
        qd_data = day_data.get("Qd", [])
        if isinstance(qd_data, dict) and any(qd_data.values()):
            # 提取Qd中的长效和短效
            if route_divide == 0:
                qd_long = self.extract_insulin_by_route(
                    qd_data, route_divide=0, stage=stage, insulin_type="长效"
                )
                qd_short = self.extract_insulin_by_route(
                    qd_data, route_divide=0, stage=stage, insulin_type="短效"
                )
                # 长效归到早餐长效，短效归到早餐短效
                morning_long += qd_long
                morning_short += qd_short
            else:
                qd_long = self.extract_insulin_by_route(
                    qd_data, route_divide=1, stage=stage, insulin_type="长效"
                )
                qd_short = self.extract_insulin_by_route(
                    qd_data, route_divide=1, stage=stage, insulin_type="短效"
                )
                # 向量相加
                if isinstance(morning_long, list) and isinstance(qd_long, list):
                    if len(morning_long) == len(qd_long):
                        morning_long = [morning_long[i] + qd_long[i] for i in range(len(morning_long))]

                if isinstance(morning_short, list) and isinstance(qd_short, list):
                    if len(morning_short) == len(qd_short):
                        morning_short = [morning_short[i] + qd_short[i] for i in range(len(morning_short))]

        # 6. 处理Qn（如果有的话）- 归到下一天早上
        # Qn的处理与"3"类似，都是归到下一天
        # 但注意：Qn可能是长效或短效，需要分别提取

        # 7. 处理前一天的"3"值（兼容不同类型）
        # 先转换为统一的字典格式
        if prev_k3_value is None:
            # 第一天，没有前一天的"3"值
            prev_k3_dict = {"长效": 0.0 if route_divide == 0 else self._get_zero_vector(stage),
                            "短效": 0.0 if route_divide == 0 else self._get_zero_vector(stage)}
        elif isinstance(prev_k3_value, dict):
            # 已经是字典格式（来自预混胰岛素）
            prev_k3_dict = prev_k3_value
        else:
            # 标量或列表格式（来自基础胰岛素）
            # 基础胰岛素通常只处理短效，所以长效部分设为0
            if route_divide == 0:
                # 不分阶段：prev_k3_value是标量
                prev_k3_dict = {"长效": 0.0, "短效": float(prev_k3_value)}
            else:
                # 分离注射方式：prev_k3_value是列表
                if isinstance(prev_k3_value, list):
                    prev_short = prev_k3_value
                else:
                    # 如果是标量，创建相应维度的列表
                    prev_short = [float(prev_k3_value)] * len(self._get_zero_vector(stage))

                prev_k3_dict = {"长效": self._get_zero_vector(stage), "短效": prev_short}

        # 8. 将前一天的"3"值加到当天早上
        if route_divide == 0:
            # 不分离注射方式
            morning_long += prev_k3_dict.get("长效", 0.0)
            morning_short += prev_k3_dict.get("短效", 0.0)
        else:
            # 分离注射方式
            prev_long = prev_k3_dict.get("长效", self._get_zero_vector(stage))
            prev_short = prev_k3_dict.get("短效", self._get_zero_vector(stage))

            # 确保维度一致
            if isinstance(morning_long, list) and isinstance(prev_long, list):
                if len(morning_long) == len(prev_long):
                    morning_long = [morning_long[i] + prev_long[i] for i in range(len(morning_long))]

            if isinstance(morning_short, list) and isinstance(prev_short, list):
                if len(morning_short) == len(prev_short):
                    morning_short = [morning_short[i] + prev_short[i] for i in range(len(morning_short))]

        # 9. 处理临时胰岛素1、2、3
        # 初始化当前天的"3"值
        if route_divide == 0:
            current_k3_value = {"长效": 0.0, "短效": 0.0}
        else:
            current_k3_value = {"长效": self._get_zero_vector(stage),
                                "短效": self._get_zero_vector(stage)}

        for k in ["1", "2", "3"]:
            temp_data = day_data.get(k, [])

            if isinstance(temp_data, dict):
                # 提取临时胰岛素中的长效和短效
                if route_divide == 0:
                    temp_long = self.extract_insulin_by_route(
                        temp_data, route_divide=0, stage=stage, insulin_type="长效"
                    )
                    temp_short = self.extract_insulin_by_route(
                        temp_data, route_divide=0, stage=stage, insulin_type="短效"
                    )
                else:
                    temp_long = self.extract_insulin_by_route(
                        temp_data, route_divide=1, stage=stage, insulin_type="长效"
                    )
                    temp_short = self.extract_insulin_by_route(
                        temp_data, route_divide=1, stage=stage, insulin_type="短效"
                    )

                if k == "1":
                    # "1"加到当天早上（长效归早长，短效归早短）
                    if route_divide == 0:
                        morning_long += temp_long
                        morning_short += temp_short
                    else:
                        if isinstance(morning_long, list) and isinstance(temp_long, list):
                            if len(morning_long) == len(temp_long):
                                morning_long = [morning_long[i] + temp_long[i] for i in range(len(morning_long))]
                        if isinstance(morning_short, list) and isinstance(temp_short, list):
                            if len(morning_short) == len(temp_short):
                                morning_short = [morning_short[i] + temp_short[i] for i in range(len(morning_short))]

                elif k == "2":
                    # "2"加到当天晚上（长效归晚长，短效归晚短）
                    if route_divide == 0:
                        evening_long += temp_long
                        evening_short += temp_short
                    else:
                        if isinstance(evening_long, list) and isinstance(temp_long, list):
                            if len(evening_long) == len(temp_long):
                                evening_long = [evening_long[i] + temp_long[i] for i in range(len(evening_long))]
                        if isinstance(evening_short, list) and isinstance(temp_short, list):
                            if len(evening_short) == len(temp_short):
                                evening_short = [evening_short[i] + temp_short[i] for i in range(len(evening_short))]

                elif k == "3":
                    # "3"保存起来，用于下一天
                    if route_divide == 0:
                        current_k3_value["长效"] = temp_long
                        current_k3_value["短效"] = temp_short
                    else:
                        current_k3_value["长效"] = temp_long
                        current_k3_value["短效"] = temp_short

        # 10. 处理Qn（如果有的话）- 归到下一天早上
        qn_data = day_data.get("Qn", [])
        if isinstance(qn_data, dict) and any(qn_data.values()):
            # 提取Qn中的长效和短效
            if route_divide == 0:
                qn_long = self.extract_insulin_by_route(
                    qn_data, route_divide=0, stage=stage, insulin_type="长效"
                )
                qn_short = self.extract_insulin_by_route(
                    qn_data, route_divide=0, stage=stage, insulin_type="短效"
                )
                # Qn归到下一天早上，所以加到current_k3_value中
                current_k3_value["长效"] += qn_long
                current_k3_value["短效"] += qn_short
            else:
                qn_long = self.extract_insulin_by_route(
                    qn_data, route_divide=1, stage=stage, insulin_type="长效"
                )
                qn_short = self.extract_insulin_by_route(
                    qn_data, route_divide=1, stage=stage, insulin_type="短效"
                )
                # 向量相加
                if isinstance(current_k3_value["长效"], list) and isinstance(qn_long, list):
                    if len(current_k3_value["长效"]) == len(qn_long):
                        current_k3_value["长效"] = [current_k3_value["长效"][i] + qn_long[i] for i in range(len(current_k3_value["长效"]))]

                if isinstance(current_k3_value["短效"], list) and isinstance(qn_short, list):
                    if len(current_k3_value["短效"]) == len(qn_short):
                        current_k3_value["短效"] = [current_k3_value["短效"][i] + qn_short[i] for i in range(len(current_k3_value["短效"]))]

        # 11. 构建主向量 [早长, 早短, 晚长, 晚短]
        main_vec = [morning_long, morning_short, evening_long, evening_short]

        return main_vec, current_k3_value

    def _get_zero_vector(self, stage):
        """
        根据阶段获取零向量

        参数:
            stage: int - 阶段 (1: 三维, 2: 二维)

        返回:
            list - 零向量
        """
        if stage == 1:
            return [0.0, 0.0, 0.0]  # 三维：皮下注射、胰岛素泵、微泵
        elif stage == 2:
            return [0.0, 0.0]  # 二维：皮下注射、胰岛素泵
        else:
            return [0.0, 0.0, 0.0]  # 默认三维

    def process_insulin_two_vectors_one_day(self, day_data, prev_k3_value, insulin_mode_merge, route_divide, stage=1):
        """
        处理单天基础胰岛素数据

        参数:
            day_data: dict - 单天的胰岛素数据
            prev_k3_value: 前一天的"3"的值，可能是：
                          - None: 第一天
                          - float: 基础胰岛素的不分阶段版本
                          - list: 基础胰岛素的分离注射方式版本
                          - dict: 预混胰岛素返回的格式 {"长效": ..., "短效": ...}
            insulin_mode_merge: int - 胰岛素合并模式 (0: Qd和Qn不合并, 1: Qd和Qn合并)
            route_divide: int - 是否分离注射方式 (0: 不分离, 1: 分离)
            stage: int - 阶段 (1: 三维, 2: 二维)

        返回:
            main_vec: list - 当天的主向量 [Qn, 短效1, 短效2, Qd] 或类似结构
            current_k3_value: dict - 当天的"3"的值，格式为{"长效": ..., "短效": ...}
        """
        # 1. 处理当前天的 KEYS_MAIN
        main_vec = []

        for k in self.KEYS_MAIN:
            # 提取当前键的值
            if route_divide == 0:
                current_value = self.extract_insulin_by_route(
                    day_data.get(k, []), route_divide=0, stage=stage
                )
            else:
                current_value = self.extract_insulin_by_route(
                    day_data.get(k, []), route_divide=1, stage=stage
                )

            # 处理胰岛素合并模式
            if insulin_mode_merge == 1 and k == "Qd":
                # 如果insulin_mode=1且当前键是Qd，合并到前一个元素
                if main_vec:
                    # 将Qd的值加到前一个元素（通常是Qn）
                    if route_divide == 0:
                        # 不分离注射方式
                        if isinstance(main_vec[-1], (int, float)):
                            main_vec[-1] = main_vec[-1] + current_value
                        else:
                            # 处理向量情况
                            main_vec[-1] = [main_vec[-1][i] + current_value for i in range(len(main_vec[-1]))]
                    else:
                        # 分离注射方式
                        if isinstance(main_vec[-1], list) and isinstance(current_value, list):
                            # 确保维度一致
                            if len(main_vec[-1]) == len(current_value):
                                main_vec[-1] = [main_vec[-1][i] + current_value[i] for i in range(len(main_vec[-1]))]
                        elif isinstance(main_vec[-1], (int, float)):
                            # 如果前一个元素是标量，转换为列表
                            if isinstance(current_value, list):
                                main_vec[-1] = [main_vec[-1] + current_value[i] for i in range(len(current_value))]
                else:
                    # 如果没有前一个元素，直接添加
                    main_vec.append(current_value)
            else:
                # 其他情况，直接添加到main_vec
                main_vec.append(current_value)

        # 2. 处理前一天的"3"值（兼容不同类型）
        # 先转换为统一的字典格式
        if prev_k3_value is None:
            # 第一天，没有前一天的"3"值
            prev_k3_dict = {"长效": 0.0 if route_divide == 0 else self._get_zero_vector(stage),
                            "短效": 0.0 if route_divide == 0 else self._get_zero_vector(stage)}
        elif isinstance(prev_k3_value, dict):
            # 已经是字典格式（来自预混胰岛素）
            prev_k3_dict = prev_k3_value
        else:
            # 标量或列表格式（来自基础胰岛素）
            # 基础胰岛素通常只处理短效，所以长效部分设为0
            if route_divide == 0:
                # 不分阶段：prev_k3_value是标量
                prev_k3_dict = {"长效": 0.0, "短效": float(prev_k3_value)}
            else:
                # 分离注射方式：prev_k3_value是列表
                if isinstance(prev_k3_value, list):
                    prev_short = prev_k3_value
                else:
                    # 如果是标量，创建相应维度的列表
                    prev_short = [float(prev_k3_value)] * len(self._get_zero_vector(stage))

                prev_k3_dict = {"长效": self._get_zero_vector(stage), "短效": prev_short}

        # 3. 将前一天的"3"值加到第0位
        # 基础胰岛素通常只使用短效部分
        if main_vec:
            if route_divide == 0:
                # 不分离注射方式
                if isinstance(main_vec[0], (int, float)):
                    main_vec[0] = main_vec[0] + prev_k3_dict.get("短效", 0.0)
                else:
                    # 处理向量情况
                    main_vec[0] = [main_vec[0][i] + prev_k3_dict.get("短效", 0.0) for i in range(len(main_vec[0]))]
            else:
                # 分离注射方式
                if isinstance(main_vec[0], list):
                    prev_short = prev_k3_dict.get("短效", self._get_zero_vector(stage))
                    if isinstance(prev_short, list) and len(main_vec[0]) == len(prev_short):
                        main_vec[0] = [main_vec[0][i] + prev_short[i] for i in range(len(main_vec[0]))]
        else:
            # 如果main_vec为空，添加prev_k3_value
            main_vec.append(prev_k3_dict.get("短效", 0.0 if route_divide == 0 else self._get_zero_vector(stage)))

        # 4. 处理当前天的 KEYS_TEMPORAL
        # 初始化当前天的"3"值
        if route_divide == 0:
            current_k3_value = {"长效": 0.0, "短效": 0.0}
        else:
            current_k3_value = {"长效": self._get_zero_vector(stage),
                                "短效": self._get_zero_vector(stage)}

        for k in self.KEYS_TEMPORAL:
            # 提取临时胰岛素的值
            if route_divide == 0:
                value = self.extract_insulin_by_route(
                    day_data.get(k, []), route_divide=0, stage=stage
                )
            else:
                value = self.extract_insulin_by_route(
                    day_data.get(k, []), route_divide=1, stage=stage
                )

            if k == "1":
                if len(main_vec) > 1:
                    if route_divide == 0:
                        if isinstance(main_vec[1], (int, float)):
                            main_vec[1] = main_vec[1] + value
                        else:
                            main_vec[1] = [main_vec[1][i] + value for i in range(len(main_vec[1]))]
                    else:
                        if isinstance(main_vec[1], list) and isinstance(value, list):
                            if len(main_vec[1]) == len(value):
                                main_vec[1] = [main_vec[1][i] + value[i] for i in range(len(main_vec[1]))]
                else:
                    main_vec.append(value)

            elif k == "2":
                # k=2 加到第三位（索引2）
                if len(main_vec) > 2:
                    if route_divide == 0:
                        if isinstance(main_vec[2], (int, float)):
                            main_vec[2] = main_vec[2] + value
                        else:
                            main_vec[2] = [main_vec[2][i] + value for i in range(len(main_vec[2]))]
                    else:
                        if isinstance(main_vec[2], list) and isinstance(value, list):
                            if len(main_vec[2]) == len(value):
                                main_vec[2] = [main_vec[2][i] + value[i] for i in range(len(main_vec[2]))]
                else:
                    # 确保有足够的位置
                    while len(main_vec) <= 2:
                        if route_divide == 0:
                            main_vec.append(0.0)
                        else:
                            main_vec.append(self._get_zero_vector(stage))

                    if route_divide == 0:
                        if isinstance(main_vec[2], (int, float)):
                            main_vec[2] = main_vec[2] + value
                        else:
                            main_vec[2] = [main_vec[2][i] + value for i in range(len(main_vec[2]))]
                    else:
                        if isinstance(main_vec[2], list) and isinstance(value, list):
                            if len(main_vec[2]) == len(value):
                                main_vec[2] = [main_vec[2][i] + value[i] for i in range(len(main_vec[2]))]

                # 根据原始逻辑，k=2也要加到第二位（索引1）
                if len(main_vec) > 1:
                    if route_divide == 0:
                        if isinstance(main_vec[1], (int, float)):
                            main_vec[1] = main_vec[1] + value
                        else:
                            main_vec[1] = [main_vec[1][i] + value for i in range(len(main_vec[1]))]
                    else:
                        if isinstance(main_vec[1], list) and isinstance(value, list):
                            if len(main_vec[1]) == len(value):
                                main_vec[1] = [main_vec[1][i] + value[i] for i in range(len(main_vec[1]))]

            elif k == "3":
                # k=3 保存起来，用于下一天
                # 基础胰岛素的"3"通常只包含短效，但我们仍然按照字典格式保存
                if route_divide == 0:
                    current_k3_value["短效"] = value
                else:
                    current_k3_value["短效"] = value

        # 5. 确保main_vec至少有4个元素
        while len(main_vec) < 4:
            if route_divide == 0:
                main_vec.append(0.0)
            else:
                main_vec.append(self._get_zero_vector(stage))

        return main_vec, current_k3_value

    @staticmethod
    def merge_invalid_days(data_before_c_pep):
        trick_delete_invalid_days_flag = 0
        valid_day_begin = 0
        pre_bg = data_before_c_pep["血糖"]
        pre_insulin = data_before_c_pep["胰岛素医嘱执行"]
        pre_insulin_type = data_before_c_pep["胰岛素类型"]
        pre_drug = data_before_c_pep["非胰岛素降糖药医嘱执行"]
        for day in range(0, len(pre_bg)):
            if all(item is None for item in pre_bg[day]) \
                    and all(isinstance(item, list) for key, item in pre_insulin[day].items()) \
                    and all((len(item) == 0 or all(drug == 0 for drug in item)) for key, item in pre_drug[day].items()):
                trick_delete_invalid_days_flag = 1
            else:
                valid_day_begin = day
                break
        data_before_c_pep["血糖"] = pre_bg[valid_day_begin:]
        data_before_c_pep["胰岛素医嘱执行"] = pre_insulin[valid_day_begin:]
        data_before_c_pep["非胰岛素降糖药医嘱执行"] = pre_drug[valid_day_begin:]
        data_before_c_pep["胰岛素类型"] = pre_insulin_type[valid_day_begin:]
        return data_before_c_pep, trick_delete_invalid_days_flag

    @staticmethod
    def _normalize_regimen_type(type_name):
        if type_name is None:
            return "basic"
        type_name = str(type_name).strip().lower()
        if type_name in ["basic", "basal_bolus", "四针"]:
            return "basic"
        if type_name in ["premix", "预混"]:
            return "premix"
        if type_name in ["none", "无", "空白"]:
            return "none"
        raise ValueError(f"未知的胰岛素类型: {type_name}")

    def _is_zero_insulin_day_except_iv_micro(self, day):
        total = 0.0
        for slot in ["早餐", "中餐", "午餐", "晚餐", "Qn", "Qd", "0", "1", "2", "3"]:
            parsed = self._parse_daily_entry(day.get(slot, []))
            total += parsed[0] + parsed[1] + parsed[2] + parsed[3]
        return total == 0.0

    def _infer_nonzero_day_regimen_from_routes(self, day):
        formal_slots = ["早餐", "中餐", "午餐", "晚餐", "Qn", "Qd"]
        has_subq = False
        has_pump = False
        premix_like = False
        for slot in formal_slots:
            parsed = self._parse_daily_entry(day.get(slot, []))
            has_long = parsed[0] > 0
            has_short = parsed[1] > 0
            has_pump = has_pump or (parsed[2] > 0 or parsed[3] > 0)
            if has_long or has_short:
                has_subq = True
            if has_long and has_short:
                premix_like = True
        if premix_like:
            return "premix"
        if has_subq:
            return "basic"
        if has_pump:
            return "basic"
        return None

    def _is_formal_slot_premix_like(self, day):
        for slot in ["早餐", "中餐", "午餐", "晚餐", "Qn", "Qd"]:
            parsed = self._parse_daily_entry(day.get(slot, []))
            if parsed[0] > 0 and parsed[1] > 0:
                return True
        return False

    def _resolve_zero_day_regimens(self, insulin_list, type_list, sequence_id=""):
        if len(insulin_list) != len(type_list):
            raise ValueError(f"insulin_list 和 type_list 长度不一致: {len(insulin_list)} vs {len(type_list)}")

        raw_types = [self._normalize_regimen_type(t) for t in type_list]
        zero_flags = [self._is_zero_insulin_day_except_iv_micro(day) for day in insulin_list]
        resolved_types = raw_types[:]
        explicit_types = []
        for i, is_zero in enumerate(zero_flags):
            if is_zero:
                explicit_types.append(None)
                continue
            inferred_type = self._infer_nonzero_day_regimen_from_routes(insulin_list[i])
            resolved_types[i] = inferred_type if inferred_type is not None else raw_types[i]
            explicit_types.append(resolved_types[i])

        for i, is_zero in enumerate(zero_flags):
            if not is_zero:
                continue

            prev_type = None
            for j in range(i - 1, -1, -1):
                if explicit_types[j] is not None:
                    prev_type = explicit_types[j]
                    break

            next_type = None
            for j in range(i + 1, len(explicit_types)):
                if explicit_types[j] is not None:
                    next_type = explicit_types[j]
                    break

            if next_type is not None:
                resolved_types[i] = next_type
            elif prev_type is not None:
                resolved_types[i] = prev_type
            else:
                resolved_types[i] = raw_types[i]

        return resolved_types

    def _build_stage2_regimen_context(self, insulin_list, type_list, sequence_id=""):
        """
        Build a single stage2 routing context shared by:
        - conflict filtering
        - 13-dim insulin features
        - 8-dim insulin+regimen features

        The routing rules handled here are:
        1. 早餐只有短效（皮下或泵短效）且无长效，中午出现皮下胰岛素 -> 转 basic
        2. 早餐无皮下胰岛素，且前一天已是 basic，中午出现皮下胰岛素 -> 转 basic
        3. 早餐长短皆有、前一天 premix、晚餐有短效 -> 转 basic，早餐长效视作夜长
        4. 三针预混（中午长+短）-> 中午预混按早晚各一半拆分
        5. 早餐和晚餐都是长+短，中午只有短效 -> 仍按 premix，午短并到早餐短效
        6. 早餐长短皆有、中午短效、晚餐短效 -> 转 basic，早餐长效视作夜长
        """
        N = len(insulin_list)
        normalized_types = self._resolve_zero_day_regimens(insulin_list, type_list, sequence_id=sequence_id)
        breakfast, lunch, dinner = [], [], []
        qn_list, qd_list = [], []
        zero, one, two = [], [], []
        daily_iv = [0.0] * N
        daily_micro = [0.0] * N
        eps = 1e-8

        for i in range(N):
            day = insulin_list[i]
            b = self._parse_daily_entry(day.get("早餐", []))
            l = self._parse_daily_entry(day.get("中餐", day.get("午餐", [])))
            d = self._parse_daily_entry(day.get("晚餐", []))
            qn = self._parse_daily_entry(day.get("Qn", []))
            qd = self._parse_daily_entry(day.get("Qd", []))
            z = self._parse_daily_entry(day.get("0", []))
            o = self._parse_daily_entry(day.get("1", []))
            t = self._parse_daily_entry(day.get("2", []))

            breakfast.append(b)
            lunch.append(l)
            dinner.append(d)
            qn_list.append(qn)
            qd_list.append(qd)
            zero.append(z)
            one.append(o)
            two.append(t)

            daily_iv[i] += b[4] + l[4] + d[4] + qn[4] + qd[4] + z[4] + o[4] + t[4]
            daily_micro[i] += b[5] + l[5] + d[5] + qn[5] + qd[5] + z[5] + o[5] + t[5]

        adjusted_types = normalized_types[:]
        day_rules = [None] * N
        for i in range(N):
            if adjusted_types[i] != "premix":
                continue

            prev_type = adjusted_types[i - 1] if i > 0 else None
            b = breakfast[i]
            l = lunch[i]
            d = dinner[i]
            qn = qn_list[i]

            lunch_subq = abs(l[0]) + abs(l[1])
            if lunch_subq <= eps:
                continue

            # Lunch long+short is treated as three-shot premix first.
            if abs(l[0]) > eps and abs(l[1]) > eps:
                continue

            morning_long_total = abs(b[0]) + abs(b[2])
            morning_short_total = abs(b[1]) + abs(b[3])
            morning_subq_total = abs(b[0]) + abs(b[1])

            # 1) 早餐只有短效（泵短效也算正常短效） -> 转 basic
            if morning_short_total > eps and morning_long_total <= eps:
                if prev_type == "basic":
                    adjusted_types[i] = "basic"
                    day_rules[i] = "basic_morning_short_only"
                elif prev_type == "premix":
                    day_rules[i] = "premix_add_lunch_to_dinner_prev_premix"
                continue

            # 2) 早餐没有皮下胰岛素，且前一天 basic -> 转 basic
            if morning_subq_total <= eps:
                if prev_type == "basic":
                    adjusted_types[i] = "basic"
                    day_rules[i] = "basic_morning_no_subq_prev_basic"
                elif prev_type == "premix":
                    day_rules[i] = "premix_add_lunch_to_dinner_prev_premix"
                continue

            # 3) 早餐长短都有，前一天 premix，晚餐有短效 -> 转 basic
            if abs(b[0]) > eps and abs(b[1]) > eps and prev_type == "premix" and abs(d[1]) > eps:
                day_rules[i] = "premix_add_lunch_to_dinner_prev_premix"
                continue

            # 5) 早晚都是长+短，中午只有短效 -> 午短并入早餐短效，仍保留 premix
            if (
                abs(b[0]) > eps and abs(b[1]) > eps and
                abs(d[0]) > eps and abs(d[1]) > eps and
                abs(l[0]) <= eps and abs(l[1]) > eps
            ):
                if prev_type == "basic":
                    adjusted_types[i] = "basic"
                    day_rules[i] = "basic_breakfast_ls_lunch_s_dinner_ls_prev_basic"
                else:
                    day_rules[i] = "premix_add_lunch_short_to_dinner_short_prev_premix"
                continue

            # 6) 早餐长短都有，中午短，晚餐短 -> 转 basic，早餐长效视作夜长
            if (
                abs(b[0]) > eps and abs(b[1]) > eps and
                abs(l[0]) <= eps and abs(l[1]) > eps and
                abs(d[0]) <= eps and abs(d[1]) > eps
            ):
                if prev_type == "basic":
                    adjusted_types[i] = "basic"
                    day_rules[i] = "basic_breakfast_ls_lunch_s_dinner_s"
                else:
                    day_rules[i] = "premix_add_lunch_short_to_dinner_short_prev_premix"
                continue

        if N > 1 and adjusted_types[0] == "premix" and day_rules[0] is None:
            next_type = adjusted_types[1]
            b = breakfast[0]
            l = lunch[0]
            d = dinner[0]

            lunch_subq = abs(l[0]) + abs(l[1])
            morning_long_total = abs(b[0]) + abs(b[2])
            morning_short_total = abs(b[1]) + abs(b[3])
            morning_subq_total = abs(b[0]) + abs(b[1])

            if lunch_subq > eps and not (abs(l[0]) > eps and abs(l[1]) > eps):
                if morning_short_total > eps and morning_long_total <= eps:
                    if next_type == "basic":
                        adjusted_types[0] = "basic"
                        day_rules[0] = "basic_morning_short_only"
                    elif next_type == "premix":
                        day_rules[0] = "premix_add_lunch_to_dinner_prev_premix"
                elif morning_subq_total <= eps:
                    if next_type == "basic":
                        adjusted_types[0] = "basic"
                        day_rules[0] = "basic_morning_no_subq_prev_basic"
                    elif next_type == "premix":
                        day_rules[0] = "premix_add_lunch_to_dinner_prev_premix"
                elif abs(b[0]) > eps and abs(b[1]) > eps and abs(d[1]) > eps:
                    if abs(d[0]) > eps:
                        if next_type == "basic":
                            adjusted_types[0] = "basic"
                            day_rules[0] = "basic_breakfast_ls_lunch_s_dinner_ls_prev_basic"
                        elif next_type == "premix":
                            day_rules[0] = "premix_add_lunch_to_dinner_prev_premix"
                    else:
                        if next_type == "basic":
                            adjusted_types[0] = "basic"
                            day_rules[0] = "basic_breakfast_ls_lunch_s_dinner_s"
                        elif next_type == "premix":
                            day_rules[0] = "premix_add_lunch_short_to_dinner_short_prev_premix"

        final_breakfast = [vec[:] for vec in breakfast]
        final_lunch = [vec[:] for vec in lunch]
        final_dinner = [vec[:] for vec in dinner]
        final_qn = [vec[:] for vec in qn_list]
        final_qd = [vec[:] for vec in qd_list]

        for i in range(N):
            typ = adjusted_types[i]
            if typ == "basic":
                for k in range(6):
                    final_lunch[i][k] += zero[i][k]
                    final_dinner[i][k] += one[i][k]
            elif typ == "premix":
                for k in range(6):
                    final_breakfast[i][k] += zero[i][k]
                    final_dinner[i][k] += one[i][k]

        for i in range(N - 1):
            for k in range(6):
                final_breakfast[i + 1][k] += two[i][k]

        for i in range(N):
            if day_rules[i] == "premix_add_lunch_to_dinner_prev_premix":
                final_dinner[i][0] += final_lunch[i][0]
                final_dinner[i][1] += final_lunch[i][1]
                final_lunch[i][0] = 0.0
                final_lunch[i][1] = 0.0

            if day_rules[i] == "premix_add_lunch_short_to_dinner_short_prev_premix":
                final_dinner[i][1] += final_lunch[i][1]
                final_lunch[i][1] = 0.0

            # 4) 三针预混的中午预混按早晚各拆一半
            if adjusted_types[i] == "premix" and abs(final_lunch[i][0]) > eps and abs(final_lunch[i][1]) > eps:
                final_breakfast[i][0] += final_lunch[i][0] / 2.0
                final_dinner[i][0] += final_lunch[i][0] / 2.0
                final_breakfast[i][1] += final_lunch[i][1] / 2.0
                final_dinner[i][1] += final_lunch[i][1] / 2.0
                final_lunch[i][0] = 0.0
                final_lunch[i][1] = 0.0

        return {
            "types": adjusted_types,
            "day_rules": day_rules,
            "breakfast": final_breakfast,
            "lunch": final_lunch,
            "dinner": final_dinner,
            "qn": final_qn,
            "qd": final_qd,
            "daily_iv": daily_iv,
            "daily_micro": daily_micro,
        }

    def _build_stage2_basic_rows(self, ctx, i):
        breakfast = ctx["breakfast"][i]
        lunch = ctx["lunch"][i]
        dinner = ctx["dinner"][i]
        qn = ctx["qn"][i]
        qd = ctx["qd"][i]
        rule = ctx["day_rules"][i]
        daily_iv = ctx["daily_iv"][i]
        daily_micro = ctx["daily_micro"][i]
        eps = 1e-8

        breakfast_sc_long = breakfast[0]
        breakfast_sc_short = breakfast[1]
        breakfast_pump = breakfast[2] + breakfast[3]
        lunch_sc = lunch[0] + lunch[1]
        lunch_pump = lunch[2] + lunch[3]
        dinner_sc_long = dinner[0]
        dinner_sc_short = dinner[1]
        dinner_pump = dinner[2] + dinner[3]
        night_sc = qn[0] + qn[1] + qd[0] + qd[1]
        night_pump = qn[2] + qn[3] + qd[2] + qd[3]

        move_breakfast_long_to_night = rule in {
            "basic_breakfast_ls_lunch_s_dinner_ls_prev_basic",
            "basic_breakfast_ls_lunch_s_dinner_s",
        }
        move_dinner_long_to_night = (
            rule in {
                "basic_morning_short_only",
                "basic_morning_no_subq_prev_basic",
                "basic_breakfast_ls_lunch_s_dinner_ls_prev_basic",
                "basic_breakfast_ls_lunch_s_dinner_s",
            }
            and abs(dinner_sc_long) > eps
            and abs(dinner_sc_short) > eps
        )

        breakfast_sc = breakfast_sc_short if move_breakfast_long_to_night else (breakfast_sc_long + breakfast_sc_short)
        dinner_sc = dinner_sc_short if move_dinner_long_to_night else (dinner_sc_long + dinner_sc_short)
        night_sc += (breakfast_sc_long if move_breakfast_long_to_night else 0.0)
        night_sc += (dinner_sc_long if move_dinner_long_to_night else 0.0)

        row_basic_13 = [
            breakfast_sc,
            breakfast_pump,
            lunch_sc,
            lunch_pump,
            dinner_sc,
            dinner_pump,
            night_sc,
            night_pump,
            daily_iv,
            daily_micro,
        ]
        row_basic_8 = [
            breakfast_sc + breakfast_pump,
            lunch_sc + lunch_pump,
            dinner_sc + dinner_pump,
            night_sc + night_pump,
        ]
        return row_basic_13, row_basic_8

    def _record_stage2_transfer_4_anomaly(self, tag, sequence_id, day_idx, detail):
        self.stage2_transfer_4_anomaly_counts[tag] = self.stage2_transfer_4_anomaly_counts.get(tag, 0) + 1
        logging.warning(
            "Stage2 insulin_transfer_4 anomaly | tag=%s | sequence=%s | day=%d | %s",
            tag,
            sequence_id,
            day_idx,
            detail,
        )

    def _build_stage2_transfer_4_row(self, ctx, i, sequence_id=""):
        base_type = ctx["types"][i]
        breakfast = ctx["breakfast"][i]
        lunch = ctx["lunch"][i]
        dinner = ctx["dinner"][i]
        qn = ctx["qn"][i]
        qd = ctx["qd"][i]
        eps = 1e-8

        morning_short = breakfast[1] + breakfast[3]
        noon_short = lunch[1] + lunch[3]
        evening_short = dinner[1] + dinner[3]
        long_dose = (
            breakfast[0] + breakfast[2] +
            lunch[0] + lunch[2] +
            dinner[0] + dinner[2]
        )

        qd_is_premix = abs(qd[0]) > eps and abs(qd[1]) > eps
        qn_is_premix = abs(qn[0]) > eps and abs(qn[1]) > eps

        if qd_is_premix:
            morning_short += qd[1]
            long_dose += qd[0] + qd[2] + qd[3]
        else:
            long_dose += qd[0] + qd[1] + qd[2] + qd[3]
        if qn_is_premix:
            evening_short += qn[1]
            long_dose += qn[0] + qn[2] + qn[3]
        else:
            long_dose += qn[0] + qn[1] + qn[2] + qn[3]

        qd_qn_subq_short_total = 0.0
        if abs(qd[1]) > eps and not qd_is_premix:
            qd_qn_subq_short_total += qd[1]
        if abs(qn[1]) > eps and not qn_is_premix:
            qd_qn_subq_short_total += qn[1]
        if abs(qd_qn_subq_short_total) > eps:
            self._record_stage2_transfer_4_anomaly(
                "qd_qn_contains_subq_short",
                sequence_id,
                i,
                f"Qd/Qn contains subcutaneous short-acting insulin; merged into long slot for now. qd={qd}, qn={qn}",
            )
        if base_type == "premix" and (abs(lunch[0]) > eps or abs(lunch[2]) > eps):
            self._record_stage2_transfer_4_anomaly(
                "premix_lunch_contains_long",
                sequence_id,
                i,
                f"premix lunch contains long-acting dose; mapped into the long slot. lunch={lunch}",
            )
        return [
            morning_short,
            noon_short,
            evening_short,
            long_dose,
        ]

    def load_insulin_daily_full_with_regimen_flags(
        self,
        insulin_list,
        type_list,
        sequence_id="",
        allow_empty_placeholder=False,
    ):
        """
        返回:
            insulin_features: 形状 (N, 8)
            ins_mask: 形状 (N, 8)
            regimen_features: 形状 (N, 2)
            regimen_mask: 形状 (N, 2)

        特征顺序:
            insulin_features = [
                basic_sc_早, basic_sc_午, basic_sc_晚, basic_sc_夜,
                premix_sc_早长, premix_sc_早短, premix_sc_晚长, premix_sc_晚短
            ]

            regimen_features =
            [is_basic, is_premix]
        """
        N = len(insulin_list)
        if N == 0 and allow_empty_placeholder:
            insulin_dim = 5 if self.insulin_transfer_4 else 8
            return (
                torch.zeros((1, insulin_dim), dtype=torch.float32),
                torch.zeros((1, insulin_dim), dtype=torch.float32),
                torch.zeros((1, 2), dtype=torch.float32),
                torch.zeros((1, 2), dtype=torch.float32),
            )

        ctx = self._build_stage2_regimen_context(insulin_list, type_list, sequence_id=sequence_id)
        normalized_types = ctx["types"]
        final_breakfast = ctx["breakfast"]
        final_lunch = ctx["lunch"]
        final_dinner = ctx["dinner"]
        final_qn = ctx["qn"]
        final_qd = ctx["qd"]

        features = []
        masks = []
        regimen_features = []
        regimen_masks = []
        for i in range(N):
            base_type = normalized_types[i]
            _, basic_sc_raw = self._build_stage2_basic_rows(ctx, i)
            transfer4_row = self._build_stage2_transfer_4_row(ctx, i, sequence_id=sequence_id)
            premix_sc_raw = transfer4_row

            regimen_total = sum(abs(v) for v in basic_sc_raw) + sum(abs(v) for v in premix_sc_raw)
            if regimen_total == 0.0:
                typ = "premix"
                is_zero_dose_day = True
            else:
                typ = base_type if base_type in ["basic", "premix"] else "basic"
                is_zero_dose_day = False

            if self.insulin_transfer_4:
                flag = 1.0 if typ == "premix" else 0.0
                row = [flag] + transfer4_row
                mask = [0.0, 1.0, 1.0, 1.0, 1.0]
            else:
                if typ == "basic":
                    row = basic_sc_raw + [0.0] * 4
                else:
                    row = [0.0] * 4 + premix_sc_raw
                mask = [1.0] * 8

            if typ == "basic":
                regimen_row = [1.0, 0.0]
                regimen_row_mask = [1.0, 1.0]
            elif typ == "premix":
                regimen_row = [0.0, 1.0]
                regimen_row_mask = [1.0, 1.0]
            else:
                raise ValueError(f"未知的胰岛素类型: {typ}")

            if is_zero_dose_day:
                if self.insulin_transfer_4:
                    row = [0.0] * 5
                    mask = [0.0] * 5
                else:
                    row = [0.0] * 8
                regimen_row = [0.0, 1.0]
                regimen_row_mask = [0.0, 0.0]

            if not self.insulin_transfer_4:
                basic_nonzero = any(abs(v) > 1e-8 for v in row[:4])
                premix_nonzero = any(abs(v) > 1e-8 for v in row[4:])
                if basic_nonzero and premix_nonzero:
                    raise ValueError(f"{sequence_id} 第{i}天生成了 basic/premix 同时非零的 8 维序列，请检查数据整理逻辑")
            else:
                if len(row) != 5:
                    raise ValueError(f"{sequence_id} 第{i}天 insulin_transfer_4 输出维度不是 5，实际为 {len(row)}")

            features.append(row)
            masks.append(mask)
            regimen_features.append(regimen_row)
            regimen_masks.append(regimen_row_mask)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.float32),
            torch.tensor(regimen_features, dtype=torch.float32),
            torch.tensor(regimen_masks, dtype=torch.float32),
        )

    def log_stage2_regimen_stats(self):
        basic_days = 0
        premix_days = 0
        zero_days = 0
        full_basic_patients = 0
        full_premix_patients = 0
        basic_premix_patients = 0
        all_zero_patients = 0

        for key, value in self.data.items():
            stage2 = value.get("c_pep_after", {})
            insulin_list = stage2.get("胰岛素医嘱执行", []) if isinstance(stage2, dict) else []
            type_list = stage2.get("胰岛素类型", []) if isinstance(stage2, dict) else []

            if not insulin_list:
                all_zero_patients += 1
                continue

            has_basic = False
            has_premix = False

            for day_idx, day in enumerate(insulin_list):
                if self._is_zero_insulin_day_except_iv_micro(day):
                    zero_days += 1
                    continue

                type_name = type_list[day_idx] if day_idx < len(type_list) else None
                typ = self._normalize_regimen_type(type_name)
                if typ == "premix":
                    premix_days += 1
                    has_premix = True
                else:
                    basic_days += 1
                    has_basic = True

            if has_basic and has_premix:
                basic_premix_patients += 1
            elif has_basic:
                full_basic_patients += 1
            elif has_premix:
                full_premix_patients += 1
            else:
                all_zero_patients += 1

        logging.info(
            "Stage2 regimen stats | total_patients=%d | full_basic_patients=%d | full_premix_patients=%d | "
            "basic_premix_patients=%d | all_zero_patients=%d | basic_days=%d | premix_days=%d | zero_days=%d",
            len(self.data),
            full_basic_patients,
            full_premix_patients,
            basic_premix_patients,
            all_zero_patients,
            basic_days,
            premix_days,
            zero_days,
        )

    def _build_stage2_raw_route_components(self, insulin_list, type_list, sequence_id=""):
        normalized = insulin_list if insulin_list else []
        normalized_types = self._resolve_zero_day_regimens(normalized, type_list, sequence_id=sequence_id)
        N = len(normalized)
        daily_components = []

        for i in range(N):
            day = normalized[i]
            basic_total = 0.0
            pump_total = 0.0
            for slot in ["早餐", "中餐", "午餐", "晚餐", "Qn", "Qd", "0", "1", "2", "3"]:
                parsed = self._parse_daily_entry(day.get(slot, []))
                basic_total += abs(parsed[0]) + abs(parsed[1])
                pump_total += abs(parsed[2]) + abs(parsed[3])
            typ = normalized_types[i]
            if basic_total == 0.0 and pump_total == 0.0:
                typ = "premix"
            daily_components.append(
                {
                    "raw_basic_total": basic_total,
                    "raw_pump_total": pump_total,
                    "raw_premix_like": self._is_formal_slot_premix_like(day),
                    "resolved_type": typ,
                }
            )

        return daily_components

    def log_stage2_pump_stats_from_data(self, data_dict, tag):
        pump_days = 0
        pump_patients = 0
        pump_basic_labeled_days = 0
        pump_premix_labeled_days = 0
        only_basic_no_pump_patients = 0
        basic_plus_pump_patients = 0
        only_pump_no_basic_patients = 0
        neither_basic_nor_pump_patients = 0

        for sample in data_dict.values():
            stage2 = sample.get("c_pep_after", {})
            insulin_list = stage2.get("胰岛素医嘱执行", []) if isinstance(stage2, dict) else []
            type_list = stage2.get("胰岛素类型", []) if isinstance(stage2, dict) else []
            if not insulin_list:
                continue

            normalized_type_inputs = []
            for typ in type_list:
                if isinstance(typ, dict):
                    typ = next(iter(typ.values())) if len(typ) > 0 else ""
                normalized_type_inputs.append(typ)
            components = self._build_stage2_raw_route_components(
                insulin_list,
                normalized_type_inputs,
                sequence_id=f"pump_stats/{tag}",
            )

            has_pump = False
            has_basic = False
            for comp in components:
                basic_present = comp["raw_basic_total"] > 0
                pump_present = comp["raw_pump_total"] > 0
                if basic_present:
                    has_basic = True
                if not pump_present:
                    continue

                has_pump = True
                pump_days += 1
                typ = comp["resolved_type"]

                if typ == "basic":
                    pump_basic_labeled_days += 1
                else:
                    pump_premix_labeled_days += 1

            if has_pump:
                pump_patients += 1

            if has_basic and (not has_pump):
                only_basic_no_pump_patients += 1
            elif has_basic and has_pump:
                basic_plus_pump_patients += 1
            elif (not has_basic) and has_pump:
                only_pump_no_basic_patients += 1
            else:
                neither_basic_nor_pump_patients += 1

        logging.info(
            "Stage2 pump stats (%s) | pump_patients=%d | pump_days=%d | pump_basic_labeled_days=%d | pump_premix_labeled_days=%d | "
            "only_basic_no_pump_patients=%d | basic_plus_pump_patients=%d | only_pump_no_basic_patients=%d | neither_basic_nor_pump_patients=%d",
            tag,
            pump_patients,
            pump_days,
            pump_basic_labeled_days,
            pump_premix_labeled_days,
            only_basic_no_pump_patients,
            basic_plus_pump_patients,
            only_pump_no_basic_patients,
            neither_basic_nor_pump_patients,
        )

    def log_stage2_pump_stats(self):
        id_data = {}
        for sample in self.id_data_list:
            sample_id = sample.get("id", None)
            if sample_id is not None:
                id_data[sample_id] = sample
        self.log_stage2_pump_stats_from_data(id_data, "final_dataset")
        
    # ---------- 胰岛素解析辅助（返回6维：[皮下长, 皮下短, 泵长, 泵短, 静脉短, 微泵短]）----------
    def _parse_daily_entry(self, entry):
        result = [0.0] * 6
        if not entry or entry == []:
            return result
        for key, value in entry.items():
            val = float(value)
            key_lower = key.lower()
            if '皮下' in key_lower:
                base = 0
            elif '胰岛素泵' in key_lower:
                base = 2
            elif '静脉' in key_lower:
                result[4] += val
                continue
            elif '微泵' in key_lower:
                result[5] += val
                continue
            else:
                continue
            if '长效' in key_lower:
                idx = base
            else:
                idx = base + 1
            result[idx] += val
        return result
    
    @staticmethod
    def extract_c_peptide(person_dict):
        """
        return:
            value: Tensor(2,)
            mask:  Tensor(2,)
        """

        values = torch.zeros(2, dtype=torch.float32)
        mask = torch.zeros(2, dtype=torch.float32)

        # 1️⃣ 空腹 C 肽
        fasting = person_dict["C-肽(空腹)"]
        if isinstance(fasting, list) and len(fasting) > 0:
            v = fasting[0]["检验项值"]
            if v is not None:
                if "<3.33" in v:
                    values[0] = 0.0
                else:
                    values[0] = float(v)
                mask[0] = 1.0

        # 2️⃣ 2 小时 C 肽
        post_2h = person_dict.get("C-肽(2小时)")
        if isinstance(post_2h, list) and len(post_2h) > 0:
            v = post_2h[0].get("检验项值")
            if v is not None:
                if "<3.33" in v:
                    values[1] = 0.0
                else:
                    values[1] = float(v)
                mask[1] = 1.0

        return values, mask

    @staticmethod
    def compute_c_peptide_normalization_stats(data_dict):
        collected = [[], []]
        for patient in data_dict.values():
            values, mask = DiabetesDataset.extract_c_peptide(patient)
            for idx in range(2):
                if mask[idx] > 0:
                    collected[idx].append(float(values[idx]))

        means = []
        stds = []
        for values in collected:
            if len(values) == 0:
                means.append(0.0)
                stds.append(1.0)
                continue
            arr = np.asarray(values, dtype=np.float32)
            mean = float(arr.mean())
            std = float(arr.std())
            if std == 0.0 or not np.isfinite(std):
                std = 1.0
            means.append(mean)
            stds.append(std)

        logging.info(
            "Stage2 C-peptide normalization | mean=%s | std=%s",
            [round(x, 4) for x in means],
            [round(x, 4) for x in stds],
        )
        return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32)

    def normalize_c_peptide(self, c_pep, c_pep_mask):
        normalized = c_pep.clone()
        valid = c_pep_mask > 0
        if valid.any():
            normalized[valid] = (normalized[valid] - self.c_pep_mean[valid]) / self.c_pep_std[valid]
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        normalized[~valid] = 0.0
        return normalized
    
    def prepend_admission_day(self, bg1, bg1_mask):
        """
        在时间序列前面补一天，模拟入院当天。
        bg1: Tensor of shape (T, 7) - T天，每天7个血糖值
        bg1_mask: Tensor of shape (T, 7) - 对应的有效掩码
        Returns:
            new_bg1: (T+1, 7)
            new_mask: (T+1, 7)
        """
        T, _ = bg1.shape
        # 获取第一天的数据
        first_day = bg1[0]          # (7,)
        first_mask = bg1_mask[0]    # (7,)
        
        # 找到第一个非零血糖值的索引
        nonzero_indices = torch.nonzero(first_day)  # 返回非零位置的索引，形状 (K,1)
        if nonzero_indices.numel() > 0:
            idx = nonzero_indices[0].item()  # 第一个非零索引
            admission_val = first_day[idx].item()
        else:
            # 如果第一天全零，则入院血糖为0，索引设为0？但根据需求，补的一天可能全0，mask全0？但用户要求补的那一天只有入院血糖非零，如果入院血糖不存在，可能补一天全零且mask全0？但最好明确。暂时设为0，索引0。
            idx = 0
            admission_val = 0
        
        # 构造补的一天
        new_day = torch.zeros_like(first_day)
        new_day[idx] = admission_val
        
        new_day_mask = torch.zeros_like(first_mask)
        new_day_mask[idx] = 1  # 只有该位置有效
        
        # 堆叠
        new_bg1 = torch.cat([new_day.unsqueeze(0), bg1], dim=0)   # (T+1, 7)
        new_mask = torch.cat([new_day_mask.unsqueeze(0), bg1_mask], dim=0)
        
        return new_bg1, new_mask
    
    def validate_numeric_ranges(self, check_id, result_dict):
        """
        数据守门员：
        - 胰岛素任一维度 > 200 直接报错
        - 血糖任一维度 > 50 直接报错
        """
        insulin_keys = ["insulin1", "insulin2"]
        bg_keys = ["bg1", "bg2"]

        for key in insulin_keys:
            value = result_dict.get(key, None)
            if value is None or not torch.is_tensor(value) or value.numel() == 0:
                continue
            max_value = torch.max(value).item()
            if max_value > 200:
                raise ValueError(
                    f"样本 {check_id} 的 {key} 存在异常胰岛素值: max={max_value:.4f} > 200"
                )

        for key in bg_keys:
            value = result_dict.get(key, None)
            if value is None or not torch.is_tensor(value) or value.numel() == 0:
                continue
            max_value = torch.max(value).item()
            if max_value > 50:
                raise ValueError(
                    f"样本 {check_id} 的 {key} 存在异常血糖值: max={max_value:.4f} > 50"
                )
    
    def merge_time_slots(self, day):
        """
        把一天内不同时间的 24 维 drug 向量 OR 合并
        return: Tensor (24,)
        """
        vec = torch.zeros(24)
        for k in self.KEYS_MAIN + self.KEYS_TEMPORAL:
            slot = day.get(k)
            if isinstance(slot, list) and len(slot) == 24:
                vec = torch.maximum(vec, torch.tensor(slot, dtype=torch.float32))
        return vec
    
    def process_drug_vectors(self, drug_list, mode=2):
        """
        drug_list: List[dict]  # T 天
        mode: 1 / 2 / 3

        return: Tensor (T, d_drug)

        drug_id
            "盐酸二甲双胍片": 0,
            "二甲双胍缓释片": 0,
            "司美格鲁肽注射液": 1,
            "度拉糖肽注射液": 2,
            "利拉鲁肽注射液": 3,
            "利司那肽注射液":4,
            "磷酸西格列汀片":5,
            "利格列汀片": 6,
            "脯氨酸恒格列净片": 7,
            "艾托格列净片": 8,
            "达格列净片": 9,
            "恩格列净片": 10,
            "吡格列酮片": 11,
            "西格列他钠片": 12,
            "格列吡嗪缓释片": 13,
            "格列美脲片": 14,
            "格列喹酮片": 15,
            "阿卡波糖片": 16,
            "阿卡波糖胶囊": 16,
            "伏格列波糖片": 17,
            "桑枝总生物碱片": 18,
            "米格列醇片": 19,
            "多格列艾汀片": 20,
            "瑞格列奈片": 21,
            "那格列奈片": 22
        """
        out_days = []
        for day in drug_list:
            day_vec = self.merge_time_slots(day)  # (24,)

            if mode == 1:
                out = day_vec

            elif mode == 2:
                out = torch.tensor([
                    day_vec[0],
                    day_vec[1:5].max(),  # 1-4
                    day_vec[5],
                    day_vec[6],
                    day_vec[7],
                    day_vec[8],
                    day_vec[9],
                    day_vec[10],
                    day_vec[11],
                    day_vec[12],
                    day_vec[13],
                    day_vec[14],
                    day_vec[15],
                    day_vec[16],
                    day_vec[17:21].max(),
                    day_vec[21],
                    day_vec[22:24].max()  # 22-23
                ])

            elif mode == 3:
                out = torch.cat([
                    day_vec[0:1],
                    day_vec[1:5].max().view(1),  # 1-4
                    day_vec[5:7],  # 5-6
                    day_vec[7:12].max().view(1),  # 7-11
                    day_vec[12:13],  # 12
                    day_vec[13:14],  # 13
                    day_vec[14:17].max().view(1),  # 14-16
                    day_vec[17:21].max().view(1),  # 17-20
                    day_vec[21:22],
                    day_vec[22:24].max().view(1)  # 22-23
                ])

            else:
                raise ValueError(f"Unsupported drug mode: {mode}")

            out_days.append(out)

        return torch.stack(out_days, dim=0)
    
    def process_discharge(self, insulin_list, insulin_mode_merge):
        main_vec = []
        day = insulin_list[0]
        for k in self.KEYS_MAIN + ["Qw"]:
            if insulin_mode_merge == 1 and k == "Qd":
                main_vec[-1] = main_vec[-1] + self.extract_value(day.get(k, []))
            else:
                main_vec.append(self.extract_value(day.get(k, [])))

        insulin_main = torch.tensor(main_vec, dtype=torch.float32)
        return insulin_main

    def construct_data(self, cfg):
        tensor_res = {}
        for check_id in self.ids:
            personality_data_patient = self.patient_data_dict[check_id]
            if cfg.personality == 1:
                # 转换为张量
                personality_tensor = torch.from_numpy(personality_data_patient).float()

                # 创建 mask（在替换之前）
                personality_mask = ~torch.isnan(personality_tensor)
                personality_mask = personality_mask.float()  # 转换为浮点数类型

                # 替换 NaN 为 0
                personality_tensor = torch.nan_to_num(personality_tensor, nan=0.0)
                # print(personality_tensor)
            else:
                personality_tensor = torch.zeros(cfg.d_person)
                personality_mask = torch.ones(cfg.d_person)
            # print(personality_tensor.shape)
            # 将入院开始非法的天数刨除，判定标志为无血糖+无胰岛素+无药物
            self.data[check_id]["c_pep_before"], delet_empty_days = self.merge_invalid_days(self.data[check_id]["c_pep_before"])
            if delet_empty_days:
                logging.info("===== Stage 1 trick delete empty days: " + str(check_id))

            self.data[check_id]["c_pep_after"], delet_empty_days = self.merge_invalid_days(self.data[check_id]["c_pep_after"])
            if delet_empty_days:
                logging.info("===== Stage 2 trick delete empty days: " + str(check_id))

            bg1, bg1_mask = self.process_bg_with_mask(
                self.data[check_id]["c_pep_before"]["血糖"],
                empty_placeholder=True,
            )
            bg2, bg2_mask = self.process_bg_with_mask(self.data[check_id]["c_pep_after"]["血糖"])

            c_pep_raw, c_pep_mask = self.extract_c_peptide(self.data[check_id])
            c_pep = self.normalize_c_peptide(c_pep_raw, c_pep_mask)

            insulin_list_1 = self.data[check_id]["c_pep_before"]["胰岛素医嘱执行"]
            insulin_type_1 = self.data[check_id]["c_pep_before"]["胰岛素类型"]
            insulin_list_2 = self.data[check_id]["c_pep_after"]["胰岛素医嘱执行"]
            insulin_type_2 = self.data[check_id]["c_pep_after"]["胰岛素类型"]

            insulin_main_1, insulin_mask_1, regimen_1, regimen_mask_1 = self.load_insulin_daily_full_with_regimen_flags(
                insulin_list_1,
                insulin_type_1,
                sequence_id=f"{check_id}/stage1",
                allow_empty_placeholder=True,
            )
            insulin_main_2, insulin_mask_2, regimen_2, regimen_mask_2 = self.load_insulin_daily_full_with_regimen_flags(
                insulin_list_2, insulin_type_2, sequence_id=f"{check_id}/stage2"
            )

            drug2 = self.process_drug_vectors(
                self.data[check_id]["c_pep_after"]["非胰岛素降糖药医嘱执行"],
                mode=int(getattr(cfg, "drug_mode", 2)),
            )

            stage2_len = min(int(bg2.size(0)), int(insulin_main_2.size(0)), int(drug2.size(0)))
            stage2_max_days = int(getattr(cfg, "stage2_max_days", 21))
            if stage2_max_days > 0:
                stage2_len = min(stage2_len, stage2_max_days)
            bg2 = bg2[:stage2_len]
            bg2_mask = bg2_mask[:stage2_len]
            insulin_main_2 = insulin_main_2[:stage2_len]
            insulin_mask_2 = insulin_mask_2[:stage2_len]
            regimen_2 = regimen_2[:stage2_len]
            regimen_mask_2 = regimen_mask_2[:stage2_len]
            drug2 = drug2[:stage2_len]
            insulin_mask_2 = self.mask_last_day_future_insulin_by_bg(bg2_mask, insulin_mask_2)

            d_insulin = insulin_main_2.shape[-1]
            d_drug = drug2.shape[-1]
            result_dict = {
                "check_id": check_id,
                "person_value": personality_tensor,
                "person_mask": personality_mask,
                "c_pep_value": c_pep,
                "c_pep_mask": c_pep_mask,
                "bg1": bg1,
                "bg1_mask": bg1_mask,
                "insulin1": insulin_main_1,
                "insulin1_mask": insulin_mask_1,
                "regimen1": regimen_1,
                "regimen1_mask": regimen_mask_1,
                "has_s1_history": torch.tensor(
                    float(self._has_stage1_day_after_merge(self.data[check_id]["c_pep_before"])),
                    dtype=torch.float32,
                ),
                "len1": bg1.size(0),
                "bg2": bg2,
                "bg2_mask": bg2_mask,
                "insulin2": insulin_main_2,
                "insulin2_mask": insulin_mask_2,
                "regimen2": regimen_2,
                "regimen2_mask": regimen_mask_2,
                "drug2": drug2,
                "len2": stage2_len,
                "d_insulin": d_insulin,
                "d_per1": len(personality_tensor),
                "d_drug": d_drug,
            }
            self.validate_numeric_ranges(check_id, result_dict)
            tensor_res[check_id] = result_dict

        return tensor_res

    def __getitem__(self, idx):
        return self.filtered_data[idx]


def collate_fn(batch):
    """
    自定义 collate 函数，用于将多个样本合并成一个批次
    """
    if not batch:
        return {}

    # 获取所有键
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [item[key] for item in batch]

        # 对于字符串类型（如 check_id），直接保存为列表
        if isinstance(values[0], str):
            collated[key] = values
        # 对于标量值（如 d_insulin, d_insulin_route_stage, d_per1, len）
        elif isinstance(values[0], (int, float)):
            if key in {"len", "len1", "len2", "real_lengths"}:
                collated[key] = torch.tensor(values, dtype=torch.long)
            else:
                collated[key] = values[0]  # 假设批次中所有样本的这些值相同
        # 对于张量，进行堆叠
        elif torch.is_tensor(values[0]):
            # 检查是否所有张量形状相同
            if all(v.shape == values[0].shape for v in values):
                collated[key] = torch.stack(values, dim=0)
            else:
                # 形状不同时，需要填充到最大长度
                max_len = max(v.shape[0] for v in values)
                padded_values = []
                for v in values:
                    if v.shape[0] < max_len:
                        # 填充到最大长度
                        pad_size = list(v.shape)
                        pad_size[0] = max_len - v.shape[0]
                        padding = torch.zeros(pad_size, dtype=v.dtype)
                        padded_v = torch.cat([v, padding], dim=0)
                        padded_values.append(padded_v)
                    else:
                        padded_values.append(v)
                collated[key] = torch.stack(padded_values, dim=0)
        else:
            # 其他类型直接保存为列表
            collated[key] = values

    return collated


