import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class PatientFeatureExtractor:
    def __init__(self):
        # 定义特征维度
        self.feature_names = []
        self.extra_features = self._load_extra_features()

    def get_feature_names(self):
        return self.feature_names

    def update_feature_names(self, col):
        if col not in self.feature_names:
            self.feature_names.append(col)

    @staticmethod
    def _safe_dict(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _load_extra_features() -> Dict[str, Dict[str, Any]]:
        extra_path = Path(__file__).resolve().parent.parent / "data" / "feature_extra.json"
        if not extra_path.exists():
            return {}
        with extra_path.open("r", encoding="utf-8") as fr:
            data = json.load(fr)
        return data if isinstance(data, dict) else {}

    def _validate_feature_values(self, features: List[Any], str_id: str) -> None:
        for idx, value in enumerate(features):
            if value is None:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                continue

            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            raise TypeError(
                f"Invalid feature type for patient {str_id}: "
                f"index={idx}, feature={feature_name}, type={type(value).__name__}, value={value!r}"
            )

    def extract_patient_features(self, patient_data: Dict[str, Any], str_id) -> np.ndarray:
        """从单个患者数据中提取特征向量"""
        features = []

        # 1. 基本信息特征 6
        basic_info, age, gender_true_false, BMI = self._extract_basic_info(patient_data, str_id)
        features.extend(basic_info)

        # 2. 病史特征 7
        features.extend(self._extract_medical_history(patient_data, str_id))

        # 3. 入院指标特征
        # admission_indicators, triglyceride = self._extract_admission_indicators(patient_data, str_id)
        # features.extend(admission_indicators)

        # 4. 生命体征特征
        features.extend(self._extract_vital_signs(patient_data))

        # 5. 血糖特征
        # features.extend(self._extract_glucose_features(patient_data))

        # 6. 检验指标特征
        lab_test, eGFR, LDL, TG = self._extract_lab_tests(patient_data, str_id, age, gender_true_false)
        features.extend(lab_test)

        # 7. 诊断特征
        features.extend(self._extract_diagnoses(patient_data, str_id))

        # 8. 额外特征
        features.extend(self._extract_extra_features(str_id))

        # 9. 基于指标的诊断判定结果
        features.extend(self._extract_diagnoses_based_on_metrics(eGFR, LDL, BMI, TG))

        self._validate_feature_values(features, str_id)
        return np.array(features, dtype=np.float32)

    def _extract_basic_info(self, patient_data: Dict[str, Any], str_id) -> List[float]:
        """提取基本信息"""
        features = []

        # 年龄 (归一化到0-1，假设年龄范围0-100)
        age = self._extract_numeric_value(patient_data.get('age', None))
        if age is not None:
            features.append(age / 100.0)
        else:
            features.append(None)
        self.update_feature_names("age")
        # 性别 (男=1, 女=0)
        gender = patient_data.get('gender', None)
        if gender is not None:
            features.append(1.0 if gender == '男' else 0.0)
        else:
            features.append(None)
        self.update_feature_names("gender")

        # BMI计算 (如果有身高体重)
        first_course = self._safe_dict(patient_data.get('首次病程'))
        height = self._extract_numeric_value(self._safe_dict(first_course.get('height')).get('value'))
        weight = self._extract_numeric_value(self._safe_dict(first_course.get('weight')).get('value'))
        features.append(height)
        features.append(weight)
        self.update_feature_names("height")
        self.update_feature_names("weight")
        BMI_ = self._extract_numeric_value(patient_data.get("BMI", None))
        if BMI_ is None:
            if height and weight:
                height_m = float(height) / 100  # cm转m
                weight_kg = float(weight)
                bmi = weight_kg / (height_m ** 2)
                # BMI归一化 (假设范围15-40)
                features.append(bmi)
            else:
                # print(str_id)
                features.append(None)  # 缺失值
        else:
            features.append(BMI_)
        self.update_feature_names("BMI")

        # 糖尿病病程 (提取年份)
        duration = patient_data.get('duration_of_diabetes', None)
        # 判断缺失：None 或 空列表/空元组
        if duration is None:
            features.append(None)
        else:
            # self.normalize_duration[0] 是原始值， [1]是对数值
            features.append(self.normalize_duration(duration, log_constant=1.0, return_years=True)[1])
        self.update_feature_names("duration_of_diabetes")

        g = True if gender == '男' else False
        b = features[-2]  # BMI
        if gender not in ('男', '女'):
            g = None
        return features, age, g, b

    @staticmethod
    def normalize_duration(duration_str, log_constant=1.0, return_years=False):
        import re
        import math
        """
        将包含"+"的患病时长字符串转换为对数归一化值

        Parameters
        ----------
        duration_str : str
            患病时长字符串，可能包含单位和"+"符号
            示例: "10年+", "5个月", "2周", "1年3个月+", "20年"
        log_constant : float, default=1.0
            对数变换中的常数C，默认log(t+1)
        return_raw : bool, default=False
            是否同时返回原始年数和对数变换值

        Returns
        -------
        如果 return_raw=False: 返回对数归一化值
        如果 return_raw=True: 返回元组 (原始年数, 对数归一化值)

        Examples
        --------
        >>> normalize_duration("10年+")
        2.3978952727983707
        >>> normalize_duration("1年3个月+", return_raw=True)
        (1.25, 0.8109302162163288)
        >>> normalize_duration("6个月")
        0.4054651081081644
        """

        # ---------- 1. 解析字符串为年数 ----------
        # 定义单位转换因子（年为单位）
        unit_conversion = {
            '年': 1.0, 'years': 1.0, 'year': 1.0, 'y': 1.0, 'yr': 1.0, 'Y': 1.0,
            '个月': 1 / 12, '月': 1 / 12, 'months': 1 / 12, 'month': 1 / 12, 'm': 1 / 12, 'M': 1 / 12,
            '周': 1 / 52, 'weeks': 1 / 52, 'week': 1 / 52, 'w': 1 / 52, 'W': 1 / 52,
            '天': 1 / 365, '日': 1 / 365, 'days': 1 / 365, 'day': 1 / 365, 'd': 1 / 365, 'D': 1 / 365
        }

        # 预处理：标准化字符串
        text = str(duration_str).strip()
        if not text:
            years = 1 / 365  # 空字符串返回1天
        else:
            total_years = 0.0

            # 处理"半"的特殊情况
            if '半' in text:
                if '年' in text:
                    text = text.replace('半年', '0.5年')
                elif '个月' in text or '月' in text:
                    text = text.replace('半个月', '0.5个月').replace('半月', '0.5个月')
                elif '周' in text:
                    text = text.replace('半周', '0.5周')
                elif '天' in text or '日' in text:
                    text = text.replace('半天', '0.5天').replace('半日', '0.5天')

            # 匹配模式：数字、可能有的+号、单位
            pattern = r'([\d\.]+)\s*(\+?)\s*([^\d\s\+]*)(\+?)'
            matches = re.findall(pattern, text)

            for match in matches:
                value_str, plus_before, unit, plus_after = match

                try:
                    # 解析数值
                    value = float(value_str)

                    # 检查是否有+号（在数字后或单位后）
                    has_plus = (plus_before == '+') or (plus_after == '+')

                    # 如果有+号，加0.5个对应单位
                    if has_plus:
                        value += 0.5

                    # 查找匹配的单位
                    matched_unit = None
                    for key in unit_conversion:
                        if key and unit and key in unit:
                            matched_unit = key
                            break

                    if matched_unit:
                        total_years += value * unit_conversion[matched_unit]
                    elif unit.strip():  # 有单位字符但未匹配到
                        # 尝试模糊匹配
                        if any(char in unit for char in ['年', 'y', 'Y']):
                            total_years += value * 1.0
                        elif any(char in unit for char in ['月', '个', 'm', 'M']):
                            total_years += value * 1 / 12
                        elif any(char in unit for char in ['周', 'w', 'W']):
                            total_years += value * 1 / 52
                        elif any(char in unit for char in ['天', '日', 'd', 'D']):
                            total_years += value * 1 / 365
                        else:
                            # 默认按年处理
                            total_years += value
                except ValueError:
                    # 解析数值失败，跳过该部分
                    continue

            # 如果没有匹配到任何模式，尝试提取纯数字
            if total_years == 0.0:
                try:
                    numbers = re.findall(r'[\d\.]+', text)
                    if numbers:
                        total_years = float(numbers[0])
                        # 检查字符串中是否有+号
                        if '+' in text:
                            total_years += 0.5
                except:
                    # 完全无法解析，返回1天作为默认值
                    total_years = 1 / 365

            years = total_years

        # 确保最小值（避免对0或负数取对数）
        if years <= 0:
            years = 1 / 365  # 设为1天

        # ---------- 2. 应用对数变换 ----------
        log_value = math.log(years + log_constant)

        # ---------- 3. 返回结果 ----------
        if return_years:
            return years, log_value
        else:
            return log_value

    @staticmethod
    def diabetic_nephropathy_to_risk(value):
        """
        将糖尿病肾病描述转换为 KDIGO 风险等级（有序数值）
        返回值：
            0 : 无肾病 (no)
            1 : 低风险
            2 : 中风险
            3 : 高风险
            4 : 极高风险
        """
        if value is None or value == 'no':
            return 0
        if value == 'yes':
            # 有肾病但未分级，保守归为高风险
            return 1

        # 清理空格，转为大写便于匹配
        s = value.upper().replace(' ', '')
        import re

        # ---------- 1. 优先尝试解析 A/G 分级 ----------
        g_val = None
        a_val = None

        # 匹配 A 后跟数字，可能带字母
        a_match = re.search(r'A(\d+[A-Z]?)', s)
        if a_match:
            a_raw = a_match.group(1)
            # 转换 A 等级为数字 1-3
            if a_raw == '1':
                a_val = 1
            elif a_raw == '2':
                a_val = 2
            elif a_raw == '3':
                a_val = 3
            else:
                a_val = 3  # 异常情况，默认最重

        # 匹配 G 后跟数字，可能带 a/b
        g_match = re.search(r'G(\d+[A-Z]?)', s)
        if g_match:
            g_raw = g_match.group(1)
            # 将 G1, G2, G3a, G3b, G4, G5 映射为数字代码
            if g_raw == '1':
                g_val = 1
            elif g_raw == '2':
                g_val = 2
            elif g_raw == '3A':
                g_val = 3
            elif g_raw == '3B':
                g_val = 4
            elif g_raw == '3':
                g_val = 3
            elif g_raw == '4':
                g_val = 5
            elif g_raw == '5':
                g_val = 6
            else:
                g_val = 6

        # 如果成功解析出 A 和 G 等级，则使用风险矩阵返回
        if g_val is not None and a_val is not None:
            risk_matrix = {
                (1, 1): 1, (1, 2): 2, (1, 3): 3,
                (2, 1): 1, (2, 2): 2, (2, 3): 3,
                (3, 1): 2, (3, 2): 3, (3, 3): 3,
                (4, 1): 3, (4, 2): 3, (4, 3): 4,
                (5, 1): 4, (5, 2): 4, (5, 3): 4,
                (6, 1): 4, (6, 2): 4, (6, 3): 4,
            }
            return risk_matrix.get((g_val, a_val), 3)

        # ---------- 2. 若未成功解析 A/G，再尝试解析中文分期 ----------
        roman_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
        stage_match = re.search(r'([IVXLCDM]+|[一二三四五]|\d+)[期]', s)
        if stage_match:
            stage_str = stage_match.group(1)
            stage_num = None
            if stage_str in roman_map:
                stage_num = roman_map[stage_str]
            elif stage_str in ['一', '二', '三', '四', '五']:
                stage_num = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5}[stage_str]
            elif stage_str.isdigit():
                stage_num = int(stage_str)
            if stage_num is not None:
                if stage_num == 1:
                    return 1
                elif stage_num == 2:
                    return 2
                elif stage_num == 3:
                    return 3
                elif stage_num >= 4:
                    return 4

        # 如果以上都未解析成功，返回默认高风险
        return 3

    def _extract_medical_history(self, patient_data: Dict[str, Any], str_id) -> List[float]:
        """提取病史特征"""
        features = []
        if str_id == "0001281818_2358640":
            a = 111
        personal_hist = patient_data.get('personal_history', None)
        lifestyle = patient_data.get('lifestyle_factors', None)

        if lifestyle:
            # 吸烟史
            smoking = lifestyle.get('smoking_history', 'no')
            features.append(1.0 if smoking == 'yes' else 0.0)

            # 饮酒史
            drinking = lifestyle.get('drinking_history', 'no')
            features.append(1.0 if drinking == 'yes' else 0.0)
        else:
            features.append(None)
            features.append(None)

        if personal_hist:

            # 心脑血管病史
            cardio = personal_hist.get('心血管病史', 'no')
            features.append(1.0 if cardio == 'yes' else 0.0)

            # 肝炎史
            cardio = personal_hist.get('肝炎史', 'no')
            features.append(1.0 if cardio == 'yes' else 0.0)

            # 骨折史
            fracture = personal_hist.get('骨折史', 'no')
            features.append(1.0 if fracture == 'yes' else 0.0)

            # 卒中
            stroke = personal_hist.get('卒中史', 'no')
            features.append(1.0 if stroke == 'yes' else 0.0)

            # 胰腺炎史
            pancreatitis = personal_hist.get('胰腺炎史', 'no')
            features.append(1.0 if pancreatitis == 'yes' else 0.0)

            # 甲状腺髓样癌史
            medullary_thyroid_cancer = personal_hist.get('甲状腺髓样癌史', 'no')
            features.append(1.0 if medullary_thyroid_cancer == 'yes' else 0.0)

            # 心衰
            heart_failure = personal_hist.get('心衰史', 'no')
            features.append(1.0 if heart_failure == 'yes' else 0.0)

            # 低血糖
            hypoglycemia = personal_hist.get('低血糖史', 'no')
            features.append(1.0 if hypoglycemia == 'yes' else 0.0)

            # 生酮饮食
            ketogenic_diet = personal_hist.get('生酮饮食', 'no')
            features.append(1.0 if ketogenic_diet == 'yes' else 0.0)

            # 胆囊切除术
            cholecystectomy = personal_hist.get('胆囊切除术史', 'no')
            features.append(1.0 if cholecystectomy == 'yes' else 0.0)

            # 肝炎肺结核史
            infection = personal_hist.get('肺结核史', 'no')
            features.append(1.0 if infection == 'yes' else 0.0)

        else:
            for i in range(0, 11):
                features.append(None)

        self.update_feature_names("smoking_history")
        self.update_feature_names("drinking_history")

        self.update_feature_names("心血管病史")
        self.update_feature_names("肝炎史")
        self.update_feature_names("骨折史")
        self.update_feature_names("卒中史")
        self.update_feature_names("胰腺炎史")
        self.update_feature_names("甲状腺髓样癌史")
        self.update_feature_names("心衰史")
        self.update_feature_names("低血糖史")
        self.update_feature_names("生酮饮食")
        self.update_feature_names("胆囊切除术")
        self.update_feature_names("肺结核史")
        return features

    def _extract_admission_indicators(self, patient_data: Dict[str, Any], str_id) -> List[float]:
        """提取入院指标"""
        features = []
        admission_indicators = self._safe_dict(patient_data.get('入院指标'))
        # indicators = patient_data.get('入院指标', {})
        if str_id == "0003335583_5241685":
            a = 1
        # # 尿素 (正常范围2.8-7.2 mmol/L)
        # urea = self._extract_numeric_value(indicators.get('尿素'))
        # features.append(self._normalize_lab_value(urea, 2.8, 7.2))
        #
        # # 钠 (正常范围135-145 mmol/L)
        # sodium = self._extract_numeric_value(indicators.get('钠'))
        # features.append(self._normalize_lab_value(sodium, 135, 145))
        #
        # # 血糖/糖 (正常范围3.9-6.1 mmol/L)
        # glucose_admit = self._extract_numeric_value(indicators.get('糖'))
        # features.append(self._normalize_lab_value(glucose_admit, 3.9, 6.1))
        #
        # # 超敏C-反应蛋白 (正常范围<10 mg/L)
        # crp = self._extract_numeric_value(indicators.get('超敏C-反应蛋白'))
        # features.append(self._normalize_lab_value(crp, 0, 10))

        # 甘油三酯
        triglyceride = self._extract_numeric_value(
            self._safe_dict(admission_indicators.get('甘油三脂')).get('value')
        )
        if triglyceride is None:
            features.append(None)
        else:
            features.append(self._normalize_lab_value(triglyceride, 0.56, 1.7))
        self.update_feature_names("甘油三酯")
        return features, triglyceride

    def _extract_vital_signs(self, patient_data: Dict[str, Any]) -> List[float]:
        """提取生命体征"""
        features = []
        first_course = self._safe_dict(patient_data.get('首次病程'))
        # vital_signs
        # 收缩压 (正常范围90-120 mmHg)
        systolic = self._extract_numeric_value(self._safe_dict(first_course.get('收缩压')).get('value'))
        if systolic is None:
            features.append(None)
        else:
            features.append(self._normalize_lab_value(systolic, 90, 120))
        self.update_feature_names("收缩压")

        # 舒张压 (正常范围60-80 mmHg)
        diastolic = self._extract_numeric_value(self._safe_dict(first_course.get('舒张压')).get('value'))
        if diastolic is None:
            features.append(None)
        else:
            features.append(self._normalize_lab_value(diastolic, 60, 80))
        self.update_feature_names("舒张压")

        # 心率 (正常范围60-100 bpm)
        heart_rate = self._extract_numeric_value(self._safe_dict(first_course.get('heart_rate')).get('value'))
        if heart_rate is None:
            features.append(None)
        else:
            features.append(self._normalize_lab_value(heart_rate, 60, 100))
        self.update_feature_names('heart_rate')

        # 血氧饱和度 (正常范围95-100%)
        spo2 = self._extract_numeric_value(self._safe_dict(first_course.get('血氧饱和度')).get('value'))
        if spo2 is None:
            features.append(None)
        else:
            features.append(self._normalize_lab_value(spo2, 95, 100))
        self.update_feature_names('血氧饱和度')

        # 呼吸频率 (正常范围12-20次/分)
        respiration = self._extract_numeric_value(self._safe_dict(first_course.get('呼吸')).get('value'))
        if respiration is None:
            features.append(None)
        else:
            features.append(self._normalize_lab_value(respiration, 12, 20))
        self.update_feature_names('呼吸')

        # 体温 (正常范围36-37.5°C)
        temperature = self._extract_numeric_value(self._safe_dict(first_course.get('体温')).get('value'))
        if temperature is None:
            features.append(None)
        else:
            features.append(self._normalize_lab_value(temperature, 36, 37.5))
        self.update_feature_names('体温')

        features.append(self.blood_pressure_grade_from_first_course(patient_data))
        self.update_feature_names("高血压分级")
        return features

    def _extract_glucose_features(self, patient_data: Dict[str, Any]) -> List[float]:
        """提取血糖特征"""
        features = []
        glucose_data = self._safe_dict(patient_data.get('血糖'))

        # 首次空腹血糖
        first_fasting = self._extract_numeric_value(
            self._safe_dict(glucose_data.get('首次空腹血糖')).get('测量值')
        )
        features.append(self._normalize_lab_value(first_fasting, 3.9, 6.1))
        self.update_feature_names('首次空腹血糖')
        # # first24h血糖统计特征
        # first24h = glucose_data.get('first24h', [])
        # if first24h:
        #     features.append(np.mean(first24h) / 30.0)  # 均值归一化
        #     features.append(np.std(first24h) / 15.0)  # 标准差归一化
        #     features.append(max(first24h) / 30.0)  # 最大值归一化
        # else:
        #     features.extend([0.0, 0.0, 0.0])
        #
        # # last24h血糖统计特征
        # last24h = glucose_data.get('last24h', [])
        # if last24h:
        #     features.append(np.mean(last24h) / 30.0)
        #     features.append(np.std(last24h) / 15.0)
        #     features.append(max(last24h) / 30.0)
        # else:
        #     features.extend([0.0, 0.0, 0.0])
        #
        # # 血糖改善程度 (first24h均值 - last24h均值)
        # if first24h and last24h:
        #     improvement = np.mean(first24h) - np.mean(last24h)
        #     features.append(improvement / 20.0)  # 归一化
        # else:
        #     features.append(0.0)

        return features

    def _extract_lab_tests(self, patient_data: Dict[str, Any], str_id, age, gender_true_false) -> List[float]:
        """提取检验指标特征"""
        features = []
        lab_tests = self._safe_dict(patient_data.get('检验'))

        # 只取最新的检验结果
        test_mappings = {
            'HbA1c': (4.0, 6.5),  # 正常范围%
            'HDL-C': (1.0, 1.7),  # 正常范围mmol/L
            'LDL-C': (0, 3.4),  # 正常范围mmol/L
            'TC': (0, 5.2),  # 总胆固醇正常范围
            'TG': (0.56, 1.7),  # 甘油三酯正常范围
            'ALT': (0, 40),  # 谷丙转氨酶
            'AST': (0, 40),  # 谷草转氨酶
            'UA': (208, 428),  # 尿酸(男性)
            'SCR': (44, 106),  # 肌酐(男性)
            'TT4': (65.0, 165.0),  # 总甲状腺素nmol/L
            'FT4': (10.0, 25.0),  # pmol/L 游离甲状腺素
            'TSH': (0.4, 4.5),  # mIU/L 促甲状腺素
            "β-羟丁酸": (20, 300),  # μmol/L
        }

        for test_name, (low, high) in test_mappings.items():
            test_values = lab_tests.get(test_name)
            if test_values and isinstance(test_values, list):
                # 取最早的检验结果
                latest_test = test_values[0]
                # self._extract_numeric_value((lambda x: x.get('测量值') if x else None)(patient_data.get('检验', {}).get('test_name,')))
                value = self._extract_numeric_value(latest_test.get('检验项值'))
                features.append(self._normalize_lab_value(value, low, high))
                self.update_feature_names(test_name)
                if test_name == "TG":
                    # Keep the raw TG value for the derived TG>=1.7 lipid marker.
                    TG = value
            else:
                features.append(None)  # 缺失值
                self.update_feature_names(test_name)
                if test_name == "TG":
                    TG = None

        urine_leukocyte = self.extract_urine_leukocyte_value(
            self._safe_dict(self.extra_features.get(str_id)).get("尿白细胞")
        )
        features.append(urine_leukocyte if urine_leukocyte is not None else None)
        self.update_feature_names("尿白细胞")

        # AST/ALT比值
        ast_alt_ratio = lab_tests.get('AST/ALT', None)
        if ast_alt_ratio and isinstance(ast_alt_ratio, list):
            latest_ratio = ast_alt_ratio[0]
            ratio_val = self._extract_numeric_value(latest_ratio.get('检验项值'))
            # 正常比值约0.8-1.5
            features.append(self._normalize_lab_value(ratio_val, 0.8, 1.5))
        elif lab_tests.get('AST') and lab_tests.get('ALT'):
            ratio_val = self.calculate_ast_alt(lab_tests.get('AST'), lab_tests.get('ALT'))
            # print(str_id, ratio_val)
            features.append(self._normalize_lab_value(ratio_val, 0.8, 1.5))
        else:
            features.append(None)
        self.update_feature_names("AST/ALT")

        # eGFR
        scr_values = lab_tests.get('SCR')
        if scr_values and isinstance(scr_values, list):
            v = self._extract_numeric_value(scr_values[0].get('检验项值'))
            eGFR = self.calculate_egfr_2021(v, age, gender_true_false)
            features.append(self._normalize_lab_value(eGFR, 1, 200))
        else:
            eGFR = None
            features.append(eGFR)
        self.update_feature_names("eGFR")

        ldl_values = lab_tests.get('LDL-C')
        if ldl_values and isinstance(ldl_values, list):
            # 取最早的检验结果
            latest_test = ldl_values[0]
            ldl = self._extract_numeric_value(latest_test.get('检验项值'))
        else:
            ldl = None

        return features, eGFR, ldl, TG

    @staticmethod
    def calculate_egfr_2021(scr: float, age, is_male: bool, scr_units: str = 'μmol/L') -> float:
        """
        使用2021 CKD-EPI公式计算eGFR（去种族化版本）。

        参数:
            scr_mg_dl: 血清肌酐 (mg/dL)
            age: 年龄 (岁)
            is_male: 是否为男性 (True/False)

        返回:
            eGFR值 (mL/min/1.73m²)，已限制在合理范围。
        """
        # 1. 输入验证与单位转换
        scr = PatientFeatureExtractor._extract_numeric_value(scr)
        age = PatientFeatureExtractor._extract_numeric_value(age)
        if scr is None or age is None or is_male is None:
            # 返回一个代表“无法计算”的特定值，如-1，或直接返回None
            return None
        if scr <= 0 or age <= 0:
            return None

        scr_mg_dl = scr
        if scr_units.lower() in ['μmol/l', 'umol/l', 'mmol/l']:
            # 核心转换: 1 mg/dL = 88.4 μmol/L
            scr_mg_dl = scr / 88.4
        elif scr_units.lower() not in ['mg/dl']:
            # 如果输入了不支持的单位，可以抛出异常或返回None
            raise ValueError(f"不支持的肌酐单位: {scr_units}。请使用 'μmol/L' 或 'mg/dL'")

        # 1. 定义性别特定参数 [citation:2]
        if is_male:
            kappa = 0.9
            alpha = -0.302
            sex_factor = 1.000
        else:  # 女性
            kappa = 0.7
            alpha = -0.241
            sex_factor = 1.012

        # 2. 计算 Scr/kappa 比值
        scr_ratio = scr_mg_dl / kappa

        # 3. 应用分段幂计算 [citation:2]
        # min(Scr/κ, 1)^α 部分
        if scr_ratio <= 1:
            term1 = scr_ratio ** alpha
        else:
            term1 = 1.0  # 因为min(Scr/κ,1)=1，1^α=1

        # max(Scr/κ, 1)^-1.200 部分
        if scr_ratio >= 1:
            term2 = scr_ratio ** (-1.200)
        else:
            term2 = 1.0  # 因为max(Scr/κ,1)=1，1^-1.200=1

        # 4. 组合计算完整公式 [citation:2]
        egfr = 142 * term1 * term2 * (0.9938 ** age) * sex_factor

        # 5. 限制在临床合理范围
        # return min(max(egfr, 1.0), 200.0)
        # print(egfr)
        return egfr

    @staticmethod
    def calculate_ast_alt(ast_data, alt_data):
        # 提取日期和值的映射
        # 格式：{日期字符串: 检验项值}
        alt_dict = {}
        ast_dict = {}

        # 处理ALT数据
        for item in alt_data:
            receive_time = item.get('接收时间')
            value_str = item.get('检验项值')

            if receive_time and value_str:
                try:
                    # 提取日期部分（去掉时间）
                    date_str = receive_time.split()[0]  # 取空格前的部分
                    value = float(value_str)

                    # 如果同一天有多个记录，取第一个（最早的）
                    if date_str not in alt_dict:
                        alt_dict[date_str] = value
                except (ValueError, AttributeError, IndexError):
                    continue

        # 处理AST数据
        for item in ast_data:
            receive_time = item.get('接收时间')
            value_str = item.get('检验项值')

            if receive_time and value_str:
                try:
                    # 提取日期部分
                    date_str = receive_time.split()[0]
                    value = float(value_str)

                    # 如果同一天有多个记录，取第一个（最早的）
                    if date_str not in ast_dict:
                        ast_dict[date_str] = value
                except (ValueError, AttributeError, IndexError):
                    continue

        # 找出共有的日期并按日期排序
        common_dates = sorted(set(alt_dict.keys()) & set(ast_dict.keys()))

        # 如果没有共有的日期，返回None
        if not common_dates:
            return None

        # 取最早的共有日期
        earliest_date = common_dates[0]

        # 获取对应的值
        alt_value = alt_dict[earliest_date]
        ast_value = ast_dict[earliest_date]

        # 避免除零错误
        if alt_value == 0:
            return None

        # 计算比值
        ratio = ast_value / alt_value

        return ratio

    def _extract_diagnoses_based_on_metrics(self, eGFR, LDL, BMI, triglyceride) -> List[float]:
        """
        基于客观指标判定疾病状态，生成特征向量。
        返回一个包含5个元素的列表，分别对应：
        [肾病分期编码, 血脂异常标志, 肥胖分级编码, 脂质代谢异常标志, 并发症计数]
        """
        features = []

        if eGFR is None:
            features.append(None)
        else:
            # 1. 基于eGFR的肾病分期 (CKD G1-G5)
            ckd_stage = self._classify_ckd_by_egfr(eGFR)
            # 将分期G1-G5映射为有序数值 0-4 (或保留G1=1, G2=2...)
            # 这里将G1映射为0，G5映射为4，便于归一化到[0,1]
            stage_mapping = {'G1': 0.0, 'G2': 1.0, 'G3a': 2.0, 'G3b': 2.5, 'G4': 3.0, 'G5': 4.0}
            features.append(stage_mapping.get(ckd_stage, 0.0))
        self.update_feature_names("CKD-N期")

        # 2. 基于LDL-C的血脂异常标志
        if LDL is None:
            features.append(None)
        else:
            ldl_abnormal = self._check_ldl_abnormal(LDL)
            features.append(1.0 if ldl_abnormal else 0.0)
        self.update_feature_names("血脂异常")

        # 3. 基于BMI的肥胖分级
        if BMI is None:
            features.append(None)
        else:
            bmi_class = self._classify_obesity_by_bmi(BMI)
            # 将分类映射为数值：偏瘦=0, 正常=1, 超重=2, 中度肥胖=3，重度肥胖=4
            bmi_mapping = {'Underweight': 0.0, 'Normal': 1.0, 'Overweight': 2.0, 'Severe obesity': 3.0, 'Obese': 4.0, 'Unknown': -1.0}
            features.append(bmi_mapping.get(bmi_class, -1.0))
        self.update_feature_names("肥胖情况")
        #
        # 4. 基于甘油三酯的脂质代谢异常标志
        if triglyceride is None:
            features.append(None)
        else:
            tg_abnormal = self._check_tg_abnormal(triglyceride)
            features.append(1.0 if tg_abnormal else 0.0)
        self.update_feature_names("脂质代谢异常标志")

        return features

    @staticmethod
    def _classify_ckd_by_egfr(egfr) -> str:
        """基于eGFR进行CKD分期"""
        # 使用2021 CKD-EPI公式计算eGFR
        if egfr is None or egfr <= 0:
            return "Unknown"

        # KDIGO 2021临床实践指南分期标准
        if egfr >= 90:
            return "G1"
        elif egfr >= 60:
            return "G2"
        elif egfr >= 45:
            return "G3a"
        elif egfr >= 30:
            return "G3b"
        elif egfr >= 15:
            return "G4"
        else:
            return "G5"

    @staticmethod
    def _check_ldl_abnormal(ldl_value) -> bool:
        """检查LDL-C是否异常 (基于最新检验值)"""
        # lab_tests = patient_data.get('检验', {})
        # ldl_tests = lab_tests.get('LDL-C', [])
        #
        # if not ldl_tests:
        #     return False  # 无数据时默认正常
        #
        # # 取最早一次检验
        # latest_ldl_test = ldl_tests[0]
        # ldl_value = self._extract_numeric_value(latest_ldl_test.get('检验项值'))

        # 根据《中国血脂管理指南(2023)》，一般人群LDL-C理想水平应<3.4mmol/L
        if ldl_value is not None:
            return ldl_value > 3.4  # 大于3.4mmol/L视为异常

        return False

    @staticmethod
    def _classify_obesity_by_bmi(bmi) -> str:
        """基于BMI进行肥胖分级 (中国标准)"""
        if not bmi:
            return "Unknown"

        try:
            # 《中国成人超重和肥胖症预防控制指南(2021)》标准
            if bmi < 18.5:
                return "Underweight"
            elif 18.5 <= bmi < 24:
                return "Normal"
            elif 24 <= bmi < 28:
                return "Overweight"
            elif 28 <= bmi < 37.5:  # 重度肥胖正
                return "Severe obesity"
            else:  # bmi >= 28
                return "Obese"
        except:
            return "Unknown"

    @staticmethod
    def _check_tg_abnormal(tg_value) -> bool:
        """检查甘油三酯是否异常"""

        # 根据《中国血脂管理指南(2023)》，空腹TG≥1.7mmol/L为升高
        if tg_value is not None:
            return tg_value >= 1.7

        return False

    def _extract_diagnoses(self, patient_data: Dict[str, Any], str_id: str) -> List[float]:
        """提取诊断特征（多标签编码）"""
        # 常见糖尿病并发症列表
        common_complications = [
            "糖尿病性周围神经病变",
            "糖尿病性视网膜病变",
            "糖尿病性周围血管病变",
            "糖尿病性肾病",
            "高血压",
            "冠状动脉粥样硬化性心脏病",
            "心力衰竭",
            "失代偿性心力衰竭",
            "慢性肾脏病",
            "肾衰竭",
            "肾透析",
            "肾结石",
            "蛋白尿",
            "高脂血症",
            "脂肪肝",
            "代谢综合征",
            "胃轻瘫",
            "胃炎",
            "胰腺炎",
            "胃肠道不良反应",
            "前列腺增生",
            "前列腺炎",
            "骨质疏松症",
            "肝功能不全",
            "酮症",
            "酮症酸中毒",
            "高渗高血糖综合征",
            "甲状腺髓样癌",
            "多发性内分泌腺瘤病2型"
        ]

        diagnoses = patient_data.get('诊断', {})
        if not isinstance(diagnoses, dict):
            diagnoses = {}

        features = []
        for comp in common_complications:
            raw = diagnoses.get(comp, 'no')
            # 处理 null 值：直接保留为 NaN
            if raw is None:
                if comp == "糖尿病性周围血管病变":
                    raw = self._extract_numeric_value(
                        self._safe_dict(self.extra_features.get(str_id)).get("糖尿病周围血管病")
                    )
                else:
                    features.append(None)
                    continue
            if comp == "糖尿病性周围血管病变":
                severity = self._extract_numeric_value(
                    self._safe_dict(self.extra_features.get(str_id)).get("糖尿病周围血管病")
                )
                features.append(1.0 if severity is not None and severity > 0 else 0.0)
                continue

            if comp == "糖尿病性肾病":
                diag = self.diabetic_nephropathy_to_risk(raw)
            elif comp == "高血压":
                diag = self.extract_hypertension_grade(raw)
            else:
                diag = raw

            if diag == "no":
                features.append(0.0)
            elif diag == 'yes':
                features.append(1.0)
            else:
                features.append(diag)

        # 特征名称记录
        for diag in common_complications:
            self.update_feature_names(f"complications-{diag}")

        return features

    def _extract_extra_features(self, str_id: str) -> List[float]:
        record = self._safe_dict(self.extra_features.get(str_id))
        dpvd_grade = self._extract_numeric_value(record.get("糖尿病周围血管病"))
        hydronephrosis = self._extract_numeric_value(record.get("肾积水"))
        hyperuricemia = self._extract_numeric_value(record.get("高尿酸血症"))
        first_fasting = self._extract_numeric_value(record.get("首次空腹血糖"))

        features = [
            dpvd_grade if dpvd_grade is not None else None,
            hydronephrosis if hydronephrosis is not None else None,
            hyperuricemia if hyperuricemia is not None else None,
            self._normalize_lab_value(first_fasting, 3.9, 6.1) if first_fasting is not None else None,
        ]
        self.update_feature_names("糖尿病周围血管病分级")
        self.update_feature_names("肾积水")
        self.update_feature_names("complications-高尿酸血症")
        self.update_feature_names("首次空腹血糖")
        return features

    @staticmethod
    def extract_hypertension_grade(value):
        """
        将高血压等级描述转换为有序数值
        """
        if value is None or value == '' or value == 'no' or value == 0:
            return 0.0

        # 若输入已经是数值
        if isinstance(value, (int, float)):
            return float(value)

        s = str(value).strip()

        # 匹配“X级”模式
        import re
        match = re.search(r'(\d+)级', s)
        if match:
            grade = int(match.group(1))
            # 一般等级为1,2,3，超过则取min(3, grade)
            return min(float(grade), 3.0)

        # 默认无高血压
        return 0.0

    @classmethod
    def blood_pressure_grade_from_first_course(cls, patient_data: Dict[str, Any]) -> Optional[float]:
        first_course = cls._safe_dict(patient_data.get('首次病程'))
        systolic = cls._extract_numeric_value(cls._safe_dict(first_course.get('收缩压')).get('value'))
        diastolic = cls._extract_numeric_value(cls._safe_dict(first_course.get('舒张压')).get('value'))
        if systolic is None or diastolic is None:
            return None

        if systolic < 120 and diastolic < 80:
            return 0.0
        if systolic < 140 and diastolic < 90:
            return 1.0
        if systolic >= 140 and diastolic < 90:
            return 2.0
        if systolic < 140 and diastolic >= 90:
            return 2.0
        if systolic >= 180 or diastolic >= 110:
            return 4.0
        if systolic >= 160 or diastolic >= 100:
            return 3.0
        return 2.0

    @staticmethod
    def _extract_numeric_value(value: Any) -> Optional[float]:
        """从数据中提取数值"""
        if value is None:
            return None

        if isinstance(value, dict):
            value = value.get('value')

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                # 尝试处理带有单位的值
                for char in value:
                    if char.isdigit() or char == '.':
                        continue
                    value = value.replace(char, '')
                try:
                    return float(value) if value else None
                except:
                    return None

        return None

    @staticmethod
    def extract_urine_leukocyte_value(value: Any) -> Optional[float]:
        """将尿白细胞分级映射为数值。"""
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None

        normalized = value.strip()
        mapping = {
            "-": 0.0,
            "+-": 0.5,
            "1+": 1.0,
            "2+": 2.0,
            "3+": 3.0,
        }
        return mapping.get(normalized)

    @staticmethod
    def _normalize_lab_value(value: Optional[float], low: float, high: float) -> float:
        """归一化实验室数值"""
        if value is None:
            return None

        # 使用sigmoid-like归一化，将正常范围映射到0.5附近
        normalized = (value - low) / (high - low)
        # 使用sigmoid函数平滑
        sigmoid_normalized = 1 / (1 + np.exp(-(normalized - 0.5) * 10))
        return float(sigmoid_normalized)

    def normalize_vectors(self, patient_vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        对所有向量进行归一化（Z-score标准化）
        假设不会有某列全部为NaN，否则抛出异常
        """
        if not patient_vectors:
            return {}

        # 转换为矩阵
        vectors = np.array(list(patient_vectors.values()))

        # 检查是否有整列都是NaN的情况
        nan_mask = np.isnan(vectors)
        if nan_mask.any():
            # 检查是否有某列全部为NaN
            all_nan_cols = np.all(nan_mask, axis=0)
            if all_nan_cols.any():
                nan_col_indices = np.where(all_nan_cols)[0]
                dropped_feature_names = [self.get_feature_names()[i] for i in nan_col_indices]
                print(
                    f"Dropping all-NaN feature columns: "
                    f"indices={nan_col_indices.tolist()}, names={dropped_feature_names}"
                )
                raise ValueError(
                    f"发现 {len(nan_col_indices)} 列全部为NaN，列索引: {nan_col_indices.tolist()}。{[self.get_feature_names()[i] for i in nan_col_indices]}"
                    "请检查数据完整性。"
                )

        # 使用nanmean和nanstd忽略NaN值计算统计量
        mean = np.nanmean(vectors, axis=0)
        std = np.nanstd(vectors, axis=0)

        # 避免除零：标准差为0的列设为1（假设常数列）
        std = np.where(std == 0, 1, std)

        # 标准化
        normalized_vectors = (vectors - mean) / std

        # 转换回字典
        result = {}
        patient_ids = list(patient_vectors.keys())
        for i, patient_id in enumerate(patient_ids):
            result[patient_id] = normalized_vectors[i]

        return result

    def create_patient_personality_vectors(self, patient_dict: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """为所有患者创建个性向量"""
        patient_vectors = {}

        for patient_id, patient_data in patient_dict.items():
            try:
                vector = self.extract_patient_features(patient_data, patient_id)
                patient_vectors[patient_id] = vector
            except Exception as e:
                print(f"Error processing patient {patient_id}: {e}")
                continue
        # print(patient_vectors)
        return patient_vectors, self.get_feature_names()


# 使用示例
if __name__ == "__main__":
    import torch

    # 可选：使用PCA降维可视化
    def reduce_dimensionality(vectors_dict: Dict[str, np.ndarray], n_components: int = 3):
        """使用PCA降维"""
        from sklearn.decomposition import PCA

        vectors = np.array(list(vectors_dict.values()))
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(vectors)

        result = {}
        patient_ids = list(vectors_dict.keys())
        for i, patient_id in enumerate(patient_ids):
            result[patient_id] = reduced[i]

        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2%}")

        return result


    def save_vectors_to_csv(patient_vectors: Dict[str, np.ndarray], feature_names: List[str], output_path: str):
        """将向量保存为CSV文件"""
        df_data = []

        for patient_id, vector in patient_vectors.items():
            row = {'patient_id': patient_id}
            row.update({feature_names[i]: vector[i] for i in range(len(vector))})
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} patient vectors to {output_path}")
        return df


    # 假设字典名为 patient_data_dict
    with open("./merged_v3.json", 'r', encoding='utf-8') as fr:
        patient_data_dict = json.load(fr)
    with open("../model_data/preprocessed/四针/stage_1_only_insulin_output_file_basic.json", 'r', encoding='utf-8') as ffr:
        xx = json.load(ffr)
    p_data_dict = {}
    for k, v in xx.items():
        if k in patient_data_dict:
            p_data_dict[k] = patient_data_dict[k]

    patient_data_dict = p_data_dict
    # print(len(patient_data_dict))

    # patient_data_dict = {...}

    # 1. 创建特征提取器
    extractor = PatientFeatureExtractor()

    # 2. 提取所有患者的特征向量
    patient_vectors, feature_names = extractor.create_patient_personality_vectors(patient_data_dict)

    print(f"Extracted vectors for {len(patient_vectors)} patients")
    print(f"Vector dimension: {len(next(iter(patient_vectors.values())))}")

    # 3. 获取特征名称
    print(f"Feature names: {feature_names}")
    # for k, v in patient_vectors.items():
    #     print(v[13:20])
    # 4. 对向量进行标准化
    # print(patient_vectors)
    normalized_vectors = extractor.normalize_vectors(patient_vectors)
    # print(normalized_vectors)
    tensor_vectors = []
    for k, v in normalized_vectors.items():
        tensor_vector = torch.from_numpy(v).float()
        # print(tensor_vector.shape)
        # print(tensor_vector)
        tensor_vectors.append(tensor_vector)
    stacked_vectors = torch.stack(tensor_vectors, dim=0)
    # print(stacked_vectors.shape)
    # 5. 保存为CSV
    if normalized_vectors:
        df = save_vectors_to_csv(normalized_vectors, feature_names, "patient_personality_vectors.csv")

        # 查看前几个患者
        print("\nFirst few patient vectors:")
        for patient_id, vector in list(normalized_vectors.items())[:30]:
            print(f"\n{patient_id}:")
            print(f"  Vector shape: {vector.shape}")
            print(vector[20:60])
            print(f"  Mean: {vector.mean():.4f}, Std: {vector.std():.4f}")
            print(f"  Min: {vector.min():.4f}, Max: {vector.max():.4f}")
