# -*- encoding: utf-8 -*-
"""Rule-based medication recommendation logic with trace output."""


from typing import Dict, List, Set, Optional, FrozenSet, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy
from collections import defaultdict
from datetime import datetime


class DrugCategory(Enum):
    """药物类别"""
    MET = "二甲双胍"
    SGLT2I1 = "SGLT2抑制剂-达格列净"
    SGLT2I2 = "SGLT2抑制剂-恩格列净"
    SGLT2I3 = "SGLT2抑制剂-卡格列净"
    SGLT2I4 = "SGLT2抑制剂-艾托格列净"
    SGLT2I5 = "SGLT2抑制剂-恒格列净"
    SGLT2I6 = "SGLT2抑制剂-加格列净"
    GLP1RA = "GLP-1受体激动剂(贝那鲁肽/艾塞那肽/利司那肽/利拉鲁肽/司美格鲁肽)"
    DPP4I1 = "DPP4抑制剂-西格列汀"
    DPP4I2 = "DPP4抑制剂-沙格列汀"
    DPP4I3 = "DPP4抑制剂-维格列汀"
    DPP4I4 = "DPP4抑制剂-利格列汀"
    DPP4I5 = "DPP4抑制剂-阿格列汀"
    DPP4I6 = "DPP4抑制剂-瑞格列汀"
    DPP4I7 = "DPP4抑制剂-考格列汀"
    SU1 = "磺脲类-格列本脲"
    SU2 = "磺脲类-格列吡嗪"
    SU3 = "磺脲类-格列齐特"
    SU4 = "磺脲类-格列喹酮"
    SU5 = "磺脲类-格列美脲"
    GLN = "格列奈类"
    AGI = "α-糖苷酶抑制剂(阿卡波糖,伏格列波糖,米格列醇,桑枝总生物碱片)"
    TZD1 = "噻唑烷二酮类-吡格列酮"
    TZD2 = "噻唑烷二酮类-罗格列酮"
    PAN_PPARA = "PPAR全激动剂-西格列他那纳"
    GKA = "葡萄糖激酶激活剂(多格列艾汀)"


SU_DRUG_CATEGORIES = {
    DrugCategory.SU1,
    DrugCategory.SU2,
    DrugCategory.SU3,
    DrugCategory.SU4,
    DrugCategory.SU5,
}


@dataclass
class Patient:
    """
    患者信息

    delta_blood_glucose_after_meal: float

    complications: Set[str]:
    白蛋白尿 "HF" "CKD" "ASCVD" "失代偿性HF" "酮症、酮症酸中毒、高渗透性高HHS"
    masld脂肪肝  高脂血症 高血压 胃轻瘫/胃炎 胆囊炎 胆石症 视神经炎 胰腺炎

    risk_factors: Set[str]:
    低血糖事件hypoglycemia 高血压、血脂异常 胃肠道不良反应gastrointestinal
    "肝功能不全" "肾功能不全" "骨质疏松" "前列腺增生" "生酮饮食"

    history: Set[str]:
    心衰史 "骨折史" 多发性内分泌腺瘤病2型 甲状腺髓样癌 卒中病史"

    family_history: Set[str] = None"
    多发性内分泌腺瘤病2型 甲状腺髓样癌

    """
    age: int
    bmi: float
    hba1c: float
    duration: float
    C_peptide_half: float
    C_peptide_2: float
    C_peptide_3: float
    systolic_blood_pressure: float
    diastolic_blood_pressure: float
    Elevated_liver: float
    diabetes_type: int
    delta_blood_glucose_after_meal: float
    complications: Set[str]
    risk_factors: Set[str]
    history: Set[str]
    blood_glucose: Dict[str, float] = None
    family_history: Set[str] = None
    ALT: float = None
    AST: float = None
    UACR: float = None
    UAER: float = None
    renal_function: Optional[float] = None
    hypoglycemia_risk: bool = False
    beta_cell_function_low: Optional[bool] = None


class Condition:
    """条件类"""

    def __init__(self, name: str, description: str, check_function: Callable[[Patient], bool]):
        self.name = name
        self.description = description
        self.check_function = check_function

    def check(self, patient: Patient) -> bool:
        return self.check_function(patient)


@dataclass
class DrugRule:
    """新的药物规则类 - 移除ConditionType，改为四个独立的药物类别集合"""
    rule_id: str
    description: str
    required_conditions: Set[str] = None
    excluded_conditions: Set[str] = None
    recommend_drug_categories: Set[DrugCategory] = None  # 推荐药物
    caution_drug_categories: Set[DrugCategory] = None  # 慎用药物（最终视为可用）
    consider_drug_categories: Set[DrugCategory] = None  # 考虑药物
    forbid_drug_categories: Set[DrugCategory] = None  # 禁用药物
    type: int = -1  # 0 第0层 1 第一层 并发症 2其他病 3 药物 4 症状

    def __post_init__(self):
        if self.required_conditions is None:
            self.required_conditions = set()
        if self.excluded_conditions is None:
            self.excluded_conditions = set()
        if self.recommend_drug_categories is None:
            self.recommend_drug_categories = set()
        if self.caution_drug_categories is None:
            self.caution_drug_categories = set()
        if self.consider_drug_categories is None:
            self.consider_drug_categories = set()
        if self.forbid_drug_categories is None:
            self.forbid_drug_categories = set()


@dataclass(frozen=True)
class DrugRecommendation:
    """药物推荐结果（不可变，可哈希），记录每个药物命中过的规则来源。"""
    drug: DrugCategory
    recommend_rules: FrozenSet[str] = field(default_factory=frozenset)  # 推荐规则
    consider_rules: FrozenSet[str] = field(default_factory=frozenset)   # 可考虑/弱推荐规则
    caution_rules: FrozenSet[str] = field(default_factory=frozenset)    # 谨慎使用规则
    forbid_rules: FrozenSet[str] = field(default_factory=frozenset)     # 禁止使用规则

    def add_recommend_rule(self, rule_id: str) -> 'DrugRecommendation':
        """添加推荐规则，返回新的DrugRecommendation"""
        return DrugRecommendation(
            drug=self.drug,
            recommend_rules=self.recommend_rules.union({rule_id}),
            consider_rules=self.consider_rules,
            caution_rules=self.caution_rules,
            forbid_rules=self.forbid_rules
        )

    def add_consider_rule(self, rule_id: str) -> 'DrugRecommendation':
        """添加可考虑/弱推荐规则，返回新的DrugRecommendation"""
        return DrugRecommendation(
            drug=self.drug,
            recommend_rules=self.recommend_rules,
            consider_rules=self.consider_rules.union({rule_id}),
            caution_rules=self.caution_rules,
            forbid_rules=self.forbid_rules
        )

    def add_caution_rule(self, rule_id: str) -> 'DrugRecommendation':
        """添加谨慎使用规则，返回新的DrugRecommendation"""
        return DrugRecommendation(
            drug=self.drug,
            recommend_rules=self.recommend_rules,
            consider_rules=self.consider_rules,
            caution_rules=self.caution_rules.union({rule_id}),
            forbid_rules=self.forbid_rules
        )

    def add_forbid_rule(self, rule_id: str) -> 'DrugRecommendation':
        """添加禁用规则，返回新的DrugRecommendation"""
        return DrugRecommendation(
            drug=self.drug,
            recommend_rules=self.recommend_rules,
            consider_rules=self.consider_rules,
            caution_rules=self.caution_rules,
            forbid_rules=self.forbid_rules.union({rule_id})
        )


class DiabetesPrescriptionSystemRefactored:

    @staticmethod
    def _positive_below(value, threshold: float) -> bool:
        try:
            numeric = float(value)
            return 0 < numeric < threshold
        except (TypeError, ValueError):
            return False

    def __init__(self):
        self.conditions: Dict[str, Condition] = {}
        self.rules: List[DrugRule] = []
        self._initialize_builtin_conditions()
        self._initialize_rules()
        self.drug_categories = {
            "SGLT2抑制剂": ["SGLT2抑制剂-达格列净", "SGLT2抑制剂-恩格列净", "SGLT2抑制剂-卡格列净",
                         "SGLT2抑制剂-艾托格列净", "SGLT2抑制剂-恒格列净", "SGLT2抑制剂-加格列净"],
            "DPP4抑制剂": ["DPP4抑制剂-西格列汀", "DPP4抑制剂-沙格列汀", "DPP4抑制剂-维格列汀",
                        "DPP4抑制剂-利格列汀", "DPP4抑制剂-阿格列汀", "DPP4抑制剂-瑞格列汀",
                        "DPP4抑制剂-考格列汀"],
            "磺脲类": ["磺脲类-格列本脲", "磺脲类-格列吡嗪", "磺脲类-格列齐特",
                    "磺脲类-格列喹酮", "磺脲类-格列美脲"],
            "噻唑烷二酮类": ["噻唑烷二酮类-吡格列酮", "噻唑烷二酮类-罗格列酮"]
        }

    def _initialize_builtin_conditions(self):
        """初始化所有条件"""
        # 体重相关条件
        self.conditions["overweight_obesity"] = Condition(
            "overweight_obesity", "超重/肥胖 (BMI ≥ 24)",
            lambda p: p.bmi >= 24
        )
        self.conditions["normal_weight"] = Condition(
            "normal_weight", "正常体重 (18.5< BMI < 24)",
            lambda p: p.bmi < 24 and p.bmi >= 18.5
        )
        self.conditions["lack_weight"] = Condition(
            "normal_weight", "偏瘦 (16 > BMI)",
            lambda p: p.bmi < 16
        )

        # 年龄相关条件
        self.conditions["age_70_plus"] = Condition(
            "age_70_plus", "老年患者 (年龄 ≥ 70岁)",
            lambda p: p.age >= 70
        )
        self.conditions["age_under_40"] = Condition(
            "age_under_40", "年轻患者 (年龄 < 40岁)",
            lambda p: 18 <= p.age < 40
        )
        self.conditions["children_under_18"] = Condition(
            "children_under_18", "青少年患者 (年龄 < 18岁)",
            lambda p: 10 <= p.age < 18
        )
        self.conditions["children_under_10"] = Condition(
            "children_under_10", "儿童患者 (年龄 < 10岁)",
            lambda p: p.age < 10
        )

        # 血糖相关条件
        self.conditions["hba1c_high"] = Condition(
            "hba1c_high", "高HbA1c (≥ 7.5%)",
            lambda p: p.hba1c >= 7.5
        )
        self.conditions["hba1c_very_high"] = Condition(
            "hba1c_very_high", "极高HbA1c (≥ 9.0%)",
            lambda p: p.hba1c >= 9.0
        )
        self.conditions["hba1c_low"] = Condition(
            "hba1c_low", "HbA1c (<3.9%)",
            lambda p: p.hba1c <= 3.9
        )

        # 并发症相关条件
        self.conditions["ascvd_present"] = Condition(
            "ascvd_present", "合并ASCVD",
            lambda p: "ASCVD" in p.complications
        )

        self.conditions["age_55_plus_2_risk_factors"] = Condition(
            "age_55_plus_2_risk_factors",
            "年龄>55岁且≥2个危险因素（吸烟、肥胖、高血压、血脂异常、白蛋白尿）",
            lambda p: p.age > 55 and sum([
                "smoking" in p.risk_factors,
                p.bmi >= 24,
                # 高血压、血脂异常、白蛋白尿
                "hypertension" in p.risk_factors or "hypertension" in p.complications,
                "dyslipidemia" in p.risk_factors or "dyslipidemia" in p.complications,
                "albuminuria" in p.complications
            ]) >= 2
        )
        # TODO 超声心动图结果
        self.conditions["hf_present"] = Condition(
            "hf_present", "合并心衰(HF), 当前或既往有HF症状 且有确证的指标（超声心动图结果）",
            lambda p: "HF" in p.complications
        )

        self.conditions["decompensated_HF"] = Condition(
            "decompensated_HF", "失代偿性HF",
            lambda p: "失代偿性HF" in p.complications
        )

        self.conditions["ckd_present"] = Condition(
            "ckd_present", "合并CKD",
            lambda p: "CKD" in p.complications or p.renal_function < 60
                      or (p.UACR is not None and p.UACR >= 30) or (p.UAER is not None and p.UAER >= 30)
        )

        self.conditions["MASLD_present"] = Condition(
            "MASLD_present", "合并MASLD脂肪肝",
            lambda p: "脂肪肝" in p.complications
        )

        self.conditions["Hyperlipidemia_present"] = Condition(
            "Hyperlipidemia_present", "合并Hyperlipidemia高脂血症",
            lambda p: "高脂血症" in p.complications
        )

        self.conditions["hypertension_present"] = Condition(
            "hypertension_present", "合并hypertension高血压",
            lambda p: "高血压" in p.complications or p.systolic_blood_pressure > 140 or p.diastolic_blood_pressure > 90
        )

        self.conditions["gastritis_present"] = Condition(
            "gastritis_present", "合并胃轻瘫/ 胃炎",
            lambda p: "胃轻瘫" in p.complications or "胃炎" in p.complications
        )

        self.conditions["eye_present"] = Condition(
            "eye_present", "合并糖尿病眼病",
            lambda p: "眼病" in p.complications
        )

        self.conditions["metabolic_syndrome_present"] = Condition(
            "metabolic_syndrome_present", "合并代谢综合征",
            lambda p: "代谢综合征" in p.complications
        )

        # SU or GLN
        # TODO C肽结构需要确认（β细胞功能较好）
        self.conditions["su_gln_met_candidate"] = Condition(
            "su_gln_met_candidate",
            "年轻、初诊HbA1c较高、胰岛β细胞功能较好、不伴超重或肥胖患者",
            lambda p: p.age < 40 and p.hba1c > 8.0 and
                      1890 > p.C_peptide_2 > 300 and
                      p.bmi < 24
        )
        # TODO
        self.conditions["secretagogue_candidate_general"] = Condition(
            "secretagogue_candidate_general",
            "促泌剂一般候选人群：字段 diabetes_type=2；HbA1c>=8.0；C_peptide_2>=600 且非 beta_cell_function_low；eGFR>=60；age<70；无 hypoglycemia 风险；BMI<28；无 ASCVD/HF/CKD 优先保护需求。解释：胰岛功能尚可、低血糖风险较低且仍需进一步降糖时，可考虑 SU/GLN。",
            lambda p: p.diabetes_type == 2 and
                      p.hba1c is not None and p.hba1c >= 8.0 and
                      p.C_peptide_2 is not None and p.C_peptide_2 >= 1000 and
                      not bool(p.beta_cell_function_low) and
                      p.renal_function is not None and p.renal_function >= 60 and
                      p.age < 70 and
                      not p.hypoglycemia_risk and
                      "hypoglycemia" not in p.risk_factors and
                      p.bmi < 28 and
                      "ASCVD" not in p.complications and
                      "HF" not in p.complications and
                      "CKD" not in p.complications
        )
        self.conditions["gln_postprandial_preferred"] = Condition(
            "gln_postprandial_preferred",
            "GLN餐后血糖优先：字段 diabetes_type=2；HbA1c>=8.0；delta_blood_glucose_after_meal>=4；C_peptide_2>=600；eGFR>=45；BMI<28；无严重肝功能不全。解释：GLN起效快、作用短，更适合餐后高血糖突出或餐时降糖需求。",
            lambda p: p.diabetes_type == 2 and
                      p.hba1c is not None and p.hba1c >= 8.0 and
                      p.delta_blood_glucose_after_meal is not None and p.delta_blood_glucose_after_meal >= 6 and
                      p.C_peptide_2 is not None and p.C_peptide_2 >= 1000 and
                      p.renal_function is not None and p.renal_function >= 60 and
                      p.age < 70 and
                      not p.hypoglycemia_risk and
                      "hypoglycemia" not in p.risk_factors and
                      p.bmi < 28 and
                      "ASCVD" not in p.complications and
                      "HF" not in p.complications and
                      "CKD" not in p.complications and
                      "严重肝功能不全" not in p.complications and
                      "肝功能不全" not in p.risk_factors
        )
        self.conditions["su_stronger_hba1c_lowering"] = Condition(
            "su_stronger_hba1c_lowering",
            "SU strict early severe candidate: diabetes_type=2; HbA1c>=9.0; duration<0.5 year; 150<=C_peptide_half<300; C_peptide_2>=1000; eGFR>=60; age<60; BMI<24; no hypoglycemia risk; no ASCVD/HF/CKD. Explanation: reserve strong SU recommendation for early, non-obese, severe hyperglycemia with preserved stimulated beta-cell response.",
            lambda p: p.diabetes_type == 2 and
                      p.hba1c is not None and p.hba1c >= 9.0 and
                      p.duration is not None and p.duration < 0.5 and
                      p.C_peptide_half is not None and 150 <= p.C_peptide_half < 300 and
                      p.C_peptide_2 is not None and p.C_peptide_2 >= 600 and
                      p.renal_function is not None and p.renal_function >= 60 and
                      p.age < 60 and
                      not p.hypoglycemia_risk and
                      "hypoglycemia" not in p.risk_factors and
                      p.bmi < 24 and
                      "ASCVD" not in p.complications and
                      "HF" not in p.complications and
                      "CKD" not in p.complications
        )
        self.conditions["su_preserved_beta_non_masld_candidate"] = Condition(
            "su_preserved_beta_non_masld_candidate",
            "SU preserved beta non-MASLD candidate: diabetes_type=2; HbA1c>=8.5; duration<=2 years; C_peptide_half>=300; C_peptide_2>=1200; eGFR>=60; age<55; BMI<24; no hypoglycemia risk; no MASLD/ASCVD/HF/CKD. Explanation: a narrower SU rule for non-obese patients with preserved beta-cell function and moderate-to-severe hyperglycemia.",
            lambda p: p.diabetes_type == 2 and
                      p.hba1c is not None and p.hba1c >= 8.5 and
                      p.duration is not None and p.duration <= 2 and
                      p.C_peptide_half is not None and p.C_peptide_half >= 300 and
                      p.C_peptide_2 is not None and p.C_peptide_2 >= 1200 and
                      p.renal_function is not None and p.renal_function >= 60 and
                      p.age < 55 and
                      not p.hypoglycemia_risk and
                      "hypoglycemia" not in p.risk_factors and
                      p.bmi < 24 and
                      "脂肪肝" not in p.complications and
                      "ASCVD" not in p.complications and
                      "HF" not in p.complications and
                      "CKD" not in p.complications
        )
        self.conditions["ckd_secretagogue_limited"] = Condition(
            "ckd_secretagogue_limited",
            "CKD促泌剂限制使用：字段 diabetes_type=2；30<=eGFR<60；C_peptide_2>=600；无低血糖风险。解释：肾功能下降时不泛推荐全部SU，仅考虑GLN或格列喹酮(SU4)。",
            lambda p: p.diabetes_type == 2 and
                      p.renal_function is not None and 30 <= p.renal_function < 60 and
                      p.C_peptide_2 is not None and p.C_peptide_2 >= 600 and
                      not p.hypoglycemia_risk and
                      "hypoglycemia" not in p.risk_factors
        )
        self.conditions["secretagogue_caution_or_forbid"] = Condition(
            "secretagogue_caution_or_forbid",
            "促泌剂慎用/禁用：字段 beta_cell_function_low、age、hypoglycemia、eGFR、肝功能、BMI。解释：胰岛功能差、老年/低血糖风险、eGFR<30、明显肝功能不全或肥胖减重需求时，SU/GLN低血糖或体重风险升高。",
            lambda p: bool(p.beta_cell_function_low) or
                      p.age >= 70 or
                      p.hypoglycemia_risk or
                      "hypoglycemia" in p.risk_factors or
                      (p.renal_function is not None and p.renal_function < 30) or
                      "严重肝功能不全" in p.complications or
                      "肝功能不全" in p.risk_factors or
                      p.bmi >= 28
        )
        self.conditions["beta_cell_function"] = Condition(
            "beta_cell_function",
            "胰岛β细胞功能较好",
            lambda p: 1890 > p.C_peptide_2 > 300
        )
        self.conditions["beta_cell_function_low"] = Condition(
            "beta_cell_function_low",
            "胰岛功能差：空腹C肽<300或OGTT餐后2小时C肽<600",
            lambda p: bool(p.beta_cell_function_low)
            if p.beta_cell_function_low is not None
            else (
                self._positive_below(p.C_peptide_half, 300)
                and self._positive_below(p.C_peptide_2, 600)
            )
        )

        # 阿卡波糖
        self.conditions["AGi_blood_glucose"] = Condition(
            "AGi_blood_glucose",
            "轻度减轻体重",
            lambda p: p.delta_blood_glucose_after_meal is not None and p.delta_blood_glucose_after_meal > 6
        )

        self.conditions["insulin_resistance"] = Condition(
            "insulin_resistance", "明显胰岛素抵抗 （胰岛素抵抗相关临床问题专家共识）",
            lambda p: p.C_peptide_2 > 1890
        )

        self.conditions["elderly_high_risk_gi"] = Condition(
            "elderly_high_risk_gi",
            "老年、低血糖风险高[低血糖事件风险，CKD 3B-5(吴批注)]、胃肠道不良反应明显患者",
            lambda p: p.age >= 70 and
                      ("hypoglycemia" in p.risk_factors or p.renal_function < 45 or
                       "gastrointestinal" in p.risk_factors)
        )

        self.conditions["hypoglycemia_high_risk"] = Condition(
            "hypoglycemia_high_risk",
            "低血糖风险高[低血糖事件风险]",
            lambda p: "hypoglycemia" in p.risk_factors
        )

        self.conditions["Gastrointestinal_adverse_reactions"] = Condition(
            "Gastrointestinal_adverse_reactions",
            "胃肠道不良反应明显患者",
            lambda p: "gastrointestinal" in p.risk_factors
        )

        self.conditions["short_duration_good_beta"] = Condition(
            "short_duration_good_beta",
            "病程较短、胰岛β细胞功能较好患者",
            lambda p: p.duration < 1 and
                      1890 > p.C_peptide_2 > 300
        )
        self.conditions["gka_recall_prioritized"] = Condition(
            "gka_recall_prioritized",
            "GKA recall-prioritized screening: insulin-resistance context, long diabetes duration with preserved stimulated C-peptide and BMI<28, or diabetic eye disease with preserved stimulated C-peptide.",
            lambda p: (
                p.C_peptide_2 is not None and p.C_peptide_2 > 1890
            ) or (
                p.duration is not None and p.duration >= 10
                and p.C_peptide_2 is not None and p.C_peptide_2 >= 1000
                and p.bmi is not None and p.bmi < 28
            ) or (
                p.C_peptide_2 is not None and p.C_peptide_2 >= 1000
                and "眼病" in p.complications
            )
        )
        # 肝功能不全 肾功能不全
        self.conditions["SU_CAUTION"] = Condition(
            "SU_CAUTION",
            "老年、肝肾功能不全",
            lambda p: p.age >= 70 and
                      ("肝功能不全" in p.risk_factors or "肾功能不全" in p.risk_factors)
        )

        self.conditions["HF_CAUTION"] = Condition(
            "HF_CAUTION",
            "HF病史和HF诱发因素的患者",
            lambda p: "心衰史" in p.history or "HF" in p.complications
        )

        self.conditions["Fracture_FORBID"] = Condition(
            "Fracture_FORBID",
            "有严重骨质疏松或近期骨折病史的人",
            lambda p: "骨质疏松" in p.risk_factors
        )

        self.conditions["Urogenital_CAUTION"] = Condition(
            "Urogenital_CAUTION",
            "泌尿生殖系感染风险增加，应注意个人外阴部卫生适当增加饮水量",
            lambda p: "前列腺增生" in p.risk_factors or "肾结石" in p.complications or "前列腺炎" in p.complications
                      or "泌尿系统感染" in p.risk_factors
        )

        # 高的血压<90
        self.conditions["Insufficient_blood_volume"] = Condition(
            "Insufficient_blood_volume",
            "血容量不足",
            lambda p: p.systolic_blood_pressure < 90
        )

        self.conditions["Ketogenic_diet_CAUTION"] = Condition(
            "Ketogenic_diet_CAUTION",
            "生酮饮食 SGLT2i增加酮症酸中毒风险",
            lambda p: "生酮饮食" in p.risk_factors
        )

        # 肾功能条件
        self.conditions["egfr_3a"] = Condition(
            "egfr_3a", "3a期 (eGFR 45-59)",
            lambda p: p.renal_function is not None and 45 <= p.renal_function < 60
        )
        self.conditions["egfr_3b"] = Condition(
            "egfr_3b", "3b (eGFR 30-45)",
            lambda p: p.renal_function is not None and 30 <= p.renal_function < 45
        )
        self.conditions["egfr_4"] = Condition(
            "egfr_4", "4期 (eGFR 15-29)",
            lambda p: p.renal_function is not None and 15 <= p.renal_function < 30
        )
        self.conditions["egfr_5"] = Condition(
            "egfr_5", "5期 (eGFR < 15)",
            lambda p: p.renal_function is not None and p.renal_function < 15
        )

        self.conditions["GLP-1RA_FORBID"] = Condition(
            "GLP-1RA_FORBID", "甲状腺髓样癌，多发性内分泌腺瘤病2型既往史或家族史，胰腺炎(史)也禁用GLP-1RA",
            lambda p: "甲状腺髓样癌" in p.risk_factors or (p.family_history is not None and "甲状腺髓样癌" in p.family_history) or "甲状腺髓样癌" in p.history or
                      "多发性内分泌腺瘤病2型" in p.history or (p.family_history is not None and "多发性内分泌腺瘤病2型" in p.family_history)
                      or "多发性内分泌腺瘤病2型" in p.risk_factors or "胰腺炎" in p.complications or "胰腺炎" in p.history
        )

        self.conditions["stroke_history"] = Condition(
            "stroke_history", "卒中病史",
            lambda p: "卒中病史" in p.history
        )

        self.conditions["Elevated_liver_enzymes_slightly"] = Condition(
            "Elevated_liver_enzymes_slightly", "轻度肝酶升高",
            lambda p: 1 < p.Elevated_liver <= 3 or (p.ALT is not None and p.AST is not None and 40 < p.ALT / p.AST <= 120)
        )
        self.conditions["Elevated_liver_enzymes_Moderate"] = Condition(
            "Elevated_liver_enzymes_Moderate", "中度肝酶升高",
            lambda p: 3 < p.Elevated_liver <= 10 or (p.ALT is not None and p.AST is not None and 120 < p.ALT / p.AST <= 400)
        )
        self.conditions["Elevated_liver_enzymes_Severe"] = Condition(
            "Elevated_liver_enzymes_Severe", "重度肝酶升高",
            lambda p: 10 < p.Elevated_liver or (p.ALT is not None and p.AST is not None and p.ALT / p.AST > 400)
        )

        # 糖尿病类型
        self.conditions["T1DM"] = Condition(
            "T1DM", "1型糖尿病",
            lambda p: p.diabetes_type == 1
        )

        self.conditions["Gestational_diabetes"] = Condition(
            "Gestational_diabetes", "妊娠性糖尿病",
            lambda p: p.diabetes_type == 3
        )

        self.conditions["Ketosis_FORBID"] = Condition(
            "Ketosis_FORBID", "酮症、酮症酸中毒、高渗透性高HHS",
            lambda p: False
        )

        self.conditions["Renal_failure_FORBID"] = Condition(
            "Renal_failure_FORBID", "肾衰竭、肾透析",
            lambda p: "肾衰竭" in p.complications or "肾透析" in p.complications
        )

        self.conditions["liver_failure_FORBID"] = Condition(
            "liver_failure_FORBID", "严重肝功能不全",
            lambda p: "严重肝功能不全" in p.complications
        )

    def _initialize_rules(self):

        # 第0层规则：糖尿病类型分型
        self.rules.append(DrugRule(
            rule_id="T1DM",
            description="推荐：T1DM因素影响，推荐二甲双胍和α-糖苷酶抑制剂；禁用：T1DM因素影响，禁用SGLT2I、GLP-1RA、DPP-4I、SU、GLN、TZD、PPAR、GKA",
            required_conditions={"T1DM"},
            recommend_drug_categories={DrugCategory.MET, DrugCategory.AGI},
            forbid_drug_categories={
                DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4,
                DrugCategory.SGLT2I5, DrugCategory.SGLT2I6, DrugCategory.GLP1RA, DrugCategory.DPP4I1,
                DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I4, DrugCategory.DPP4I5,
                DrugCategory.DPP4I6, DrugCategory.DPP4I7, DrugCategory.SU1, DrugCategory.SU2,
                DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5, DrugCategory.GLN, DrugCategory.TZD1,
                DrugCategory.TZD2, DrugCategory.PAN_PPARA, DrugCategory.GKA
            },
            type=0,
        ))

        self.rules.append(DrugRule(
            rule_id="Gestational_diabetes",
            description="推荐：妊娠型糖尿病因素影响，推荐二甲双胍；禁用：妊娠型糖尿病因素影响，禁用SGLT2I、GLP-1RA、DPP-4I、SU、GLN、TZD、PPAR、GKA、AGI",
            recommend_drug_categories={DrugCategory.MET},
            required_conditions={"Gestational_diabetes"},
            forbid_drug_categories={DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4,
                                    DrugCategory.SGLT2I5, DrugCategory.SGLT2I6, DrugCategory.GLP1RA, DrugCategory.DPP4I1,
                                    DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I4, DrugCategory.DPP4I5, DrugCategory.DPP4I6,
                                    DrugCategory.DPP4I7,
                                    DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5, DrugCategory.GLN,
                                    DrugCategory.TZD1,
                                    DrugCategory.TZD2, DrugCategory.PAN_PPARA, DrugCategory.GKA, DrugCategory.AGI},
            type=0,
        ))

        # 第1层规则：心肾保护

        self.rules.append(DrugRule(
            rule_id="ascvd_protection_1",
            description="推荐：合并ASCVD患者推荐心肾保护药物SGLT2I、GLP-1RA，联合MET潜在获益，联合TZD（吡格列酮） 指南2025 [8]",
            required_conditions={"ascvd_present"},
            recommend_drug_categories={DrugCategory.GLP1RA,
                                       DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3,
                                       DrugCategory.SGLT2I4, DrugCategory.SGLT2I5, DrugCategory.SGLT2I6,
                                       DrugCategory.MET, DrugCategory.TZD1},
            type=1
        ))

        self.rules.append(DrugRule(
            rule_id="ascvd_protection_2",
            description="推荐：合并ASCVD或高风险患者推荐心肾保护药物SGLT2I、GLP-1RA，联合MET潜在获益，联合TZD（吡格列酮） 指南2025 [8]（指标判定）",
            recommend_drug_categories={DrugCategory.GLP1RA, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.MET, DrugCategory.TZD1},
            required_conditions={"age_55_plus_2_risk_factors"},
            type=1
        ))

        self.rules.append(DrugRule(
            rule_id="hf_protection",
            description="推荐：合并心衰患者推荐SGLT2I；慎用：合并心衰患者慎用沙格列汀（指南）、阿格列汀（吴正昊）；禁用：合并心衰患者禁用TZD",
            required_conditions={"hf_present"},
            recommend_drug_categories={DrugCategory.SGLT2I1, DrugCategory.SGLT2I2},
            forbid_drug_categories={DrugCategory.TZD1, DrugCategory.TZD2},
            caution_drug_categories={DrugCategory.DPP4I2, DrugCategory.DPP4I5},
            type=1
        ))

        self.rules.append(DrugRule(
            rule_id="ckd_recommend",
            description="推荐：合并CKD患者推荐达格列净、恩格列净，GLP-1RA潜在获益 指南2025 [8]",
            recommend_drug_categories={DrugCategory.SGLT2I1, DrugCategory.SGLT2I2},
            consider_drug_categories={DrugCategory.GLP1RA},
            required_conditions={"ckd_present"},
            type=1
        ))

        self.rules.append(DrugRule(
            rule_id="no_complications",
            description="推荐：没有SGLT2I或GLP-1RA心肾保护强适应症时，通常选择以MET为基础的联合治疗和西格列汀",
            recommend_drug_categories={DrugCategory.MET, DrugCategory.DPP4I1},
            required_conditions={},
            excluded_conditions={"ckd_present", "hf_present", "age_55_plus_2_risk_factors", "ascvd_present"},
            type=1
        ))

        # 体重相关规则
        # self.rules.append(DrugRule(
        #     rule_id="normal_weight_first_line",
        #     description="正常体重患者一线药物选择(不依赖β细胞功能)",
        #     recommend_drug_categories={DrugCategory.MET, DrugCategory.TZD1, DrugCategory.TZD2, DrugCategory.PAN_PPARA, DrugCategory.AGI,
        #                                DrugCategory.SGLT2I1},
        #     required_conditions={"normal_weight"},
        #     type=1
        # ))

        # 第2层规则：其他病症

        self.rules.append(DrugRule(
            rule_id="overweight_first_line",
            description="推荐：超重/肥胖患者一线药物选择推荐卡格列净；禁用：超重/肥胖患者禁用SU",
            recommend_drug_categories={DrugCategory.SGLT2I3},
            consider_drug_categories={DrugCategory.MET, DrugCategory.AGI, DrugCategory.GLP1RA,
                                      DrugCategory.SGLT2I1, DrugCategory.SGLT2I2,
                                      DrugCategory.SGLT2I4, DrugCategory.SGLT2I5, DrugCategory.SGLT2I6, },
            forbid_drug_categories={DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5},
            required_conditions={"overweight_obesity"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="lack_weight",
            description="推荐：BMI低，推荐DPP-4I；禁用：BMI低，禁用SGLT2I、GLP-1RA",
            recommend_drug_categories={DrugCategory.DPP4I1, DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I4, DrugCategory.DPP4I5,
                                       DrugCategory.DPP4I6, DrugCategory.DPP4I7, },
            forbid_drug_categories={DrugCategory.GLP1RA, DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4,
                                    DrugCategory.SGLT2I5, DrugCategory.SGLT2I6},
            required_conditions={"lack_weight"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="hf_protection_2",
            description="禁用：失代偿性HF禁用MET",
            forbid_drug_categories={DrugCategory.MET},
            required_conditions={"decompensated_HF"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="MASLD_present",
            description="推荐：合并MASLD脂肪肝，推荐使用MET、吡格列酮、GLP-1RA；慎用：合并MASLD脂肪肝，慎用SU",
            required_conditions={"MASLD_present"},
            recommend_drug_categories={DrugCategory.MET, DrugCategory.TZD1, DrugCategory.GLP1RA},
            caution_drug_categories={DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="Hyperlipidemia_present",
            description="推荐：合并Hyperlipidemia高脂血症，推荐使用MET与TZD",
            recommend_drug_categories={DrugCategory.TZD1, DrugCategory.TZD2},
            consider_drug_categories={DrugCategory.MET, },
            required_conditions={"Hyperlipidemia_present"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="hypertension_present",
            description="推荐：合并hypertension高血压，推荐SGLT2I和GLP-1RA",
            recommend_drug_categories={DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4, DrugCategory.SGLT2I5,
                                       DrugCategory.SGLT2I6},
            consider_drug_categories={DrugCategory.GLP1RA},
            required_conditions={"hypertension_present"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="EYE_present",
            description="禁用：糖尿病眼病禁用TZD",
            forbid_drug_categories={DrugCategory.TZD2, DrugCategory.TZD1},
            required_conditions={"eye_present"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="gastritis_present_forbid",
            description="禁用：胃轻瘫/胃炎禁用GLP-1RA和α-糖苷酶抑制剂",
            forbid_drug_categories={DrugCategory.GLP1RA, DrugCategory.AGI},
            required_conditions={"gastritis_present"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="metabolic_syndrome_present",
            description="推荐：代谢综合征推荐使用双胍类、SGLT2i、GLP-1RA、TZD、PPAR；慎用：代谢综合征慎用磺脲类",
            recommend_drug_categories={DrugCategory.MET},
            consider_drug_categories={DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4, DrugCategory.SGLT2I5,
                                      DrugCategory.SGLT2I6, DrugCategory.GLP1RA, DrugCategory.TZD1, DrugCategory.TZD2, DrugCategory.PAN_PPARA},
            caution_drug_categories={DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5},
            required_conditions={"metabolic_syndrome_present"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="stroke_history",
            description="推荐：卒中病史者选用GLP-1RA或吡格列酮",
            recommend_drug_categories={DrugCategory.TZD1},
            consider_drug_categories={DrugCategory.GLP1RA},
            required_conditions={"stroke_history"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="Ketosis_FORBID",
            description="禁用：酮症、酮症酸中毒、高渗透性高HHS禁用SGLT2I、DPP-4I、SU、TZD、PPAR、GKA、AGI、MET药物",
            forbid_drug_categories={DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4,
                                    DrugCategory.SGLT2I5, DrugCategory.SGLT2I6, DrugCategory.GLP1RA, DrugCategory.DPP4I1,
                                    DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I4, DrugCategory.DPP4I5, DrugCategory.DPP4I6,
                                    DrugCategory.DPP4I7,
                                    DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5, DrugCategory.GLN,
                                    DrugCategory.TZD1,
                                    DrugCategory.TZD2, DrugCategory.PAN_PPARA, DrugCategory.GKA, DrugCategory.AGI, DrugCategory.MET},
            required_conditions={"Ketosis_FORBID"},
            type=2

        ))

        self.rules.append(DrugRule(
            rule_id="Renal_failure_FORBID",
            description="禁用：肾衰竭、肾透析禁用SGLT2I、DPP-4I、SU、TZD、PPAR、GKA、AGI、MET药物",
            forbid_drug_categories={DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4,
                                    DrugCategory.SGLT2I5, DrugCategory.SGLT2I6, DrugCategory.GLP1RA, DrugCategory.DPP4I1,
                                    DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I4, DrugCategory.DPP4I5, DrugCategory.DPP4I6,
                                    DrugCategory.DPP4I7,
                                    DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5, DrugCategory.GLN,
                                    DrugCategory.TZD1,
                                    DrugCategory.TZD2, DrugCategory.PAN_PPARA, DrugCategory.GKA, DrugCategory.AGI, DrugCategory.MET},
            required_conditions={"Renal_failure_FORBID"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="liver_failure_FORBID",
            description="禁用：严重肝功能不全禁用SGLT2I、GLP-1RA、DPP-4I、SU、TZD、PPAR、GKA、MET、AGI",
            forbid_drug_categories={DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4,
                                    DrugCategory.SGLT2I5, DrugCategory.SGLT2I6, DrugCategory.GLP1RA, DrugCategory.DPP4I1,
                                    DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I4, DrugCategory.DPP4I5, DrugCategory.DPP4I6,
                                    DrugCategory.DPP4I7,
                                    DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5, DrugCategory.GLN,
                                    DrugCategory.TZD1,
                                    DrugCategory.TZD2, DrugCategory.PAN_PPARA, DrugCategory.GKA, DrugCategory.AGI, DrugCategory.MET},
            required_conditions={"liver_failure_FORBID"},
            type=2
        ))

        # self.rules.append(DrugRule(
        #     rule_id="Elevated_liver_enzymes_Severe_CAUTION",
        #     description="重度肝功能不全:可用利格列汀; 禁用磺脲类 和TZD; 避免使用二甲双胍 慎用葡萄糖抑制剂 和非磺脲类促胰岛素分泌剂 慎用DPP-4 ",
        #     caution_drug_categories={DrugCategory.AGI, DrugCategory.GLN, DrugCategory.GKA, DrugCategory.GLP1RA,
        #                              DrugCategory.DPP4I1, DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I5, DrugCategory.DPP4I6,
        #                              DrugCategory.DPP4I7},
        #     forbid_drug_categories={DrugCategory.TZD1, DrugCategory.TZD2, DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4,
        #                             DrugCategory.SU5},
        #     recommend_drug_categories={DrugCategory.DPP4I4},
        #     required_conditions={"Elevated_liver_enzymes_Severe"},
        #     type=2
        # ))

        self.rules.append(DrugRule(
            rule_id="beta_cell_function_low",
            description="推荐：胰岛功能差（空腹C肽<300或OGTT餐后2小时C肽<600）患者推荐二甲双胍；禁用：胰岛功能差提示胰岛素分泌差、促泌剂疗效差，禁用GLP-1RA、DPP-4I、SU、TZD、PPAR",
            recommend_drug_categories={DrugCategory.MET},
            forbid_drug_categories={DrugCategory.GLP1RA, DrugCategory.DPP4I1, DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I4,
                                    DrugCategory.DPP4I5, DrugCategory.DPP4I6, DrugCategory.DPP4I7, DrugCategory.SU1, DrugCategory.SU2,
                                    DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5, DrugCategory.TZD2, DrugCategory.TZD1,
                                    DrugCategory.PAN_PPARA},
            required_conditions={"beta_cell_function_low"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="insulin_resistance_forbid",
            description="推荐：胰岛素强抵抗推荐GLP-1RA、二甲双胍、TZD；禁用：胰岛素强抵抗禁用格列奈类、磺脲类",
            recommend_drug_categories={DrugCategory.GLP1RA, DrugCategory.MET, DrugCategory.TZD1, DrugCategory.TZD2},
            forbid_drug_categories={DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5, DrugCategory.GLN},
            required_conditions={"insulin_resistance"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="Fracture_FORBID_2",
            description="禁用：有严重骨质疏松或近期骨折病史的人禁用卡格列净、PPAR和TZD",
            forbid_drug_categories={DrugCategory.SGLT2I3, DrugCategory.PAN_PPARA, DrugCategory.TZD1, DrugCategory.TZD2},
            required_conditions={"Fracture_FORBID"},
            type=2
        ))

        # 肾功能状态
        self.rules.append(DrugRule(
            rule_id="CKD3A",
            description="推荐：CKD3A期推荐格列喹酮、格列奈类、TZD、GLN、α-糖苷酶抑制剂、DPP-4I；慎用：CKD3A期慎用二甲双胍、格列美脲、格列吡嗪、格列齐特、达格列净、卡格列净、恩格列净；禁用：CKD3A期禁用格列苯脲",
            caution_drug_categories={DrugCategory.MET, DrugCategory.SU5, DrugCategory.SU2, DrugCategory.SU3,
                                     DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3},
            forbid_drug_categories={DrugCategory.SU1},
            recommend_drug_categories={DrugCategory.SU4, DrugCategory.GLN, DrugCategory.TZD2, DrugCategory.TZD1, DrugCategory.AGI,
                                       DrugCategory.DPP4I1, DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I4, DrugCategory.DPP4I5,
                                       DrugCategory.DPP4I6,
                                       DrugCategory.DPP4I7},
            required_conditions={"egfr_3a"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="CKD3B",
            description="推荐：CKD3B期推荐格列喹酮、格列奈类、TZD、GLN、α-糖苷酶抑制剂、利格列汀；慎用：CKD3B期慎用格列吡嗪、格列齐特、西格列汀、维格列汀、沙格列汀、阿格列汀；禁用：CKD3B期禁用二甲双胍、格列苯脲、格列美脲、SGLT2I类",
            recommend_drug_categories={DrugCategory.SU4, DrugCategory.GLN, DrugCategory.TZD2, DrugCategory.TZD1, DrugCategory.AGI,
                                       DrugCategory.DPP4I4},
            caution_drug_categories={DrugCategory.SU2, DrugCategory.SU3, DrugCategory.DPP4I1, DrugCategory.DPP4I2, DrugCategory.DPP4I3,
                                     DrugCategory.DPP4I5,
                                     DrugCategory.DPP4I6, DrugCategory.DPP4I7},
            forbid_drug_categories={DrugCategory.MET, DrugCategory.SU1, DrugCategory.SU5, DrugCategory.SGLT2I1, DrugCategory.SGLT2I2,
                                    DrugCategory.SGLT2I3,
                                    DrugCategory.SGLT2I4, DrugCategory.SGLT2I5, DrugCategory.SGLT2I6},
            required_conditions={"egfr_3b"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="CKD4",
            description="推荐：CKD4期推荐格列喹酮、格列奈类、TZD、GLN、利格列汀；慎用：CKD4期慎用西格列汀、维格列汀、沙格列汀、阿格列汀；禁用：CKD4期禁用二甲双胍、SU（除了格列喹酮SU4）、α-糖苷酶抑制剂、SGLT2I类",
            recommend_drug_categories={DrugCategory.SU4, DrugCategory.GLN, DrugCategory.TZD2, DrugCategory.TZD1, DrugCategory.DPP4I4},
            forbid_drug_categories={DrugCategory.MET, DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU5, DrugCategory.AGI,
                                    DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4, DrugCategory.SGLT2I5,
                                    DrugCategory.SGLT2I6},
            caution_drug_categories={DrugCategory.DPP4I1, DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I5,
                                     DrugCategory.DPP4I6, DrugCategory.DPP4I7},
            required_conditions={"egfr_4"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="CKD5_RECOMMEND",
            description="推荐：CKD5期推荐格列喹酮、格列奈类、TZD、GLN、利格列汀；慎用：CKD5期慎用西格列汀、维格列汀、沙格列汀、阿格列汀；禁用：CKD5期禁用二甲双胍、SU（除了格列喹酮SU4）、α-糖苷酶抑制剂、SGLT2I类",
            recommend_drug_categories={DrugCategory.SU4, DrugCategory.GLN, DrugCategory.TZD2, DrugCategory.TZD1, DrugCategory.DPP4I4},
            caution_drug_categories={DrugCategory.DPP4I1, DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I5,
                                     DrugCategory.DPP4I6, DrugCategory.DPP4I7},
            forbid_drug_categories={DrugCategory.MET, DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU5, DrugCategory.AGI,
                                    DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4, DrugCategory.SGLT2I5,
                                    DrugCategory.SGLT2I6},
            required_conditions={"egfr_5"},
            type=2
        ))

        self.rules.append(DrugRule(
            rule_id="hypoglycemia_high_risk",
            description="推荐：低血糖风险高时推荐DPP-4I和MET；慎用：低血糖风险高时慎用磺脲类、格列奈类",
            recommend_drug_categories={DrugCategory.DPP4I1, DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I4, DrugCategory.DPP4I5,
                                       DrugCategory.DPP4I6, DrugCategory.DPP4I7},
            consider_drug_categories={DrugCategory.MET},
            caution_drug_categories={DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5, DrugCategory.GLN, },
            required_conditions={"Ketogenic_diet_CAUTION"},
            type=2
        ))

        # 第3层规则：药物相关

        # dpp-4i
        self.rules.append(DrugRule(
            rule_id="DPP4I_MET",
            description="推荐：DPP-4I和二甲双胍双联适用于老年、低血糖风险高、胃肠道不良反应患者",
            recommend_drug_categories={DrugCategory.MET, DrugCategory.DPP4I1, DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I4,
                                       DrugCategory.DPP4I5, DrugCategory.DPP4I6},
            required_conditions={"elderly_high_risk_gi"},
            type=3
        ))

        # SU GLN
        self.rules.append(DrugRule(
            rule_id="SU_CAUTION",
            description="慎用：SU增加低血糖风险，老年、肝肾功能不全者慎用",
            caution_drug_categories={DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5},
            required_conditions={"short_duration_good_beta"},
            type=3
        ))
        self.rules.append(DrugRule(
            rule_id="MET-SU_OR_GLN_1",
            description="禁用：超重/肥胖患者禁用磺脲类或格列奈类",
            forbid_drug_categories={DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5, DrugCategory.GLN},
            required_conditions={"overweight_obesity"},
            type=3
        ))
        self.rules.append(DrugRule(
            rule_id="MET-SU_OR_GLN_2",
            description="推荐：磺脲类或格列奈类疗效确切但增加低血糖风险和体重，可与MET联合用于年轻、初诊HbA1c较高、胰岛素β细胞功能较好且不伴超重或肥胖的患者",
            recommend_drug_categories={DrugCategory.MET, DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5,
                                       DrugCategory.GLN},
            required_conditions={"su_gln_met_candidate"},
            type=3
        ))

        self.rules.append(DrugRule(
            rule_id="secretagogue_candidate_general",
            description="推荐/考虑：SU或GLN。字段：diabetes_type=2，HbA1c>=8.0，C_peptide_2>=600，beta_cell_function_low=False，eGFR>=60，age<70，无hypoglycemia风险，BMI<28，且无ASCVD/HF/CKD优先保护需求。解释：胰岛功能尚可、低血糖风险较低且仍需进一步降糖时，促泌剂可作为后线或补充降糖选择。",
            caution_drug_categories={DrugCategory.GLN, DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5},
            required_conditions={"secretagogue_candidate_general"},
            type=3
        ))
        self.rules.append(DrugRule(
            rule_id="gln_postprandial_preferred",
            description="推荐：GLN。字段：diabetes_type=2，delta_blood_glucose_after_meal>=3，C_peptide_2>=600，无严重肝功能不全。解释：GLN起效快、作用短，更适合餐后高血糖突出、餐时降糖需求或进餐不规律风险较高者。",
            recommend_drug_categories={DrugCategory.GLN},
            required_conditions={"gln_postprandial_preferred"},
            type=3
        ))
        self.rules.append(DrugRule(
            rule_id="su_stronger_hba1c_lowering",
            description="推荐：SU。字段：diabetes_type=2，HbA1c>=8.5，C_peptide_2>=600，eGFR>=60，age<70，无低血糖风险，BMI<28。解释：SU降糖强度较高，但低血糖和体重增加风险更高，因此仅在胰岛功能尚可且风险可控时优先推荐。",
            recommend_drug_categories={DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5},
            required_conditions={"su_stronger_hba1c_lowering"},
            type=3
        ))
        self.rules.append(DrugRule(
            rule_id="su_preserved_beta_non_masld_candidate",
            description="Recommend SU: diabetes_type=2, HbA1c>=8.5, duration<=2 years, C_peptide_half>=300, C_peptide_2>=1200, eGFR>=60, age<55, BMI<24, no hypoglycemia risk, and no MASLD/ASCVD/HF/CKD. Explanation: narrow SU recommendation for non-obese patients with preserved beta-cell function and persistent hyperglycemia.",
            recommend_drug_categories={DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5},
            required_conditions={"su_preserved_beta_non_masld_candidate"},
            type=3
        ))
        self.rules.append(DrugRule(
            rule_id="ckd_secretagogue_limited",
            description="考虑/推荐：GLN或格列喹酮(SU4)。字段：diabetes_type=2，30<=eGFR<60，C_peptide_2>=600，无低血糖风险。解释：CKD患者不应泛推荐全部SU；如需促泌剂，优先考虑GLN或相对适合肾功能下降者的格列喹酮。",
            caution_drug_categories={DrugCategory.GLN, DrugCategory.SU4},
            required_conditions={"ckd_secretagogue_limited"},
            type=3
        ))
        self.rules.append(DrugRule(
            rule_id="secretagogue_caution_or_forbid",
            description="慎用/禁用：SU/GLN。字段：beta_cell_function_low=True，或age>=70，或hypoglycemia风险，或eGFR<30，或明显肝功能不全，或BMI>=28。解释：这些状态下促泌剂疗效或安全性下降，SU尤其增加低血糖和体重增加风险，应避免或至少降级为慎用。",
            forbid_drug_categories={DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5},
            caution_drug_categories={DrugCategory.GLN},
            required_conditions={"secretagogue_caution_or_forbid"},
            type=3
        ))

        # AGI
        self.rules.append(DrugRule(
            rule_id="AGI_MET",
            description="推荐：α-糖苷酶抑制剂可轻度减轻体重，适用于餐后血糖控制差的T2DM患者",
            recommend_drug_categories={DrugCategory.AGI, },
            required_conditions={"AGi_blood_glucose"},
            type=2

        ))
        self.rules.append(DrugRule(
            rule_id="AGI_CAUTION",
            description="慎用：胃肠功能差者慎用α-糖苷酶抑制剂",
            required_conditions={"Gastrointestinal_adverse_reactions"},
            caution_drug_categories={DrugCategory.AGI},
            type=3
        ))

        # SGLT2I
        self.rules.append(DrugRule(
            rule_id="Urogenital_CAUTION",
            description="禁用：存在泌尿生殖系感染风险时禁用SGLT2I",
            forbid_drug_categories={DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4, DrugCategory.SGLT2I5, DrugCategory.SGLT2I6},
            required_conditions={"Urogenital_CAUTION"},
            type=3
        ))

        # TZD
        self.rules.append(DrugRule(
            rule_id="TZD_DPP4I_CAUTION",
            description="慎用：TZD与沙格列汀与HF风险增加有关，HF病史和HF诱发因素患者慎用",
            caution_drug_categories={DrugCategory.TZD1, DrugCategory.TZD2, DrugCategory.DPP4I2},
            required_conditions={"HF_CAUTION"},
            type=3
        ))
        self.rules.append(DrugRule(
            rule_id="TZD_CAUTION",
            description="慎用：肝酶升高者慎用TZD",
            caution_drug_categories={DrugCategory.TZD1, DrugCategory.TZD2},
            required_conditions={"Elevated_liver_enzymes_slightly"},
            type=3,
        ))


        self.rules.append(DrugRule(
            rule_id="Fracture_FORBID_1",
            description="禁用：TZD使用与骨折风险增加相关，有严重骨质疏松或近期骨折病史的人禁用",
            forbid_drug_categories={DrugCategory.TZD1, DrugCategory.TZD2},
            required_conditions={"Fracture_FORBID"},
            type=3
        ))


        self.rules.append(DrugRule(
            rule_id="TZD_PPAR",
            description="推荐：TZD或PPAR疗效确切、低血糖风险小但可增加体重，可与MET联合用于伴有明显胰岛素抵抗的T2DM患者",
            recommend_drug_categories={DrugCategory.MET, DrugCategory.TZD1, DrugCategory.TZD2, DrugCategory.PAN_PPARA},
            required_conditions={"insulin_resistance"},
            type=3
        ))

        # GLP-1RA
        self.rules.append(DrugRule(
            rule_id="GLP-1RA_FORBID",
            description="禁用：甲状腺髓样癌、多发性内分泌腺瘤病2型既往史或家族史、胰腺炎病史患者禁用GLP-1RA",
            forbid_drug_categories={DrugCategory.GLP1RA},
            required_conditions={"GLP-1RA_FORBID"},
            type=3
        ))

        # GKA
        self.rules.append(DrugRule(
            rule_id="GKA",
            description="推荐：GKA改善血糖稳态失调，与MET兼顾空腹和餐后血糖，适合病程较短、胰岛细胞功能较好的患者",
            recommend_drug_categories={DrugCategory.GKA},
            required_conditions={"short_duration_good_beta"},
            type=3
        ))
        self.rules.append(DrugRule(
            rule_id="GKA_recall_prioritized",
            description="Recommend GKA as recall-prioritized screening when glycemic homeostasis may benefit from Dorzagliatin and preserved beta-cell response is present.",
            recommend_drug_categories={DrugCategory.GKA},
            required_conditions={"gka_recall_prioritized"},
            type=3
        ))

        # 第4层规则：症状及补充
        # 年龄
        self.rules.append(DrugRule(
            rule_id="MET_base_therapy_1",
            description="推荐：70岁以下血糖高的患者，推荐以二甲双胍为基础的单药/联合治疗",
            required_conditions={"hba1c_high"},
            excluded_conditions={"age_70_plus"},
            recommend_drug_categories={DrugCategory.MET},
            type=4
        ))

        self.rules.append(DrugRule(
            rule_id="MET_base_therapy_2",
            description="推荐：血糖特别高的70岁以上患者，推荐以二甲双胍为基础的单药/联合治疗 指南2025 [6]",
            recommend_drug_categories={DrugCategory.MET},
            required_conditions={"hba1c_very_high", "age_70_plus"},
            type=4
        ))
        self.rules.append(DrugRule(
            rule_id="old_recommend",
            description="推荐：老年患者建议优选MET、DPP-4I、SGLT2I、GLP-1RA或以此为基础的联合治疗方案，减少低血糖风险，并权衡SGLT2I和GLP-1RA的治疗获益与风险",
            recommend_drug_categories={DrugCategory.MET, DrugCategory.DPP4I1, DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I4,
                                       DrugCategory.DPP4I5,
                                       DrugCategory.DPP4I6, DrugCategory.DPP4I7},
            required_conditions={"age_70_plus"},
            type=4
        ))

        self.rules.append(DrugRule(
            rule_id="teenager_recommend",
            description="推荐：10-18岁儿童和青少年仅批准使用二甲双胍；禁用：10-18岁儿童和青少年禁用SGLT2I、GLP-1RA、DPP-4I、SU、GLN、TZD、PPAR、GKA、AGI",
            recommend_drug_categories={DrugCategory.MET, },
            forbid_drug_categories={DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4,
                                    DrugCategory.SGLT2I5, DrugCategory.SGLT2I6, DrugCategory.GLP1RA, DrugCategory.DPP4I1,
                                    DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I4, DrugCategory.DPP4I5, DrugCategory.DPP4I6,
                                    DrugCategory.DPP4I7,
                                    DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5, DrugCategory.GLN,
                                    DrugCategory.TZD1,
                                    DrugCategory.TZD2, DrugCategory.PAN_PPARA, DrugCategory.GKA, DrugCategory.AGI},
            required_conditions={"children_under_18"},
            type=4
        ))

        self.rules.append(DrugRule(
            rule_id="children_forbid",
            description="禁用：小于10岁儿童只考虑胰岛素，禁用非胰岛素降糖药物",
            recommend_drug_categories={},
            forbid_drug_categories={DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4,
                                    DrugCategory.SGLT2I5, DrugCategory.SGLT2I6, DrugCategory.GLP1RA, DrugCategory.DPP4I1,
                                    DrugCategory.DPP4I2, DrugCategory.DPP4I3, DrugCategory.DPP4I4, DrugCategory.DPP4I5, DrugCategory.DPP4I6,
                                    DrugCategory.DPP4I7,
                                    DrugCategory.SU1, DrugCategory.SU2, DrugCategory.SU3, DrugCategory.SU4, DrugCategory.SU5, DrugCategory.GLN,
                                    DrugCategory.TZD1,
                                    DrugCategory.TZD2, DrugCategory.PAN_PPARA, DrugCategory.GKA, DrugCategory.AGI, DrugCategory.MET},
            required_conditions={"children_under_10"},
            type=4
        ))

        self.rules.append(DrugRule(
            rule_id="Ketogenic_diet_CAUTION",
            description="慎用：生酮饮食慎用SGLT2I",
            caution_drug_categories={DrugCategory.SGLT2I1, DrugCategory.SGLT2I2, DrugCategory.SGLT2I3, DrugCategory.SGLT2I4, DrugCategory.SGLT2I5},
            required_conditions={"Ketogenic_diet_CAUTION"},
            type=4
        ))

        # print(f"全部规则数量：{len(self.rules)}")
        # print("第0层糖尿病类型分型\n 第1层规则:糖尿病合并症\n第2层规则:相关病症\n第3层规则:药物及相关使用\n第4层规则:症状、病程及指标禁忌反馈\n")
        # for i in range(5):
        #     layer_rules = [r for r in self.rules if r.type == i]
        #     print(f"第{i}层规则数量：{len(layer_rules)}")

    def check_condition(self, patient: Patient, condition_name: str) -> bool:
        """检查单个条件"""
        if condition_name not in self.conditions:
            return False
        return self.conditions[condition_name].check(patient)

    def check_rule_applicable(self, patient: Patient, rule: DrugRule) -> bool:
        """检查规则是否适用于患者"""
        for condition in rule.required_conditions:
            if not self.check_condition(patient, condition):
                return False
        for condition in rule.excluded_conditions:
            if self.check_condition(patient, condition):
                return False
        return True

    def evaluate_patient(self, patient: Patient) -> Tuple[Set[DrugRecommendation], Set[DrugRecommendation], Set[DrugRecommendation]]:
        """
        评估患者并返回三个集合：
        S1: 强推荐药物（包含推荐规则ID）
        S2: 弱推荐药物（包含推荐规则ID）
        S3: 禁用药物（包含禁用规则ID）
        """
        # 初始化药物推荐字典
        drug_data: Dict[DrugCategory, DrugRecommendation] = {}
        for drug in DrugCategory:
            drug_data[drug] = DrugRecommendation(drug=drug)

        # 按规则类型分组
        rules_by_type = defaultdict(list)
        for rule in self.rules:
            rules_by_type[rule.type].append(rule)

        # 记录每个规则对应的药物
        rule_to_drugs: Dict[str, Set[DrugCategory]] = {}

        # 初始化S1, S2, S3集合（存储DrugCategory，便于比较）
        S1_drugs = set()  # 强推荐
        S2_drugs = set()  # 弱推荐
        S3_drugs = set()  # 禁用

        # ==================== 处理type=0的规则 ====================
        for rule in rules_by_type[0]:
            if self.check_rule_applicable(patient, rule):
                rule_to_drugs[rule.rule_id] = set()

                # type=0: recommend -> S1, forbid -> S3
                for drug in rule.recommend_drug_categories:
                    drug_data[drug] = drug_data[drug].add_recommend_rule(rule.rule_id)
                    rule_to_drugs[rule.rule_id].add(drug)
                    # 直接放入S1
                    if drug not in S3_drugs:  # 比较药物，而不是对象
                        S1_drugs.add(drug)

                for drug in rule.forbid_drug_categories:
                    drug_data[drug] = drug_data[drug].add_forbid_rule(rule.rule_id)
                    # 直接放入S3，并从S1,S2中移除
                    S1_drugs.discard(drug)
                    S2_drugs.discard(drug)
                    S3_drugs.add(drug)

        # ==================== 处理type=1的规则 ====================
        # 先收集所有type=1规则的结果
        temp_s1_for_type1 = set()
        temp_s2_for_type1 = set()
        temp_s3_for_type1 = set()

        for rule in rules_by_type[1]:
            if self.check_rule_applicable(patient, rule):
                rule_to_drugs[rule.rule_id] = set()

                # recommend -> S1
                for drug in rule.recommend_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_recommend_rule(rule.rule_id)
                    temp_s1_for_type1.add(drug)

                # consider -> trace only; do not output as S2
                for drug in rule.consider_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_consider_rule(rule.rule_id)

                # caution -> S2（谨慎使用）
                for drug in rule.caution_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_caution_rule(rule.rule_id)
                    temp_s2_for_type1.add(drug)

                # forbid -> S3
                for drug in rule.forbid_drug_categories:
                    drug_data[drug] = drug_data[drug].add_forbid_rule(rule.rule_id)
                    temp_s3_for_type1.add(drug)

        # 合并type=1的结果，优先级S3 > S1 > S2
        # 先处理S3
        for drug in temp_s3_for_type1:
            S1_drugs.discard(drug)
            S2_drugs.discard(drug)
            S3_drugs.add(drug)

        # 再处理S1（不在S3中的）
        for drug in temp_s1_for_type1:
            if drug not in S3_drugs:
                S1_drugs.add(drug)
                # 如果同时也在S2中，从S2移除
                S2_drugs.discard(drug)

        # 最后处理S2（不在S3和S1中的）
        for drug in temp_s2_for_type1:
            if drug not in S3_drugs and drug not in S1_drugs:
                S2_drugs.add(drug)

        # ==================== 处理type=2的规则 ====================
        for rule in rules_by_type[2]:
            if self.check_rule_applicable(patient, rule):
                rule_to_drugs[rule.rule_id] = set()

                # 记录被forbid的药物
                drugs_forbidden_in_this_rule = []

                # 处理forbid药物
                for drug in rule.forbid_drug_categories:
                    drug_data[drug] = drug_data[drug].add_forbid_rule(rule.rule_id)

                    # 从S1和S2中移除，添加到S3
                    if drug in S1_drugs:
                        S1_drugs.remove(drug)
                    if drug in S2_drugs:
                        S2_drugs.remove(drug)
                    S3_drugs.add(drug)
                    drugs_forbidden_in_this_rule.append(drug)

                # 处理recommend药物
                for drug in rule.recommend_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_recommend_rule(rule.rule_id)

                    # 确保不在S3中
                    if drug not in S3_drugs:
                        # 如果不在S1中，添加到S1
                        if drug not in S1_drugs:
                            S1_drugs.add(drug)
                            # 如果从S2移到S1，从S2移除
                            S2_drugs.discard(drug)

                # 处理consider药物：只记录规则，不输出到S2
                for drug in rule.consider_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_consider_rule(rule.rule_id)

                # 处理caution药物：记录谨慎使用规则，最终进入S2（除非已推荐或禁用）
                for drug in rule.caution_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_caution_rule(rule.rule_id)

                    if drug not in S3_drugs and drug not in S1_drugs:
                        S2_drugs.add(drug)

                # 检查是否需要从S2补充药物到S1
                for drug in drugs_forbidden_in_this_rule:
                    # 找出所有推荐这个药物的规则
                    for rule_id in drug_data[drug].recommend_rules:
                        if rule_id in rule_to_drugs:
                            # 检查这个规则对应的推荐药物在S1中是否还有
                            rule_recommend_in_S1 = False
                            for rule_drug in rule_to_drugs[rule_id]:
                                if rule_drug in S1_drugs:
                                    rule_recommend_in_S1 = True
                                    break

                            if not rule_recommend_in_S1:
                                # 从S2中找一个对应该规则的推荐药物
                                found_in_S2 = False
                                for rule_drug in rule_to_drugs[rule_id]:
                                    if rule_drug not in S3_drugs and rule_drug in S2_drugs:
                                        # 移到S1
                                        S2_drugs.remove(rule_drug)
                                        S1_drugs.add(rule_drug)
                                        found_in_S2 = True
                                        break

                                if not found_in_S2:
                                    # 检查consider药物
                                    for rule_drug in rule_to_drugs[rule_id]:
                                        if rule_drug not in S3_drugs and rule_drug in S2_drugs:
                                            # 移到S1
                                            S2_drugs.remove(rule_drug)
                                            S1_drugs.add(rule_drug)
                                            found_in_S2 = True
                                            break

                                if not found_in_S2:
                                    print(f"异常: 规则 {rule_id} 的推荐药物全部被移除，且S2中没有可补充的药物")

        # ==================== 处理type=3的规则 ====================
        for rule in rules_by_type[3]:
            if self.check_rule_applicable(patient, rule):
                rule_to_drugs[rule.rule_id] = set()

                # type=3: 药物相关规则，同样按 forbid > recommend > consider/caution 处理
                for drug in rule.forbid_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_forbid_rule(rule.rule_id)

                    # 从S1和S2中移除，添加到S3
                    if drug in S1_drugs:
                        S1_drugs.remove(drug)
                    if drug in S2_drugs:
                        S2_drugs.remove(drug)
                    S3_drugs.add(drug)

                for drug in rule.recommend_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_recommend_rule(rule.rule_id)
                    if drug not in S3_drugs:
                        S1_drugs.add(drug)
                        S2_drugs.discard(drug)

                for drug in rule.consider_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_consider_rule(rule.rule_id)

                for drug in rule.caution_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_caution_rule(rule.rule_id)
                    if drug not in S3_drugs and drug not in S1_drugs:
                        S2_drugs.add(drug)

        # ==================== 处理type=4的规则 ====================
        for rule in rules_by_type[4]:
            if self.check_rule_applicable(patient, rule):
                rule_to_drugs[rule.rule_id] = set()

                # recommend到S1
                for drug in rule.recommend_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_recommend_rule(rule.rule_id)

                    # 确保不在S3中
                    if drug not in S3_drugs:
                        if drug not in S1_drugs:
                            S1_drugs.add(drug)
                        # 如果从S2移到S1，从S2移除
                        S2_drugs.discard(drug)

                # consider只记录规则，不输出到S2
                for drug in rule.consider_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_consider_rule(rule.rule_id)

                # caution输出到S2
                for drug in rule.caution_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_caution_rule(rule.rule_id)

                    # 确保不在S3和S1中
                    if drug not in S3_drugs and drug not in S1_drugs:
                        S2_drugs.add(drug)

                # forbid到S3
                for drug in rule.forbid_drug_categories:
                    rule_to_drugs[rule.rule_id].add(drug)
                    drug_data[drug] = drug_data[drug].add_forbid_rule(rule.rule_id)

                    if drug in S1_drugs:
                        S1_drugs.remove(drug)
                    if drug in S2_drugs:
                        S2_drugs.remove(drug)
                    S3_drugs.add(drug)

        # ==================== 最终验证和清理 ====================
        # 验证：确保S3中的药物不在S1或S2中
        for drug in SU_DRUG_CATEGORIES:
            if drug_data[drug].forbid_rules:
                S1_drugs.discard(drug)
                S2_drugs.discard(drug)
                S3_drugs.add(drug)

        for drug in S3_drugs:
            if drug in S1_drugs:
                print(f"警告: 药物 {drug.value} 同时在S1和S3中，已从S1移除")
                S1_drugs.remove(drug)
            if drug in S2_drugs:
                print(f"警告: 药物 {drug.value} 同时在S2和S3中，已从S2移除")
                S2_drugs.remove(drug)

        # 验证：确保S1中的药物不在S2中
        for drug in S1_drugs:
            if drug in S2_drugs:
                print(f"警告: 药物 {drug.value} 同时在S1和S2中，已从S2移除")
                S2_drugs.remove(drug)

        # 将DrugCategory集合转换为DrugRecommendation集合
        S1 = {drug_data[drug] for drug in S1_drugs}
        S2 = {drug_data[drug] for drug in S2_drugs}
        S3 = {drug_data[drug] for drug in S3_drugs}

        return S1, S2, S3

    def display_recommendations(self, patient: Patient, group_by_category: bool = True) -> None:
        """
        显示患者的药物推荐结果，包含S1、S2、S3及每个药物的推荐/禁用规则
        参数:
            group_by_category: 是否按药物类别分组显示
        """
        S1, S2, S3 = self.evaluate_patient(patient)

        print("=" * 80)
        print("糖尿病药物推荐系统")
        print("=" * 80)

        # 显示患者基本信息
        print(f"\n患者基本信息:")
        print(f"  年龄: {patient.age}岁")
        print(f"  BMI: {patient.bmi}")
        print(f"  HbA1c: {patient.hba1c}%")
        print(f"  糖尿病类型: {patient.diabetes_type}型")
        print(f"  病程: {patient.duration}年")

        # 显示并发症
        if patient.complications:
            print(f"  并发症: {', '.join(patient.complications)}")

        # 显示风险因素
        if patient.risk_factors:
            print(f"  风险因素: {', '.join(patient.risk_factors)}")

        # 显示病史
        if patient.history:
            print(f"  病史: {', '.join(patient.history)}")

        print("\n" + "=" * 80)

        # 获取规则映射，用于显示规则描述
        rule_map = {rule.rule_id: rule for rule in self.rules}

        def format_rules(rule_ids, rule_type="推荐"):
            """格式化规则ID为可读的描述"""
            if not rule_ids:
                return "无"

            formatted = []
            for rule_id in rule_ids:
                if rule_id in rule_map:
                    rule = rule_map[rule_id]
                    formatted.append(f"{rule_id}: {rule.description}")
                else:
                    formatted.append(f"{rule_id}")

            return "\n    ".join(formatted)

        def group_drugs_by_category(drug_set):
            """将药物按类别分组"""
            category_groups = {}

            # 遍历所有药物
            for drug_rec in drug_set:
                drug_name = drug_rec.drug.value
                found_category = False

                # 查找药物所属类别
                for category_name, drug_list in self.drug_categories.items():
                    if drug_name in drug_list:
                        # 如果药物在字典中，按类别分组
                        if category_name not in category_groups:
                            category_groups[category_name] = []
                        category_groups[category_name].append(drug_rec)
                        found_category = True
                        break

                # 如果不在字典中，药物自己作为一类
                if not found_category:
                    # 使用药物名称作为类别名
                    category_groups[drug_name] = [drug_rec]

            return category_groups

        def display_drug_group(group_name, drug_list, set_name="强推荐"):
            """显示一个药物类别组"""
            if not drug_list:
                return

            # 判断是否是单药类别（类别名就是药物名）
            is_single_drug = len(drug_list) == 1 and group_name == drug_list[0].drug.value

            if is_single_drug:
                # 单药类别，直接显示药物，不显示类别标题
                drug_rec = drug_list[0]
                print(f"\n    • {drug_rec.drug.value}")

                # 显示推荐规则
                if drug_rec.recommend_rules:
                    print(f"       📋 推荐理由:")
                    print(f"         {format_rules(sorted(drug_rec.recommend_rules))}")
                else:
                    print(f"       📋 推荐理由: 无")

                if drug_rec.consider_rules:
                    print(f"       💡 可考虑理由:")
                    print(f"         {format_rules(sorted(drug_rec.consider_rules), '可考虑')}")

                if drug_rec.caution_rules:
                    print(f"       ⚠️  谨慎使用理由:")
                    print(f"         {format_rules(sorted(drug_rec.caution_rules), '谨慎')}")

                # 如果是S3，显示禁用规则
                if set_name == "禁用" and drug_rec.forbid_rules:
                    print(f"       🚫 禁用理由:")
                    print(f"         {format_rules(sorted(drug_rec.forbid_rules), '禁用')}")

                # 如果是S1或S2，显示可能存在的禁用规则
                if set_name in ["强推荐", "弱推荐"] and drug_rec.forbid_rules:
                    print(f"       ⚠️  禁用警告:")
                    print(f"         {format_rules(sorted(drug_rec.forbid_rules), '禁用')}")
            else:
                # 多药类别，显示类别标题
                print(f"\n  📁 {group_name}:")

                for i, drug_rec in enumerate(sorted(drug_list, key=lambda x: x.drug.value), 1):
                    print(f"\n    {i}. {drug_rec.drug.value}")

                    # 显示推荐规则
                    if drug_rec.recommend_rules:
                        print(f"       📋 推荐理由:")
                        print(f"         {format_rules(sorted(drug_rec.recommend_rules))}")
                    else:
                        print(f"       📋 推荐理由: 无")

                    if drug_rec.consider_rules:
                        print(f"       💡 可考虑理由:")
                        print(f"         {format_rules(sorted(drug_rec.consider_rules), '可考虑')}")

                    if drug_rec.caution_rules:
                        print(f"       ⚠️  谨慎使用理由:")
                        print(f"         {format_rules(sorted(drug_rec.caution_rules), '谨慎')}")

                    # 如果是S3，显示禁用规则
                    if set_name == "禁用" and drug_rec.forbid_rules:
                        print(f"       🚫 禁用理由:")
                        print(f"         {format_rules(sorted(drug_rec.forbid_rules), '禁用')}")

                    # 如果是S1或S2，显示可能存在的禁用规则
                    if set_name in ["强推荐", "弱推荐"] and drug_rec.forbid_rules:
                        print(f"       ⚠️  禁用警告:")
                        print(f"         {format_rules(sorted(drug_rec.forbid_rules), '禁用')}")

        # 显示S1：强推荐药物
        print("\n🏥 强推荐药物 (S1):")
        if S1:
            if group_by_category:
                grouped_drugs = group_drugs_by_category(S1)
                # 先显示字典中定义的类别，按类别名排序
                predefined_categories = [cat for cat in sorted(grouped_drugs.keys())
                                         if cat in self.drug_categories]
                for category_name in predefined_categories:
                    display_drug_group(category_name, grouped_drugs[category_name], "强推荐")

                # 再显示单药类别（不在字典中的药物），按药物名排序
                single_drug_categories = [cat for cat in sorted(grouped_drugs.keys())
                                          if cat not in self.drug_categories]
                for category_name in single_drug_categories:
                    display_drug_group(category_name, grouped_drugs[category_name], "强推荐")
            else:
                for i, drug_rec in enumerate(sorted(S1, key=lambda x: x.drug.value), 1):
                    print(f"\n  {i}. {drug_rec.drug.value}")

                    # 显示推荐规则
                    if drug_rec.recommend_rules:
                        print(f"     📋 推荐理由:")
                        print(f"       {format_rules(sorted(drug_rec.recommend_rules))}")
                    else:
                        print(f"     📋 推荐理由: 无")

                    if drug_rec.consider_rules:
                        print(f"     💡 可考虑理由:")
                        print(f"       {format_rules(sorted(drug_rec.consider_rules), '可考虑')}")
                    if drug_rec.caution_rules:
                        print(f"     ⚠️  谨慎使用理由:")
                        print(f"       {format_rules(sorted(drug_rec.caution_rules), '谨慎')}")

                    # 显示禁用规则（理论上S1中不应有，但显示以防万一）
                    if drug_rec.forbid_rules:
                        print(f"     ⚠️  禁用警告:")
                        print(f"       {format_rules(sorted(drug_rec.forbid_rules), '禁用')}")
        else:
            print("  暂无强推荐药物")

        # 显示S2：弱推荐药物
        print("\n\n⚠️ 谨慎/可考虑药物 (S2):")
        if S2:
            if group_by_category:
                grouped_drugs = group_drugs_by_category(S2)
                # 先显示字典中定义的类别，按类别名排序
                predefined_categories = [cat for cat in sorted(grouped_drugs.keys())
                                         if cat in self.drug_categories]
                for category_name in predefined_categories:
                    display_drug_group(category_name, grouped_drugs[category_name], "弱推荐")

                # 再显示单药类别（不在字典中的药物），按药物名排序
                single_drug_categories = [cat for cat in sorted(grouped_drugs.keys())
                                          if cat not in self.drug_categories]
                for category_name in single_drug_categories:
                    display_drug_group(category_name, grouped_drugs[category_name], "弱推荐")
            else:
                for i, drug_rec in enumerate(sorted(S2, key=lambda x: x.drug.value), 1):
                    print(f"\n  {i}. {drug_rec.drug.value}")

                    # 显示推荐规则
                    if drug_rec.recommend_rules:
                        print(f"     📋 推荐理由:")
                        print(f"       {format_rules(sorted(drug_rec.recommend_rules))}")
                    else:
                        print(f"     📋 推荐理由: 无")

                    if drug_rec.consider_rules:
                        print(f"     💡 可考虑理由:")
                        print(f"       {format_rules(sorted(drug_rec.consider_rules), '可考虑')}")
                    if drug_rec.caution_rules:
                        print(f"     ⚠️  谨慎使用理由:")
                        print(f"       {format_rules(sorted(drug_rec.caution_rules), '谨慎')}")

                    # 显示禁用规则（理论上S2中不应有，但显示以防万一）
                    if drug_rec.forbid_rules:
                        print(f"     ⚠️  禁用警告:")
                        print(f"       {format_rules(sorted(drug_rec.forbid_rules), '禁用')}")
        else:
            print("  暂无谨慎/可考虑药物")

        # 显示S3：禁用药物
        print("\n\n🚫 禁用药物 (S3):")
        if S3:
            if group_by_category:
                grouped_drugs = group_drugs_by_category(S3)
                # 先显示字典中定义的类别，按类别名排序
                predefined_categories = [cat for cat in sorted(grouped_drugs.keys())
                                         if cat in self.drug_categories]
                for category_name in predefined_categories:
                    display_drug_group(category_name, grouped_drugs[category_name], "禁用")

                # 再显示单药类别（不在字典中的药物），按药物名排序
                single_drug_categories = [cat for cat in sorted(grouped_drugs.keys())
                                          if cat not in self.drug_categories]
                for category_name in single_drug_categories:
                    display_drug_group(category_name, grouped_drugs[category_name], "禁用")
            else:
                for i, drug_rec in enumerate(sorted(S3, key=lambda x: x.drug.value), 1):
                    print(f"\n  {i}. {drug_rec.drug.value}")

                    # 显示禁用规则
                    if drug_rec.forbid_rules:
                        print(f"     🚫 禁用理由:")
                        print(f"       {format_rules(sorted(drug_rec.forbid_rules), '禁用')}")
                    else:
                        print(f"     🚫 禁用理由: 无")

                    if drug_rec.consider_rules:
                        print(f"     💡 可考虑理由:")
                        print(f"       {format_rules(sorted(drug_rec.consider_rules), '可考虑')}")
                    if drug_rec.caution_rules:
                        print(f"     ⚠️  谨慎使用理由:")
                        print(f"       {format_rules(sorted(drug_rec.caution_rules), '谨慎')}")

                    # 显示可能存在的推荐规则（如果药物同时有推荐和禁用规则）
                    if drug_rec.recommend_rules:
                        print(f"     ℹ️  注意：此药物也有推荐理由:")
                        print(f"       {format_rules(sorted(drug_rec.recommend_rules))}")
        else:
            print("  暂无禁用药物")

        print("\n" + "=" * 80)

        # 显示总结统计
        print(f"\n📊 推荐总结:")
        print(f"  强推荐药物: {len(S1)} 种")
        print(f"  谨慎/可考虑药物: {len(S2)} 种")
        print(f"  禁用药物: {len(S3)} 种")

        print("\n" + "=" * 80)

    def get_detailed_report(self, patient: Patient) -> Dict[str, List[Dict]]:
        """
        获取详细的药物推荐报告，返回结构化数据
        可用于前端展示或导出

        返回格式:
        {
            "S1": [
                {
                    "drug": "药物名称",
                    "drug_enum": DrugCategory,
                    "recommend_rules": [
                        {"rule_id": "规则ID", "description": "规则描述", "type": 规则类型}
                    ],
                    "forbid_rules": [...]  # 同上格式
                },
                ...
            ],
            "S2": [...],
            "S3": [...]
        }
        """
        S1, S2, S3 = self.evaluate_patient(patient)
        rule_map = {rule.rule_id: rule for rule in self.rules}

        def format_drug_records(drug_set):
            result = []
            for drug_rec in sorted(drug_set, key=lambda x: x.drug.value):
                drug_record = {
                    "drug": drug_rec.drug.value,
                    "drug_enum": drug_rec.drug,
                    "recommend_rules": [],
                    "consider_rules": [],
                    "caution_rules": [],
                    "forbid_rules": []
                }

                # 添加推荐规则详情
                for rule_id in sorted(drug_rec.recommend_rules):
                    if rule_id in rule_map:
                        rule = rule_map[rule_id]
                        drug_record["recommend_rules"].append({
                            "rule_id": rule_id,
                            "description": rule.description,
                            "type": rule.type
                        })
                    else:
                        drug_record["recommend_rules"].append({
                            "rule_id": rule_id,
                            "description": "未知规则",
                            "type": -1
                        })


                # 添加可考虑规则详情
                for rule_id in sorted(drug_rec.consider_rules):
                    if rule_id in rule_map:
                        rule = rule_map[rule_id]
                        drug_record["consider_rules"].append({
                            "rule_id": rule_id,
                            "description": rule.description,
                            "type": rule.type
                        })
                    else:
                        drug_record["consider_rules"].append({
                            "rule_id": rule_id,
                            "description": "未知规则",
                            "type": -1
                        })

                # 添加谨慎使用规则详情
                for rule_id in sorted(drug_rec.caution_rules):
                    if rule_id in rule_map:
                        rule = rule_map[rule_id]
                        drug_record["caution_rules"].append({
                            "rule_id": rule_id,
                            "description": rule.description,
                            "type": rule.type
                        })
                    else:
                        drug_record["caution_rules"].append({
                            "rule_id": rule_id,
                            "description": "未知规则",
                            "type": -1
                        })

                # 添加禁用规则详情
                for rule_id in sorted(drug_rec.forbid_rules):
                    if rule_id in rule_map:
                        rule = rule_map[rule_id]
                        drug_record["forbid_rules"].append({
                            "rule_id": rule_id,
                            "description": rule.description,
                            "type": rule.type
                        })
                    else:
                        drug_record["forbid_rules"].append({
                            "rule_id": rule_id,
                            "description": "未知规则",
                            "type": -1
                        })

                result.append(drug_record)

            return result

        s1_records = format_drug_records(S1)
        s2_records = format_drug_records(S2)
        s3_records = format_drug_records(S3)

        return {
            "S1": s1_records,
            "S2": s2_records,
            "S3": s3_records,
            "recommended_drugs": s1_records,
            "caution_drugs": s2_records,
            "forbidden_drugs": s3_records
        }

    def export_to_text(self, patient: Patient, filename: str = None, group_by_category: bool = True) -> str:
        """
        将推荐结果导出为文本格式
        如果提供文件名，则保存到文件；否则返回文本内容
        参数:
            group_by_category: 是否按药物类别分组显示
        """
        S1, S2, S3 = self.evaluate_patient(patient)
        rule_map = {rule.rule_id: rule for rule in self.rules}

        def format_rules_for_export(rule_ids):
            """为导出格式化规则"""
            if not rule_ids:
                return "无"

            formatted = []
            for rule_id in rule_ids:
                if rule_id in rule_map:
                    rule = rule_map[rule_id]
                    formatted.append(f"{rule_id} (type={rule.type}): {rule.description}")
                else:
                    formatted.append(f"{rule_id}")

            return "\n     • ".join(formatted)

        def group_drugs_by_category(drug_set):
            """将药物按类别分组"""
            category_groups = {}

            # 遍历所有药物
            for drug_rec in drug_set:
                drug_name = drug_rec.drug.value
                found_category = False

                # 查找药物所属类别
                for category_name, drug_list in self.drug_categories.items():
                    if drug_name in drug_list:
                        # 如果药物在字典中，按类别分组
                        if category_name not in category_groups:
                            category_groups[category_name] = []
                        category_groups[category_name].append(drug_rec)
                        found_category = True
                        break

                # 如果不在字典中，药物自己作为一类
                if not found_category:
                    # 使用药物名称作为类别名
                    category_groups[drug_name] = [drug_rec]

            return category_groups

        # 构建文本内容
        text_lines = []
        text_lines.append("=" * 80)
        text_lines.append("糖尿病药物推荐报告")
        text_lines.append("=" * 80)
        text_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        text_lines.append("")

        # 患者信息
        text_lines.append("患者信息:")
        text_lines.append(f"  年龄: {patient.age}岁")
        text_lines.append(f"  BMI: {patient.bmi}")
        text_lines.append(f"  HbA1c: {patient.hba1c}%")
        text_lines.append(f"  糖尿病类型: {patient.diabetes_type}型")
        text_lines.append(f"  病程: {patient.duration}年")

        if patient.complications:
            text_lines.append(f"  并发症: {', '.join(patient.complications)}")

        if patient.risk_factors:
            text_lines.append(f"  风险因素: {', '.join(patient.risk_factors)}")

        if patient.history:
            text_lines.append(f"  病史: {', '.join(patient.history)}")

        text_lines.append("")
        text_lines.append("=" * 80)
        text_lines.append("")

        # 强推荐药物
        text_lines.append("强推荐药物 (S1):")
        text_lines.append("")

        if S1:
            if group_by_category:
                grouped_drugs = group_drugs_by_category(S1)

                # 先显示字典中定义的类别，按类别名排序
                predefined_categories = [cat for cat in sorted(grouped_drugs.keys())
                                         if cat in self.drug_categories]
                for category_name in predefined_categories:
                    drug_list = grouped_drugs[category_name]
                    if len(drug_list) == 1 and category_name == drug_list[0].drug.value:
                        # 单药类别
                        drug_rec = drug_list[0]
                        text_lines.append(f"• {drug_rec.drug.value}")

                        if drug_rec.recommend_rules:
                            text_lines.append("  推荐理由:")
                            text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.recommend_rules))}")

                        if drug_rec.forbid_rules:
                            text_lines.append("  禁用警告:")
                            text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.forbid_rules))}")
                    else:
                        # 多药类别
                        text_lines.append(f"{category_name}:")

                        for i, drug_rec in enumerate(sorted(drug_list, key=lambda x: x.drug.value), 1):
                            text_lines.append(f"  {i}. {drug_rec.drug.value}")

                            if drug_rec.recommend_rules:
                                text_lines.append("    推荐理由:")
                                text_lines.append(f"      • {format_rules_for_export(sorted(drug_rec.recommend_rules))}")

                            if drug_rec.forbid_rules:
                                text_lines.append("    禁用警告:")
                                text_lines.append(f"      • {format_rules_for_export(sorted(drug_rec.forbid_rules))}")

                    text_lines.append("")

                # 再显示单药类别（不在字典中的药物），按药物名排序
                single_drug_categories = [cat for cat in sorted(grouped_drugs.keys())
                                          if cat not in self.drug_categories]
                for category_name in single_drug_categories:
                    drug_list = grouped_drugs[category_name]
                    drug_rec = drug_list[0]  # 单药类别只有一个药物
                    text_lines.append(f"• {drug_rec.drug.value}")

                    if drug_rec.recommend_rules:
                        text_lines.append("  推荐理由:")
                        text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.recommend_rules))}")

                    if drug_rec.forbid_rules:
                        text_lines.append("  禁用警告:")
                        text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.forbid_rules))}")

                    text_lines.append("")
            else:
                for i, drug_rec in enumerate(sorted(S1, key=lambda x: x.drug.value), 1):
                    text_lines.append(f"{i}. {drug_rec.drug.value}")

                    if drug_rec.recommend_rules:
                        text_lines.append("  推荐理由:")
                        text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.recommend_rules))}")

                    if drug_rec.forbid_rules:
                        text_lines.append("  禁用警告:")
                        text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.forbid_rules))}")

                    text_lines.append("")
        else:
            text_lines.append("  暂无强推荐药物")
            text_lines.append("")

        # 弱推荐药物
        text_lines.append("弱推荐药物 (S2):")
        text_lines.append("")

        if S2:
            if group_by_category:
                grouped_drugs = group_drugs_by_category(S2)

                # 先显示字典中定义的类别，按类别名排序
                predefined_categories = [cat for cat in sorted(grouped_drugs.keys())
                                         if cat in self.drug_categories]
                for category_name in predefined_categories:
                    drug_list = grouped_drugs[category_name]
                    if len(drug_list) == 1 and category_name == drug_list[0].drug.value:
                        # 单药类别
                        drug_rec = drug_list[0]
                        text_lines.append(f"• {drug_rec.drug.value}")

                        if drug_rec.recommend_rules:
                            text_lines.append("  推荐理由:")
                            text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.recommend_rules))}")

                        if drug_rec.forbid_rules:
                            text_lines.append("  禁用警告:")
                            text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.forbid_rules))}")
                    else:
                        # 多药类别
                        text_lines.append(f"{category_name}:")

                        for i, drug_rec in enumerate(sorted(drug_list, key=lambda x: x.drug.value), 1):
                            text_lines.append(f"  {i}. {drug_rec.drug.value}")

                            if drug_rec.recommend_rules:
                                text_lines.append("    推荐理由:")
                                text_lines.append(f"      • {format_rules_for_export(sorted(drug_rec.recommend_rules))}")

                            if drug_rec.forbid_rules:
                                text_lines.append("    禁用警告:")
                                text_lines.append(f"      • {format_rules_for_export(sorted(drug_rec.forbid_rules))}")

                    text_lines.append("")

                # 再显示单药类别（不在字典中的药物），按药物名排序
                single_drug_categories = [cat for cat in sorted(grouped_drugs.keys())
                                          if cat not in self.drug_categories]
                for category_name in single_drug_categories:
                    drug_list = grouped_drugs[category_name]
                    drug_rec = drug_list[0]  # 单药类别只有一个药物
                    text_lines.append(f"• {drug_rec.drug.value}")

                    if drug_rec.recommend_rules:
                        text_lines.append("  推荐理由:")
                        text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.recommend_rules))}")

                    if drug_rec.forbid_rules:
                        text_lines.append("  禁用警告:")
                        text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.forbid_rules))}")

                    text_lines.append("")
            else:
                for i, drug_rec in enumerate(sorted(S2, key=lambda x: x.drug.value), 1):
                    text_lines.append(f"{i}. {drug_rec.drug.value}")

                    if drug_rec.recommend_rules:
                        text_lines.append("  推荐理由:")
                        text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.recommend_rules))}")

                    text_lines.append("")
        else:
            text_lines.append("  暂无谨慎/可考虑药物")
            text_lines.append("")

        # 禁用药物
        text_lines.append("禁用药物 (S3):")
        text_lines.append("")

        if S3:
            if group_by_category:
                grouped_drugs = group_drugs_by_category(S3)

                # 先显示字典中定义的类别，按类别名排序
                predefined_categories = [cat for cat in sorted(grouped_drugs.keys())
                                         if cat in self.drug_categories]
                for category_name in predefined_categories:
                    drug_list = grouped_drugs[category_name]
                    if len(drug_list) == 1 and category_name == drug_list[0].drug.value:
                        # 单药类别
                        drug_rec = drug_list[0]
                        text_lines.append(f"• {drug_rec.drug.value}")

                        if drug_rec.forbid_rules:
                            text_lines.append("  禁用理由:")
                            text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.forbid_rules))}")

                        if drug_rec.recommend_rules:
                            text_lines.append("  注意：此药物也有推荐理由:")
                            text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.recommend_rules))}")
                    else:
                        # 多药类别
                        text_lines.append(f"{category_name}:")

                        for i, drug_rec in enumerate(sorted(drug_list, key=lambda x: x.drug.value), 1):
                            text_lines.append(f"  {i}. {drug_rec.drug.value}")

                            if drug_rec.forbid_rules:
                                text_lines.append("    禁用理由:")
                                text_lines.append(f"      • {format_rules_for_export(sorted(drug_rec.forbid_rules))}")

                            if drug_rec.recommend_rules:
                                text_lines.append("    注意：此药物也有推荐理由:")
                                text_lines.append(f"      • {format_rules_for_export(sorted(drug_rec.recommend_rules))}")

                    text_lines.append("")

                # 再显示单药类别（不在字典中的药物），按药物名排序
                single_drug_categories = [cat for cat in sorted(grouped_drugs.keys())
                                          if cat not in self.drug_categories]
                for category_name in single_drug_categories:
                    drug_list = grouped_drugs[category_name]
                    drug_rec = drug_list[0]  # 单药类别只有一个药物
                    text_lines.append(f"• {drug_rec.drug.value}")

                    if drug_rec.forbid_rules:
                        text_lines.append("  禁用理由:")
                        text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.forbid_rules))}")

                    if drug_rec.recommend_rules:
                        text_lines.append("  注意：此药物也有推荐理由:")
                        text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.recommend_rules))}")

                    text_lines.append("")
            else:
                for i, drug_rec in enumerate(sorted(S3, key=lambda x: x.drug.value), 1):
                    text_lines.append(f"{i}. {drug_rec.drug.value}")

                    if drug_rec.forbid_rules:
                        text_lines.append("  禁用理由:")
                        text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.forbid_rules))}")

                    if drug_rec.recommend_rules:
                        text_lines.append("  注意：此药物也有推荐理由:")
                        text_lines.append(f"    • {format_rules_for_export(sorted(drug_rec.recommend_rules))}")

                    text_lines.append("")
        else:
            text_lines.append("  暂无禁用药物")
            text_lines.append("")

        # 统计信息
        text_lines.append("=" * 80)
        text_lines.append("统计信息:")
        text_lines.append(f"  强推荐药物: {len(S1)} 种")
        text_lines.append(f"  谨慎/可考虑药物: {len(S2)} 种")
        text_lines.append(f"  禁用药物: {len(S3)} 种")

        # 按类别统计
        if group_by_category:
            text_lines.append("")
            text_lines.append("按类别统计:")

            def count_by_category(drug_set):
                """统计每个类别的药物数量"""
                category_counts = defaultdict(int)

                for drug_rec in drug_set:
                    drug_name = drug_rec.drug.value
                    found_category = False

                    for category_name, drug_list in self.drug_categories.items():
                        if drug_name in drug_list:
                            category_counts[category_name] += 1
                            found_category = True
                            break

                    if not found_category:
                        category_counts[drug_name] += 1  # 单药类别

                return category_counts

            s1_counts = count_by_category(S1)
            s2_counts = count_by_category(S2)
            s3_counts = count_by_category(S3)

            all_categories = set(s1_counts.keys()) | set(s2_counts.keys()) | set(s3_counts.keys())

            for category in sorted(all_categories):
                s1_count = s1_counts.get(category, 0)
                s2_count = s2_counts.get(category, 0)
                s3_count = s3_counts.get(category, 0)
                total = s1_count + s2_count + s3_count

                if total > 0:
                    text_lines.append(f"  {category}: 强推荐 {s1_count}种, 弱推荐 {s2_count}种, 禁用 {s3_count}种")

        text_lines.append("=" * 80)

        text_content = "\n".join(text_lines)

        # 如果提供了文件名，则保存到文件
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"报告已保存到: {filename}")

        return text_content

    def show_drug_rules(self, drug_category: DrugCategory, patient: Patient = None) -> None:
        """
        显示特定药物的所有相关规则
        如果提供了患者，则显示该患者情况下该药物的状态
        """
        print(f"\n药物规则查询: {drug_category.value}")
        print("-" * 60)

        # 查找所有涉及该药物的规则
        relevant_rules = []
        for rule in self.rules:
            if (drug_category in rule.recommend_drug_categories or
                    drug_category in rule.consider_drug_categories or
                    drug_category in rule.caution_drug_categories or
                    drug_category in rule.forbid_drug_categories):
                relevant_rules.append(rule)

        if not relevant_rules:
            print("未找到该药物的相关规则")
            return

        print(f"找到 {len(relevant_rules)} 条相关规则:")
        print()

        for rule in relevant_rules:
            action_type = []
            if drug_category in rule.recommend_drug_categories:
                action_type.append("推荐")
            if drug_category in rule.consider_drug_categories:
                action_type.append("考虑")
            if drug_category in rule.caution_drug_categories:
                action_type.append("慎用")
            if drug_category in rule.forbid_drug_categories:
                action_type.append("禁用")

            print(f"规则ID: {rule.rule_id}")
            print(f"类型: type={rule.type}")
            print(f"动作: {', '.join(action_type)}")
            print(f"描述: {rule.description}")

            # 显示规则条件
            if rule.required_conditions:
                print(f"必需条件: {', '.join(rule.required_conditions)}")
            if rule.excluded_conditions:
                print(f"排除条件: {', '.join(rule.excluded_conditions)}")

            # 如果提供了患者，显示规则是否适用
            if patient:
                is_applicable = self.check_rule_applicable(patient, rule)
                status = "✅ 适用" if is_applicable else "❌ 不适用"
                print(f"当前患者: {status}")

            print("-" * 40)

        # 如果提供了患者，显示该药物的最终状态
        if patient:
            S1, S2, S3 = self.evaluate_patient(patient)
            drug_rec = None

            # 在所有集合中查找该药物
            for dr in S1 | S2 | S3:
                if dr.drug == drug_category:
                    drug_rec = dr
                    break

            print("\n患者情况下该药物状态:")
            if drug_rec:
                if drug_rec in S1:
                    print("  🏥 强推荐药物 (S1)")
                elif drug_rec in S2:
                    print("  💡 弱推荐药物 (S2)")
                elif drug_rec in S3:
                    print("  🚫 禁用药物 (S3)")

                if drug_rec.recommend_rules:
                    print(f"  推荐规则: {', '.join(sorted(drug_rec.recommend_rules))}")
                if drug_rec.consider_rules:
                    print(f"  可考虑规则: {', '.join(sorted(drug_rec.consider_rules))}")
                if drug_rec.caution_rules:
                    print(f"  谨慎使用规则: {', '.join(sorted(drug_rec.caution_rules))}")
                if drug_rec.forbid_rules:
                    print(f"  禁用规则: {', '.join(sorted(drug_rec.forbid_rules))}")
            else:
                print("  ℹ️  未出现在任何推荐集合中")


def main_refactored():
    # #0001674199_38778 23岁 女 8-4: HbA1c: 10.40%↑ 空腹C-肽: 8-7 2型糖尿病；肝功能异常:脂肪肝；高脂血症；肥胖症；甲状腺结节；肺部结节
    """
        患者信息

        delta_blood_glucose_after_meal: float

        complications: Set[str]:
        白蛋白尿 "HF" "CKD" "ASCVD" "失代偿性HF" "酮症、酮症酸中毒、高渗透性高HHS"
        masld脂肪肝  高脂血症 高血压

        risk_factors: Set[str]:
        低血糖事件hypoglycemia 高血压、血脂异常 胃肠道不良反应gastrointestinal
        "肝功能不全" "肾功能不全" "骨质疏松" "前列腺增生" "生酮饮食"

        history: Set[str]:
        心衰史 "骨折史" 多发性内分泌腺瘤病2型 甲状腺髓样癌 卒中病史"

        family_history: Set[str] = None"
        多发性内分泌腺瘤病2型 甲状腺髓样癌

    """
    # 创建测试患者
    patient_1 = Patient(
        age=23,
        bmi=28.0,
        hba1c=10.4,
        complications={"高脂血症", "脂肪肝"},
        risk_factors=set(),
        renal_function=120,
        hypoglycemia_risk=False,
        duration=1,
        C_peptide_half=1200,
        C_peptide_2=1200,
        C_peptide_3=1200,
        systolic_blood_pressure=120,
        diastolic_blood_pressure=80,
        Elevated_liver=0.8,
        diabetes_type=2,
        delta_blood_glucose_after_meal=2,
        history={},
    )

    patient_2 = Patient(
        age=51,
        bmi=24.46,
        hba1c=12.9,
        complications={"高脂血症", "胃炎", "肾结石"},
        risk_factors=set(),
        renal_function=120,
        hypoglycemia_risk=False,
        duration=1,
        C_peptide_half=418,
        C_peptide_2=1817.00,
        C_peptide_3=1200,
        systolic_blood_pressure=120,
        diastolic_blood_pressure=80,
        Elevated_liver=0.8,
        diabetes_type=2,
        delta_blood_glucose_after_meal=2,
        history={},
    )

    patient_3 = Patient(
        age=38,
        bmi=18.92,
        hba1c=10.7,
        complications={"ASCVD", "胃轻瘫"},
        risk_factors=set("骨质疏松"),
        renal_function=120,
        hypoglycemia_risk=False,
        duration=1,
        C_peptide_half=418,
        C_peptide_2=618.4,
        C_peptide_3=1200,
        systolic_blood_pressure=117,
        diastolic_blood_pressure=60,
        Elevated_liver=0.8,
        diabetes_type=2,
        delta_blood_glucose_after_meal=7,
        history={},
    )

    patient_4 = Patient(
        age=38,
        bmi=18.92,
        hba1c=10.7,
        complications={"CKD", "胃轻瘫"},
        risk_factors=set(),
        renal_function=15,
        hypoglycemia_risk=False,
        duration=1,
        C_peptide_half=418,
        C_peptide_2=618.4,
        C_peptide_3=1200,
        systolic_blood_pressure=117,
        diastolic_blood_pressure=60,
        Elevated_liver=0.8,
        diabetes_type=2,
        delta_blood_glucose_after_meal=7,
        history={"卒中"},
    )

    patient = patient_1

    return patient


if __name__ == "__main__":
    system = DiabetesPrescriptionSystemRefactored()
    patient_ = main_refactored()
    # 打印
    print("\n" + "=" * 80)

    print(f"\n患者信息:")
    print(f"  年龄: {patient_.age}岁, BMI: {patient_.bmi}, HbA1c: {patient_.hba1c}%")
    print(f"  并发症: {', '.join(patient_.complications)}")
    print(f"  肾功能: eGFR {patient_.renal_function}")
    print(f"  风险因素: {', '.join(patient_.risk_factors)}")
    print(f"  患病时长: {patient_.duration}")
    print(f"  C肽（2小时）[beta胰岛细胞功能/胰岛素抵抗情况]:{patient_.C_peptide_2}")
    print(f"  餐前餐后血糖变化: delta {patient_.delta_blood_glucose_after_meal}")
    print(
        f"  既往史: {', '.join(patient_.history) if patient_.history else {} }, 家族史: {', '.join(patient_.family_history) if patient_.family_history else {} }")
    print(f"  ")

    # 生成处方计划
    plan = system.evaluate_patient(patient_)
    # print(plan)
    # 1. 显示推荐结果（控制台输出）
    system.display_recommendations(patient_)

    # 2. 获取结构化报告（用于程序处理）
    report = system.get_detailed_report(patient_)
    print(f"强推荐药物数量: {len(report['S1'])}")

    # 3. 导出为文本文件
    text_report = system.export_to_text(patient_, "药物推荐报告.txt")

    # 4. 查询特定药物的规则
    system.show_drug_rules(DrugCategory.MET, patient_)
