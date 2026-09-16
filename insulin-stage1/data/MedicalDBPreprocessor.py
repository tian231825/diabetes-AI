# -*- encoding: utf-8 -*-
"""
@File    : MedicalDBPreprocessor.py
@Time    : 2026/1/23 21:55
@Author  : junruitian
@Software: PyCharm
"""
import numpy as np
import torch


class MedicalDBPreprocessor:
    """
        医疗数据预处理，专门处理缺失值。
        """
    def __init__(self):
        a = 1

    def detect_missing_values(self, data, missing_tokens=[np.nan, None, -999, 'NA', '']):
        """
        检测数据中的缺失值。

        返回:
            missing_mask: 1表示有效，0表示缺失
            missing_summary: 缺失统计
        """
        if isinstance(data, torch.Tensor):
            # PyTorch张量处理
            return self._detect_missing_tensor(data, missing_tokens)
        elif isinstance(data, np.ndarray):
            # NumPy数组处理
            return self._detect_missing_numpy(data, missing_tokens)
        else:
            # 其他类型转换为NumPy数组
            data_array = np.array(data)
            return self.detect_missing_values(data_array, missing_tokens)

    @staticmethod
    def _detect_missing_numpy(data, missing_tokens):
        """
        专门处理NumPy数组的缺失值检测。
        """
        # 初始掩码：全部有效
        missing_mask = np.ones_like(data, dtype=bool)

        for token in missing_tokens:
            if token is None:
                # None通常不会出现在NumPy数组中，跳过
                continue
            elif token is np.nan:
                # 检测NaN值
                missing_mask = missing_mask & (~np.isnan(data))
            elif isinstance(token, str):
                # 字符串标记：只有当数据是字符串类型时才检测
                if data.dtype.kind in {'U', 'S', 'O'}:  # 字符串或对象类型
                    missing_mask = missing_mask & (data != token)
            else:
                # 数值标记（如-999）
                try:
                    missing_mask = missing_mask & (data != token)
                except (TypeError, ValueError):
                    # Ignore markers that cannot be converted to the target type.
                    continue

        # 计算缺失统计
        missing_rate = 1.0 - missing_mask.mean()
        per_feature_missing = 1.0 - missing_mask.mean(axis=0)

        summary = {
            'overall_missing_rate': missing_rate,
            'per_feature_missing': per_feature_missing,
            'complete_cases': missing_mask.all(axis=1).sum(),
        }

        return missing_mask, summary

    @staticmethod
    def _detect_missing_tensor(data, missing_tokens):
        """
        专门处理PyTorch张量的缺失值检测。
        """
        # 初始掩码：全部有效
        missing_mask = torch.ones_like(data, dtype=torch.bool)

        for token in missing_tokens:
            if token is None:
                continue
            elif token is np.nan:
                missing_mask = missing_mask & (~torch.isnan(data))
            elif isinstance(token, str):
                # PyTorch张量通常不包含字符串，跳过
                continue
            else:
                try:
                    missing_mask = missing_mask & (data != token)
                except (TypeError, RuntimeError):
                    continue

        # 计算缺失统计
        missing_rate = 1.0 - missing_mask.float().mean()
        per_feature_missing = 1.0 - missing_mask.float().mean(dim=0)

        summary = {
            'overall_missing_rate': missing_rate,
            'per_feature_missing': per_feature_missing,
            'complete_cases': missing_mask.all(dim=1).sum().item(),
        }

        return missing_mask, summary

    @staticmethod
    def impute_missing_values(data, missing_mask, strategy='mean'):
        """
        填充缺失值。

        策略:
            - 'mean': 用特征均值填充
            - 'median': 用特征中位数填充
            - 'zero': 用0填充
            - 'knn': K近邻填充
        """
        data_filled = data.copy() if isinstance(data, np.ndarray) else data.clone()

        if strategy == 'mean':
            for col in range(data.shape[1]):
                col_data = data[:, col]
                col_mask = missing_mask[:, col]
                if col_mask.any():  # 如果有有效值
                    mean_val = col_data[col_mask].mean()
                    data_filled[~col_mask, col] = mean_val
                else:
                    data_filled[:, col] = 0  # 如果全部缺失，填0

        elif strategy == 'median':
            for col in range(data.shape[1]):
                col_data = data[:, col]
                col_mask = missing_mask[:, col]
                if col_mask.any():
                    median_val = np.median(col_data[col_mask]) if isinstance(data, np.ndarray) \
                        else torch.median(col_data[col_mask])
                    data_filled[~col_mask, col] = median_val
                else:
                    data_filled[:, col] = 0

        elif strategy == 'zero':
            data_filled[~missing_mask] = 0

        elif strategy == 'knn':
            # 简单KNN填充（实际项目中可以使用更复杂的实现）
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            data_filled = imputer.fit_transform(data)

        return data_filled

    @staticmethod
    def create_missing_indicators(data, missing_mask):
        """
        Create missing-value indicator features.
        """
        # 1. 转换为NumPy数组（确保数据类型）
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.array(data, dtype=np.float32)

        if isinstance(missing_mask, torch.Tensor):
            missing_mask_np = missing_mask.detach().cpu().numpy()
        else:
            missing_mask_np = np.array(missing_mask)

        # 2. 确保数据维度匹配
        if data_np.shape != missing_mask_np.shape:
            raise ValueError(f"数据形状 {data_np.shape} 与掩码形状 {missing_mask_np.shape} 不匹配")

        # Convert the supplied mask to boolean values.
        if missing_mask_np.dtype == bool:
            bool_mask = missing_mask_np
        elif np.issubdtype(missing_mask_np.dtype, np.integer) or np.issubdtype(missing_mask_np.dtype, np.floating):
            bool_mask = missing_mask_np != 0
        # 方法3：其他情况，尝试转换为布尔型
        else:
            try:
                bool_mask = missing_mask_np.astype(bool)
            except:
                # 最后手段：假设所有值都是有效的
                print("警告：无法转换掩码为布尔型，假设所有值都有效")
                bool_mask = np.ones_like(missing_mask_np, dtype=bool)

        # 4. 填充缺失值（用0填充）
        data_filled = np.where(bool_mask, data_np, 0)

        # 5. 创建缺失指示符（1表示缺失，0表示有效）
        # 注意：这里使用 bool_mask 而不是 missing_mask_np
        missing_indicators = (~bool_mask).astype(np.float32)

        # 6. 组合特征
        combined = np.concatenate([data_filled, missing_indicators], axis=1)

        return {
            'filled_data': data_filled,
            'missing_indicators': missing_indicators,
            'combined': combined,
            'bool_mask': bool_mask  # 返回布尔掩码用于调试
        }
