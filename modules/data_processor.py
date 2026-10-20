# modules/data_processor.py
"""
数据处理模块
包含缺失值处理、编码转换、数据标准化等功能
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class DataProcessor:
    """
    数据处理器类

    处理流程: 删除目标缺失行 → 识别列类型 → 日期转数值 → 填充缺失值
             → 分类变量编码 → 目标编码 → 标准化
    所有拟合得到的规则都会保存，transform() 可对新数据复用同一套处理。
    """

    def __init__(self):
        self.label_encoders = {}  # 存储分类变量的编码器
        self.scaler = None  # 存储标准化器
        self.fill_values = {}  # 每列的缺失填充值
        self.feature_names = []  # 特征名称
        self.target_column = None
        self.target_encoder = None  # 目标变量编码器（分类问题）
        self.numeric_columns = []  # 数值列
        self.categorical_columns = []  # 分类列
        self.datetime_columns = []  # 日期列（转换为距 1970 年的天数）
        self.scaling_method = None  # 标准化方法
        self.problem_type = None

    # ------------------------------------------------------------------
    # 拟合
    # ------------------------------------------------------------------
    def fit_transform(self, df, target_column, feature_columns, scaling_method="none",
                      problem_type=None):
        """
        拟合并转换数据

        参数:
            df: 原始数据框
            target_column: 目标列名
            feature_columns: 特征列名列表
            scaling_method: 标准化方法 ("none", "zscore", "minmax")
            problem_type: "classification" / "regression"；为 None 时按目标列类型推断

        返回:
            X: 处理后的特征数据 (numpy 数组)
            y: 处理后的目标数据 (numpy 数组)
            processing_log: 处理日志
        """
        if not feature_columns:
            raise ValueError("请至少选择一个特征列")
        if target_column in feature_columns:
            raise ValueError(f"目标列 '{target_column}' 不能同时作为特征列")

        processing_log = []
        self.feature_names = list(feature_columns)
        self.target_column = target_column
        self.scaling_method = scaling_method
        self.label_encoders = {}
        self.fill_values = {}
        self.scaler = None
        self.target_encoder = None

        data = df[self.feature_names + [target_column]].copy()
        processing_log.append(f"原始数据维度: {data.shape}")

        # 0. 删除目标值缺失的行
        target_na = data[target_column].isna()
        if target_na.any():
            data = data.loc[~target_na].reset_index(drop=True)
            processing_log.append(f"删除目标列缺失的样本: {int(target_na.sum())} 行")
        if len(data) < 10:
            raise ValueError(f"有效样本仅 {len(data)} 行，至少需要 10 行才能训练")

        # 1. 识别列类型
        self.numeric_columns, self.categorical_columns, self.datetime_columns = [], [], []
        for col in self.feature_names:
            series = data[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                self.datetime_columns.append(col)
            elif pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
                self.numeric_columns.append(col)
            else:
                self.categorical_columns.append(col)

        processing_log.append(f"数值型特征: {len(self.numeric_columns)} 列")
        processing_log.append(f"分类型特征: {len(self.categorical_columns)} 列")
        if self.datetime_columns:
            processing_log.append(f"日期型特征: {len(self.datetime_columns)} 列（转换为天数）")

        data = self._coerce_types(data)

        # 2. 处理缺失值：数值列中位数，分类列众数
        for col in self.numeric_columns + self.datetime_columns:
            median = data[col].median()
            self.fill_values[col] = 0.0 if pd.isna(median) else float(median)
        for col in self.categorical_columns:
            mode = data[col].mode(dropna=True)
            self.fill_values[col] = mode.iloc[0] if not mode.empty else "缺失"

        n_missing = int(data[self.feature_names].isna().sum().sum())
        data[self.feature_names] = data[self.feature_names].fillna(self.fill_values)
        if n_missing:
            processing_log.append(f"缺失值填充: {n_missing} 个（数值列用中位数，分类列用众数）")

        # 3. 编码分类变量
        for col in self.categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            self.label_encoders[col] = le
            processing_log.append(f"列 '{col}' 编码: {len(le.classes_)} 个类别")

        # 4. 处理目标变量
        y_raw = data[target_column]
        if problem_type is None:
            is_numeric_target = pd.api.types.is_numeric_dtype(y_raw) and not pd.api.types.is_bool_dtype(y_raw)
            problem_type = "regression" if is_numeric_target and y_raw.nunique() > 20 else "classification"
        self.problem_type = problem_type

        if problem_type == "classification":
            # 统一编码为 0..K-1，XGBoost 等模型要求类别从 0 开始连续
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(self._target_as_labels(y_raw))
            n_classes = len(self.target_encoder.classes_)
            if n_classes < 2:
                raise ValueError("目标列只有 1 个类别，无法训练分类模型")
            processing_log.append(f"目标变量编码: {n_classes} 个类别")
        else:
            y_numeric = pd.to_numeric(y_raw, errors="coerce")
            if y_numeric.isna().mean() > 0.05:
                raise ValueError(
                    f"目标列 '{target_column}' 不是数值型，无法进行回归。请改为分类问题或选择其他目标列"
                )
            if y_numeric.isna().any():
                keep = ~y_numeric.isna()
                processing_log.append(f"删除目标无法转为数值的样本: {int((~keep).sum())} 行")
                data, y_numeric = data.loc[keep], y_numeric.loc[keep]
            y = y_numeric.astype(float).values

        # 5. 提取特征
        X = data[self.feature_names].astype(float).values

        # 6. 数据标准化
        X = self._fit_scaler(X, scaling_method, processing_log)

        processing_log.append(f"处理后数据维度: X={X.shape}, y={y.shape}")
        return X, y, processing_log

    def _fit_scaler(self, X, scaling_method, processing_log):
        if scaling_method == "zscore":
            self.scaler = StandardScaler()
            processing_log.append("数据标准化: Z-score 标准化")
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
            processing_log.append("数据标准化: MinMax 归一化")
        else:
            processing_log.append("数据标准化: 无")
            return X
        return self.scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # 转换新数据
    # ------------------------------------------------------------------
    def transform(self, df):
        """
        使用已拟合的处理器转换新数据

        参数:
            df: 新数据框

        返回:
            X: 处理后的特征数据
        """
        missing_cols = [col for col in self.feature_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"新数据缺少以下列: {missing_cols}")

        data = self._coerce_types(df[self.feature_names].copy())
        data = data.fillna(self.fill_values)

        # 编码分类变量，未见过的类别编码为 -1
        for col in self.categorical_columns:
            le = self.label_encoders[col]
            mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
            data[col] = data[col].astype(str).map(mapping).fillna(-1).astype(int)

        X = data[self.feature_names].astype(float).values

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X

    def get_unseen_categories(self, df):
        """返回新数据中训练时未出现过的类别，用于给出预测提示"""
        unseen = {}
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            known = set(self.label_encoders[col].classes_)
            values = set(df[col].dropna().astype(str)) - known
            if values:
                unseen[col] = sorted(values)[:10]
        return unseen

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _coerce_types(self, data):
        """把各列统一为拟合时识别的类型"""
        for col in self.numeric_columns:
            series = data[col]
            if pd.api.types.is_bool_dtype(series):
                series = series.astype(float)
            data[col] = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        for col in self.datetime_columns:
            parsed = pd.to_datetime(data[col], errors="coerce")
            data[col] = (parsed - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
        for col in self.categorical_columns:
            series = data[col]
            data[col] = series.where(series.isna(), series.astype(str).str.strip())
        return data

    @staticmethod
    def _target_as_labels(y):
        """把目标列转为可读的字符串标签（整数型浮点数去掉 .0）"""
        if pd.api.types.is_float_dtype(y) and np.allclose(y, np.round(y)):
            y = y.astype(int)
        return y.astype(str)

    def get_feature_names(self):
        """获取特征名称"""
        return self.feature_names

    def get_target_classes(self):
        """获取目标变量的类别（分类问题）"""
        if self.target_encoder is not None:
            return self.target_encoder.classes_
        return None

    def inverse_transform_target(self, y):
        """反转目标变量编码"""
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(np.asarray(y).astype(int).ravel())
        return y
