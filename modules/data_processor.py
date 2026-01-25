# modules/data_processor.py
"""
数据处理模块
包含缺失值处理、编码转换、数据标准化等功能
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


class DataProcessor:
    """
    数据处理器类
    """

    def __init__(self):
        self.label_encoders = {}  # 存储分类变量的编码器
        self.scaler = None  # 存储标准化器
        self.imputers = {}  # 存储缺失值填充器
        self.feature_names = []  # 特征名称
        self.target_encoder = None  # 目标变量编码器（分类问题）
        self.numeric_columns = []  # 数值列
        self.categorical_columns = []  # 分类列
        self.scaling_method = None  # 标准化方法

    def fit_transform(self, df, target_column, feature_columns, scaling_method="none"):
        """
        拟合并转换数据

        参数:
            df: 原始数据框
            target_column: 目标列名
            feature_columns: 特征列名列表
            scaling_method: 标准化方法 ("none", "zscore", "minmax")

        返回:
            X: 处理后的特征数据
            y: 处理后的目标数据
            processing_log: 处理日志
        """
        processing_log = []
        self.feature_names = feature_columns.copy()
        self.scaling_method = scaling_method

        # 复制数据避免修改原始数据
        data = df[feature_columns + [target_column]].copy()
        processing_log.append(f"原始数据维度: {data.shape}")

        # 1. 识别列类型
        self.numeric_columns = []
        self.categorical_columns = []

        for col in feature_columns:
            if data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                self.numeric_columns.append(col)
            else:
                self.categorical_columns.append(col)

        processing_log.append(f"数值型特征: {len(self.numeric_columns)} 列")
        processing_log.append(f"分类型特征: {len(self.categorical_columns)} 列")

        # 2. 处理缺失值
        # 数值列用中位数填充
        if self.numeric_columns:
            num_imputer = SimpleImputer(strategy='median')
            data[self.numeric_columns] = num_imputer.fit_transform(data[self.numeric_columns])
            self.imputers['numeric'] = num_imputer
            processing_log.append("数值型缺失值: 使用中位数填充")

        # 分类列用众数填充
        if self.categorical_columns:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            data[self.categorical_columns] = cat_imputer.fit_transform(data[self.categorical_columns])
            self.imputers['categorical'] = cat_imputer
            processing_log.append("分类型缺失值: 使用众数填充")

        # 3. 编码分类变量
        for col in self.categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            self.label_encoders[col] = le
            processing_log.append(f"列 '{col}' 编码: {len(le.classes_)} 个类别")

        # 4. 处理目标变量
        y = data[target_column].copy()

        # 检查目标变量是否需要编码
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y.astype(str))
            processing_log.append(f"目标变量编码: {len(self.target_encoder.classes_)} 个类别")
        else:
            y = y.values

        # 5. 提取特征
        X = data[feature_columns].values

        # 6. 数据标准化
        if scaling_method == "zscore":
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            processing_log.append("数据标准化: Z-score 标准化")
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
            X = self.scaler.fit_transform(X)
            processing_log.append("数据标准化: MinMax 归一化")
        else:
            processing_log.append("数据标准化: 无")

        processing_log.append(f"处理后数据维度: X={X.shape}, y={y.shape}")

        return X, y, processing_log

    def transform(self, df):
        """
        使用已拟合的处理器转换新数据

        参数:
            df: 新数据框

        返回:
            X: 处理后的特征数据
        """
        # 确保包含所有需要的特征列
        missing_cols = set(self.feature_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"新数据缺少以下列: {missing_cols}")

        data = df[self.feature_names].copy()

        # 1. 处理缺失值
        if self.numeric_columns and 'numeric' in self.imputers:
            # 找出存在于新数据中的数值列
            existing_num_cols = [col for col in self.numeric_columns if col in data.columns]
            if existing_num_cols:
                data[existing_num_cols] = self.imputers['numeric'].transform(data[existing_num_cols])

        if self.categorical_columns and 'categorical' in self.imputers:
            existing_cat_cols = [col for col in self.categorical_columns if col in data.columns]
            if existing_cat_cols:
                data[existing_cat_cols] = self.imputers['categorical'].transform(data[existing_cat_cols])

        # 2. 编码分类变量
        for col in self.categorical_columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                # 处理未见过的类别
                data[col] = data[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # 3. 提取特征
        X = data[self.feature_names].values

        # 4. 数据标准化
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X

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
            return self.target_encoder.inverse_transform(y)
        return y