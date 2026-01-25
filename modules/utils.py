# modules/utils.py
"""
工具函数模块
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import json
import joblib
import pickle


def load_data(uploaded_file):
    """
    加载数据文件
    支持 CSV, XLSX, XLS 格式
    """
    try:
        file_name = uploaded_file.name
        file_extension = file_name.split('.')[-1].lower()

        if file_extension == 'csv':
            # 尝试不同编码
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except:
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, encoding='gbk')
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin1')
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            return None, f"不支持的文件格式: {file_extension}"

        return df, None
    except Exception as e:
        return None, f"文件读取错误: {str(e)}"


def detect_problem_type(y, threshold=20):
    """
    自动检测问题类型（分类/回归）
    """
    unique_values = y.nunique()
    if unique_values <= threshold:
        return "classification"
    else:
        return "regression"


def get_column_types(df):
    """
    获取列的数据类型信息
    """
    column_info = {}
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()

        if dtype in ['int64', 'float64']:
            col_type = "数值型"
        else:
            col_type = "分类型"

        column_info[col] = {
            "type": col_type,
            "dtype": str(dtype),
            "unique_count": unique_count,
            "null_count": null_count,
            "null_ratio": f"{null_count / len(df) * 100:.2f}%"
        }

    return column_info


def save_model(model, model_name, save_dir, scaler=None, label_encoders=None,
               feature_names=None, model_params=None):
    """
    保存模型及相关配置
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存模型 (.pkl 和 .joblib)
    pkl_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pkl")
    joblib_path = os.path.join(save_dir, f"{model_name}_{timestamp}.joblib")

    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)
    joblib.dump(model, joblib_path)

    # 保存配置信息
    config = {
        "model_name": model_name,
        "timestamp": timestamp,
        "feature_names": feature_names,
        "model_params": model_params if model_params else str(model.get_params()) if hasattr(model,
                                                                                             'get_params') else None
    }

    # 保存 JSON 配置
    json_path = os.path.join(save_dir, f"{model_name}_{timestamp}_config.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)

    # 保存 TXT 配置（易读版本）
    txt_path = os.path.join(save_dir, f"{model_name}_{timestamp}_config.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"模型名称: {model_name}\n")
        f.write(f"保存时间: {timestamp}\n")
        f.write(f"特征列: {feature_names}\n")
        f.write(f"模型参数: {config['model_params']}\n")

    # 保存标准化器
    if scaler is not None:
        scaler_path = os.path.join(save_dir, f"{model_name}_{timestamp}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    # 保存标签编码器
    if label_encoders is not None:
        le_path = os.path.join(save_dir, f"{model_name}_{timestamp}_label_encoders.pkl")
        with open(le_path, 'wb') as f:
            pickle.dump(label_encoders, f)

    return {
        "pkl_path": pkl_path,
        "joblib_path": joblib_path,
        "json_path": json_path,
        "txt_path": txt_path
    }


def format_metrics(metrics_dict, problem_type):
    """
    格式化评估指标
    """
    formatted = []
    if problem_type == "classification":
        formatted.append(f"准确率 (Accuracy): {metrics_dict.get('accuracy', 0):.4f}")
        formatted.append(f"精确率 (Precision): {metrics_dict.get('precision', 0):.4f}")
        formatted.append(f"召回率 (Recall): {metrics_dict.get('recall', 0):.4f}")
        formatted.append(f"F1分数: {metrics_dict.get('f1', 0):.4f}")
        if 'auc' in metrics_dict:
            formatted.append(f"AUC: {metrics_dict.get('auc', 0):.4f}")
    else:
        formatted.append(f"R² 分数: {metrics_dict.get('r2', 0):.4f}")
        formatted.append(f"均方误差 (MSE): {metrics_dict.get('mse', 0):.4f}")
        formatted.append(f"均方根误差 (RMSE): {metrics_dict.get('rmse', 0):.4f}")
        formatted.append(f"平均绝对误差 (MAE): {metrics_dict.get('mae', 0):.4f}")

    return "\n".join(formatted)


class Logger:
    """
    日志记录器
    """

    def __init__(self):
        self.logs = []

    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        return log_entry

    def info(self, message):
        return self.log(message, "INFO")

    def warning(self, message):
        return self.log(message, "WARNING")

    def error(self, message):
        return self.log(message, "ERROR")

    def success(self, message):
        return self.log(message, "SUCCESS")

    def get_logs(self):
        return "\n".join(self.logs)

    def clear(self):
        self.logs = []