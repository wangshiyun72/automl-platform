# modules/utils.py
"""
工具函数模块
"""

import io
import os
import json
import pickle
import zipfile
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


CSV_ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin1")


def _clean_columns(df):
    """统一列名：转为字符串、去除首尾空白、处理重名和空列名"""
    seen = {}
    new_columns = []
    for i, col in enumerate(df.columns):
        name = str(col).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"列_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        new_columns.append(name)
    df.columns = new_columns
    return df


def load_data(uploaded_file):
    """
    加载数据文件
    支持 CSV, XLSX, XLS 格式；兼容 Streamlit UploadedFile、文件路径和二进制流

    返回:
        (df, error) —— 成功时 error 为 None
    """
    try:
        file_name = getattr(uploaded_file, "name", str(uploaded_file))
        file_extension = file_name.rsplit(".", 1)[-1].lower()

        if isinstance(uploaded_file, (str, os.PathLike)):
            with open(uploaded_file, "rb") as f:
                raw = f.read()
        else:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            raw = uploaded_file.read()

        if not raw:
            return None, "文件为空"

        if file_extension == "csv":
            df, last_error = None, None
            for encoding in CSV_ENCODINGS:
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
                    if df.shape[1] == 1:
                        # 可能是分号 / 制表符分隔，自动嗅探分隔符
                        df = pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=None, engine="python")
                    break
                except (UnicodeDecodeError, pd.errors.ParserError) as e:
                    last_error = e
            if df is None:
                return None, f"CSV 解析失败: {last_error}"
        elif file_extension in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            return None, f"不支持的文件格式: {file_extension}"

        df = df.dropna(how="all").dropna(axis=1, how="all")
        df = _clean_columns(df).reset_index(drop=True)

        if df.empty:
            return None, "文件中没有有效数据"
        if df.shape[1] < 2:
            return None, "数据至少需要 2 列（特征列 + 目标列）"

        return df, None
    except Exception as e:
        return None, f"文件读取错误: {str(e)}"


def detect_problem_type(y, threshold=20):
    """
    自动检测问题类型（分类/回归）

    规则:
        - 非数值型 / 布尔型目标 → 分类
        - 含非整数的浮点数且唯一值较多 → 回归
        - 整数型且唯一值不超过 threshold → 分类
    """
    y = y.dropna()
    if y.empty:
        return "classification"
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"

    unique_values = y.nunique()
    if unique_values <= 2:
        return "classification"

    is_integer_like = np.allclose(y.astype(float), np.round(y.astype(float)))
    if is_integer_like and unique_values <= threshold:
        return "classification"
    if not is_integer_like and unique_values <= min(threshold, 10):
        return "classification"
    return "regression"


def is_numeric_column(series):
    """判断列是否为数值型（布尔型视为数值）"""
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def get_column_types(df):
    """
    获取列的数据类型信息
    """
    column_info = {}
    total = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())

        if pd.api.types.is_datetime64_any_dtype(series):
            col_type = "日期型"
        elif is_numeric_column(series):
            col_type = "数值型"
        else:
            col_type = "分类型"

        column_info[col] = {
            "type": col_type,
            "dtype": str(series.dtype),
            "unique_count": int(series.nunique()),
            "null_count": null_count,
            "null_ratio": f"{null_count / total * 100:.2f}%"
        }

    return column_info


def make_arrow_safe(df):
    """
    将混合类型的 object 列转为字符串，避免 st.dataframe 的 Arrow 序列化错误
    """
    safe = df.copy()
    for col in safe.columns:
        if safe[col].dtype == object:
            non_null = safe[col].dropna()
            if non_null.map(type).nunique() > 1 or (len(non_null) and not isinstance(non_null.iloc[0], str)):
                safe[col] = safe[col].where(safe[col].isna(), safe[col].astype(str))
    return safe


def fig_to_bytes(fig, fmt="png", dpi=300):
    """将 matplotlib 图像导出为字节流，用于下载按钮"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=dpi if fmt == "png" else None,
                bbox_inches="tight", facecolor="white")
    buffer.seek(0)
    return buffer.getvalue()


def df_to_excel_bytes(df, sheet_name="Sheet1"):
    """将 DataFrame 导出为 xlsx 字节流"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buffer.getvalue()


def save_model(model, model_name, save_dir, scaler=None, label_encoders=None,
               feature_names=None, model_params=None, data_processor=None):
    """
    保存模型及相关配置
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{model_name}_{timestamp}"

    # 保存模型 (.pkl 和 .joblib)
    pkl_path = os.path.join(save_dir, f"{prefix}.pkl")
    joblib_path = os.path.join(save_dir, f"{prefix}.joblib")

    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)
    joblib.dump(model, joblib_path)

    if model_params:
        params = model_params
    elif hasattr(model, "get_params"):
        params = model.get_params()
    else:
        params = None

    # 保存配置信息
    config = {
        "model_name": model_name,
        "timestamp": timestamp,
        "feature_names": list(feature_names) if feature_names is not None else None,
        "model_params": params,
    }

    json_path = os.path.join(save_dir, f"{prefix}_config.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)

    # 保存 TXT 配置（易读版本）
    txt_path = os.path.join(save_dir, f"{prefix}_config.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"模型名称: {model_name}\n")
        f.write(f"保存时间: {timestamp}\n")
        f.write(f"特征列: {config['feature_names']}\n")
        f.write(f"模型参数: {params}\n")

    paths = {
        "pkl_path": pkl_path,
        "joblib_path": joblib_path,
        "json_path": json_path,
        "txt_path": txt_path
    }

    if scaler is not None:
        paths["scaler_path"] = os.path.join(save_dir, f"{prefix}_scaler.pkl")
        with open(paths["scaler_path"], 'wb') as f:
            pickle.dump(scaler, f)

    if label_encoders:
        paths["label_encoders_path"] = os.path.join(save_dir, f"{prefix}_label_encoders.pkl")
        with open(paths["label_encoders_path"], 'wb') as f:
            pickle.dump(label_encoders, f)

    # 完整的数据处理器（新数据预测时直接 transform 即可）
    if data_processor is not None:
        paths["processor_path"] = os.path.join(save_dir, f"{prefix}_processor.joblib")
        joblib.dump(data_processor, paths["processor_path"])

    return paths


def zip_paths(paths):
    """把多个文件打包为 zip 字节流"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            zf.write(path, arcname=os.path.basename(path))
    return buffer.getvalue()


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
        self.entries = []

    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        self.entries.append((timestamp, level, str(message)))
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

    def get_entries(self):
        return list(self.entries)

    def clear(self):
        self.logs = []
        self.entries = []
