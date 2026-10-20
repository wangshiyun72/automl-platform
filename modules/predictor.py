"""
模型加载与批量预测模块

该模块把训练后的模型使用流程封装为可复用能力，包括模型文件发现、
输入字段校验、批量预测、预测概率整理、标签反编码和结果导出。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import os
import pickle

import joblib
import numpy as np
import pandas as pd


@dataclass
class PredictionSchema:
    """预测输入字段约束。"""

    feature_names: List[str]
    problem_type: str = "classification"
    target_name: Optional[str] = None
    required_columns: List[str] = field(default_factory=list)
    optional_columns: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.required_columns:
            self.required_columns = list(self.feature_names)

    def validate(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """检查输入数据字段。"""
        columns = set(df.columns)
        required = set(self.required_columns)
        known = set(self.required_columns) | set(self.optional_columns)
        return {
            "missing": sorted(required - columns),
            "extra": sorted(columns - known) if known else [],
            "available": sorted(columns & required),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "problem_type": self.problem_type,
            "target_name": self.target_name,
            "required_columns": self.required_columns,
            "optional_columns": self.optional_columns,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionSchema":
        return cls(
            feature_names=list(data.get("feature_names", [])),
            problem_type=data.get("problem_type", "classification"),
            target_name=data.get("target_name"),
            required_columns=list(data.get("required_columns", [])),
            optional_columns=list(data.get("optional_columns", [])),
        )


@dataclass
class PredictionResult:
    """预测结果对象。"""

    predictions: np.ndarray
    dataframe: pd.DataFrame
    probabilities: Optional[np.ndarray] = None
    probability_columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return self.dataframe.copy()

    def summary(self, prediction_column: str = "预测结果") -> Dict[str, Any]:
        """生成预测结果摘要。"""
        result = {
            "row_count": len(self.dataframe),
            "created_at": self.metadata.get("created_at"),
            "has_probabilities": self.probabilities is not None,
            "warnings": self.warnings,
        }
        if prediction_column in self.dataframe.columns:
            series = self.dataframe[prediction_column]
            if pd.api.types.is_numeric_dtype(series):
                result["prediction_mean"] = float(series.mean())
                result["prediction_std"] = float(series.std()) if len(series) > 1 else 0.0
                result["prediction_min"] = float(series.min())
                result["prediction_max"] = float(series.max())
            else:
                result["prediction_distribution"] = series.value_counts().to_dict()
        return result


class BatchPredictor:
    """
    批量预测器。

    支持直接传入模型对象，也支持从 joblib/pickle 文件加载模型。
    如果提供 DataProcessor 或 FeatureEngineer，会在预测前自动复用训练时
    的特征处理流程。
    """

    def __init__(
        self,
        model: Any,
        schema: Optional[PredictionSchema] = None,
        feature_names: Optional[Sequence[str]] = None,
        data_processor: Optional[Any] = None,
        feature_engineer: Optional[Any] = None,
        problem_type: str = "classification",
        prediction_column: str = "预测结果",
    ):
        if model is None:
            raise ValueError("model 不能为空")
        self.model = model
        self.data_processor = data_processor
        self.feature_engineer = feature_engineer
        self.prediction_column = prediction_column
        self.problem_type = problem_type
        self.schema = schema or PredictionSchema(
            feature_names=self._resolve_feature_names(feature_names, data_processor, feature_engineer),
            problem_type=problem_type,
        )
        self.last_result: Optional[PredictionResult] = None

    @classmethod
    def from_model_file(
        cls,
        model_path: str,
        schema: Optional[PredictionSchema] = None,
        feature_names: Optional[Sequence[str]] = None,
        data_processor: Optional[Any] = None,
        feature_engineer: Optional[Any] = None,
        problem_type: str = "classification",
    ) -> "BatchPredictor":
        """从模型文件创建预测器。"""
        model = ModelArtifactManager.load_model(model_path)
        return cls(
            model=model,
            schema=schema,
            feature_names=feature_names,
            data_processor=data_processor,
            feature_engineer=feature_engineer,
            problem_type=problem_type,
        )

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str,
        model_name: Optional[str] = None,
        problem_type: str = "classification",
    ) -> "BatchPredictor":
        """从模型产物目录加载最新模型。"""
        manager = ModelArtifactManager(artifact_dir)
        model_path = manager.find_latest_model(model_name=model_name)
        if not model_path:
            raise FileNotFoundError(f"未在目录中找到模型文件: {artifact_dir}")

        config = manager.load_sidecar_config(model_path)
        feature_names = config.get("feature_names") if config else None
        schema = PredictionSchema(
            feature_names=feature_names or [],
            problem_type=problem_type,
            target_name=config.get("target_column") if config else None,
        )
        return cls.from_model_file(
            model_path=model_path,
            schema=schema,
            problem_type=problem_type,
        )

    def predict(self, df: pd.DataFrame, include_input: bool = True) -> PredictionResult:
        """执行批量预测。"""
        self._validate_dataframe(df)
        warnings = self._collect_schema_warnings(df)
        X = self.prepare_features(df)

        predictions = self.model.predict(X)
        decoded_predictions = self._decode_predictions(predictions)
        probabilities, probability_columns = self._predict_probabilities(X)

        result_df = pd.DataFrame(index=df.index)
        if include_input:
            result_df = df.copy()
        result_df[self.prediction_column] = decoded_predictions

        if probabilities is not None:
            proba_df = pd.DataFrame(probabilities, columns=probability_columns, index=df.index)
            result_df = pd.concat([result_df, proba_df], axis=1)
            if probabilities.ndim == 2 and probabilities.shape[1] > 0:
                result_df["预测置信度"] = probabilities.max(axis=1)

        metadata = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "row_count": len(df),
            "feature_count": X.shape[1] if hasattr(X, "shape") else None,
            "problem_type": self.problem_type,
            "model_class": type(self.model).__name__,
        }
        result = PredictionResult(
            predictions=np.asarray(decoded_predictions),
            probabilities=probabilities,
            probability_columns=probability_columns,
            dataframe=result_df,
            metadata=metadata,
            warnings=warnings,
        )
        self.last_result = result
        return result

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """预测单条样本。"""
        df = pd.DataFrame([record])
        result = self.predict(df, include_input=True)
        return result.to_dataframe().iloc[0].to_dict()

    def prepare_features(self, df: pd.DataFrame) -> Any:
        """按训练时规则准备模型输入矩阵。"""
        working = df.copy()
        if self.feature_engineer is not None:
            working = self.feature_engineer.transform(working)

        if self.data_processor is not None:
            return self.data_processor.transform(working)

        feature_names = self.schema.feature_names
        if feature_names:
            missing = [col for col in feature_names if col not in working.columns]
            if missing:
                raise ValueError(f"预测数据缺少特征列: {missing}")
            working = working[feature_names]

        return working.values if isinstance(working, pd.DataFrame) else working

    def build_input_template(self, rows: int = 3) -> pd.DataFrame:
        """生成预测输入模板。"""
        rows = max(int(rows), 1)
        data = {}
        for name in self.schema.required_columns:
            data[name] = [None for _ in range(rows)]
        return pd.DataFrame(data)

    def explain_schema(self) -> pd.DataFrame:
        """返回预测字段说明。"""
        rows = []
        for name in self.schema.required_columns:
            rows.append({
                "字段名": name,
                "是否必填": "是",
                "说明": "训练模型需要的输入特征",
            })
        for name in self.schema.optional_columns:
            rows.append({
                "字段名": name,
                "是否必填": "否",
                "说明": "可选辅助字段，不一定进入模型",
            })
        return pd.DataFrame(rows)

    def export_predictions(self, path: str, result: Optional[PredictionResult] = None) -> str:
        """导出预测结果，支持 csv、xlsx、json。"""
        result = result or self.last_result
        if result is None:
            raise RuntimeError("暂无预测结果可导出")

        df = result.to_dataframe()
        ModelArtifactManager.ensure_parent_dir(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            df.to_csv(path, index=False, encoding="utf-8-sig")
        elif ext in {".xlsx", ".xls"}:
            df.to_excel(path, index=False)
        elif ext == ".json":
            df.to_json(path, orient="records", force_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {ext}")
        return path

    def export_schema(self, path: str) -> str:
        """导出预测输入 schema。"""
        ModelArtifactManager.ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.schema.to_dict(), file, ensure_ascii=False, indent=2)
        return path

    def _collect_schema_warnings(self, df: pd.DataFrame) -> List[str]:
        validation = self.schema.validate(df)
        warnings: List[str] = []
        if validation["missing"]:
            raise ValueError(f"预测输入缺少必填字段: {validation['missing']}")
        if validation["extra"]:
            warnings.append(f"输入数据包含未使用字段: {validation['extra']}")
        return warnings

    def _predict_probabilities(self, X: Any) -> Tuple[Optional[np.ndarray], List[str]]:
        if self.problem_type != "classification":
            return None, []
        if not hasattr(self.model, "predict_proba"):
            return None, []

        try:
            probabilities = self.model.predict_proba(X)
        except Exception:
            return None, []

        classes = getattr(self.model, "classes_", None)
        if classes is None:
            classes = list(range(probabilities.shape[1])) if probabilities.ndim == 2 else [0]

        decoded_classes = self._decode_predictions(np.asarray(classes))
        columns = [
            f"概率_{str(label)}"
            for label in decoded_classes
        ]
        return np.asarray(probabilities), columns

    def _decode_predictions(self, predictions: Sequence[Any]) -> np.ndarray:
        if self.data_processor is not None and hasattr(self.data_processor, "inverse_transform_target"):
            try:
                return np.asarray(self.data_processor.inverse_transform_target(predictions))
            except Exception:
                return np.asarray(predictions)
        return np.asarray(predictions)

    @staticmethod
    def _resolve_feature_names(
        feature_names: Optional[Sequence[str]],
        data_processor: Optional[Any],
        feature_engineer: Optional[Any],
    ) -> List[str]:
        if feature_names is not None:
            return list(feature_names)
        if feature_engineer is not None and hasattr(feature_engineer, "get_output_columns"):
            return list(feature_engineer.get_output_columns())
        if data_processor is not None and hasattr(data_processor, "get_feature_names"):
            return list(data_processor.get_feature_names())
        return []

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 必须是 pandas.DataFrame")
        if df.empty:
            raise ValueError("预测数据为空")


class ModelArtifactManager:
    """
    模型产物管理器。

    负责发现训练阶段保存的模型、配置和辅助文件。保存格式兼容当前项目中
    save_model 生成的 .pkl、.joblib 与 *_config.json 文件。
    """

    MODEL_EXTENSIONS = {".pkl", ".joblib"}

    def __init__(self, artifact_dir: str):
        self.artifact_dir = os.path.abspath(artifact_dir)

    def list_model_files(self, model_name: Optional[str] = None) -> List[str]:
        """列出目录中的模型文件。"""
        if not os.path.isdir(self.artifact_dir):
            return []
        files: List[str] = []
        for root, _, names in os.walk(self.artifact_dir):
            for name in names:
                ext = os.path.splitext(name)[1].lower()
                if ext not in self.MODEL_EXTENSIONS:
                    continue
                if model_name and not name.startswith(model_name):
                    continue
                files.append(os.path.join(root, name))
        return sorted(files, key=lambda path: os.path.getmtime(path), reverse=True)

    def find_latest_model(self, model_name: Optional[str] = None) -> Optional[str]:
        """查找最新模型文件。"""
        files = self.list_model_files(model_name=model_name)
        return files[0] if files else None

    def discover_artifact_sets(self) -> pd.DataFrame:
        """按文件名前缀归并模型产物。"""
        if not os.path.isdir(self.artifact_dir):
            return pd.DataFrame()
        records: Dict[str, Dict[str, Any]] = {}
        for root, _, names in os.walk(self.artifact_dir):
            for name in names:
                path = os.path.join(root, name)
                key = self._artifact_key(name)
                record = records.setdefault(key, {
                    "产物组": key,
                    "目录": root,
                    "模型文件": "",
                    "配置文件": "",
                    "标准化器": "",
                    "标签编码器": "",
                    "更新时间": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S"),
                })
                lower = name.lower()
                if lower.endswith((".pkl", ".joblib")) and "_scaler" not in lower and "_label_encoders" not in lower:
                    record["模型文件"] = path
                elif lower.endswith("_config.json"):
                    record["配置文件"] = path
                elif lower.endswith("_scaler.pkl"):
                    record["标准化器"] = path
                elif lower.endswith("_label_encoders.pkl"):
                    record["标签编码器"] = path
        return pd.DataFrame(records.values())

    def load_sidecar_config(self, model_path: str) -> Dict[str, Any]:
        """加载模型旁路配置 JSON。"""
        directory = os.path.dirname(model_path)
        stem = os.path.splitext(os.path.basename(model_path))[0]
        candidates = [
            os.path.join(directory, f"{stem}_config.json"),
            os.path.join(directory, stem.replace(".joblib", "") + "_config.json"),
        ]
        for name in os.listdir(directory) if os.path.isdir(directory) else []:
            if name.endswith("_config.json") and stem.split("_")[0] in name:
                candidates.append(os.path.join(directory, name))

        for candidate in candidates:
            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as file:
                    return json.load(file)
        return {}

    @staticmethod
    def load_model(model_path: str) -> Any:
        """加载模型文件。"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".joblib":
            return joblib.load(model_path)
        if ext == ".pkl":
            with open(model_path, "rb") as file:
                return pickle.load(file)
        raise ValueError(f"不支持的模型文件格式: {ext}")

    @staticmethod
    def save_prediction_package(
        output_dir: str,
        result: PredictionResult,
        schema: PredictionSchema,
        prefix: str = "prediction",
    ) -> Dict[str, str]:
        """保存预测结果、摘要和输入 schema。"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(output_dir, f"{prefix}_{timestamp}")

        csv_path = f"{base}.csv"
        json_path = f"{base}_summary.json"
        schema_path = f"{base}_schema.json"

        result.to_dataframe().to_csv(csv_path, index=False, encoding="utf-8-sig")
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(result.summary(), file, ensure_ascii=False, indent=2, default=str)
        with open(schema_path, "w", encoding="utf-8") as file:
            json.dump(schema.to_dict(), file, ensure_ascii=False, indent=2)

        return {
            "prediction_csv": csv_path,
            "summary_json": json_path,
            "schema_json": schema_path,
        }

    @staticmethod
    def ensure_parent_dir(path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def _artifact_key(filename: str) -> str:
        name = os.path.splitext(filename)[0]
        suffixes = [
            "_config",
            "_scaler",
            "_label_encoders",
        ]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name


def load_prediction_file(path: str) -> pd.DataFrame:
    """读取待预测文件，支持 csv、xlsx、xls。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        for encoding in ("utf-8", "gbk", "latin1"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"不支持的预测文件格式: {ext}")
