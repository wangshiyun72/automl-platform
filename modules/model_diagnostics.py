"""
模型诊断模块

在基础训练指标之外，提供更细粒度的分类和回归模型诊断能力，包括
阈值分析、分类别指标、混淆矩阵、概率校准、提升表、残差分析、
误差分箱、训练/测试差异和自动诊断建议。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class DiagnosticResult:
    """单次诊断结果。"""

    problem_type: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def add_table(self, name: str, table: pd.DataFrame) -> None:
        self.tables[name] = table

    def to_summary_frame(self) -> pd.DataFrame:
        rows = [
            {"指标": key, "值": value}
            for key, value in self.metrics.items()
            if not isinstance(value, (list, dict, np.ndarray, pd.DataFrame))
        ]
        return pd.DataFrame(rows)


class ModelDiagnostics:
    """
    模型诊断器。

    参数:
        problem_type: classification 或 regression
    """

    def __init__(self, problem_type: str = "classification"):
        if problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type 必须是 classification 或 regression")
        self.problem_type = problem_type

    def evaluate(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        y_proba: Optional[Sequence[Any]] = None,
        model_name: str = "model",
    ) -> DiagnosticResult:
        """根据真实值、预测值和可选概率执行诊断。"""
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        result = DiagnosticResult(problem_type=self.problem_type)
        result.metrics["model_name"] = model_name
        result.metrics["sample_count"] = int(len(y_true_arr))

        if self.problem_type == "classification":
            self._evaluate_classification(y_true_arr, y_pred_arr, y_proba, result)
        else:
            self._evaluate_regression(y_true_arr, y_pred_arr, result)

        result.recommendations = self._build_recommendations(result)
        return result

    def compare_train_test(
        self,
        train_metrics: Dict[str, Any],
        test_metrics: Dict[str, Any],
    ) -> pd.DataFrame:
        """比较训练集和测试集指标差异。"""
        rows = []
        metric_names = sorted(set(train_metrics.keys()) & set(test_metrics.keys()))
        for name in metric_names:
            train_value = train_metrics.get(name)
            test_value = test_metrics.get(name)
            if not self._is_number(train_value) or not self._is_number(test_value):
                continue
            gap = float(train_value) - float(test_value)
            rows.append({
                "指标": name,
                "训练集": float(train_value),
                "测试集": float(test_value),
                "差异": gap,
                "风险判断": self._gap_level(name, gap),
            })
        return pd.DataFrame(rows)

    def threshold_analysis(
        self,
        y_true: Sequence[Any],
        positive_proba: Sequence[float],
        thresholds: Optional[Sequence[float]] = None,
    ) -> pd.DataFrame:
        """二分类阈值分析。"""
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 19)
        y_true_arr = np.asarray(y_true)
        proba_arr = np.asarray(positive_proba, dtype=float)
        positive_label = self._positive_label(y_true_arr)
        y_binary = (y_true_arr == positive_label).astype(int)

        rows = []
        for threshold in thresholds:
            pred = (proba_arr >= threshold).astype(int)
            rows.append({
                "阈值": float(threshold),
                "Accuracy": accuracy_score(y_binary, pred),
                "Precision": precision_score(y_binary, pred, zero_division=0),
                "Recall": recall_score(y_binary, pred, zero_division=0),
                "F1": f1_score(y_binary, pred, zero_division=0),
                "预测正例占比": float(pred.mean()),
            })
        return pd.DataFrame(rows)

    def segment_metrics(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        segment: Sequence[Any],
    ) -> pd.DataFrame:
        """按业务分组计算指标。"""
        df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "segment": segment,
        })
        rows = []
        for value, group in df.groupby("segment"):
            if len(group) == 0:
                continue
            row = {
                "分组": value,
                "样本数": len(group),
            }
            if self.problem_type == "classification":
                row.update({
                    "Accuracy": accuracy_score(group["y_true"], group["y_pred"]),
                    "F1": f1_score(group["y_true"], group["y_pred"], average="weighted", zero_division=0),
                })
            else:
                row.update({
                    "R2": r2_score(group["y_true"], group["y_pred"]) if len(group) > 1 else np.nan,
                    "RMSE": np.sqrt(mean_squared_error(group["y_true"], group["y_pred"])),
                    "MAE": mean_absolute_error(group["y_true"], group["y_pred"]),
                })
            rows.append(row)
        return pd.DataFrame(rows)

    def render_markdown(self, result: DiagnosticResult) -> str:
        """将诊断结果渲染为 Markdown。"""
        lines = ["# 模型诊断报告", ""]
        lines.append("## 核心指标")
        for key, value in result.metrics.items():
            if self._is_simple_value(value):
                lines.append(f"- {key}: {self._format_value(value)}")

        if result.warnings:
            lines.extend(["", "## 风险提示"])
            for warning in result.warnings:
                lines.append(f"- {warning}")

        if result.recommendations:
            lines.extend(["", "## 优化建议"])
            for recommendation in result.recommendations:
                lines.append(f"- {recommendation}")

        for name, table in result.tables.items():
            lines.extend(["", f"## {name}", self._table_preview(table)])
        return "\n".join(lines)

    def _evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[Sequence[Any]],
        result: DiagnosticResult,
    ) -> None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        # 二分类时以排序后的第二个标签为正类，避免字符串标签下默认 pos_label=1 报错
        if len(labels) == 2:
            score_kwargs = {"average": "binary", "pos_label": labels[1]}
        else:
            score_kwargs = {"average": "weighted"}
        result.metrics.update({
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0, **score_kwargs),
            "recall": recall_score(y_true, y_pred, zero_division=0, **score_kwargs),
            "f1": f1_score(y_true, y_pred, zero_division=0, **score_kwargs),
            "class_count": int(len(labels)),
        })

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        result.add_table("混淆矩阵", self._confusion_matrix_df(cm, labels))
        result.add_table("分类别指标", self._per_class_metrics(y_true, y_pred, labels))

        if y_proba is not None:
            proba_arr = np.asarray(y_proba)
            if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2 and len(labels) == 2:
                positive_scores = proba_arr[:, 1]
                positive_label = labels[1]
                y_binary = (y_true == positive_label).astype(int)
                try:
                    result.metrics["auc"] = roc_auc_score(y_binary, positive_scores)
                except ValueError:
                    result.warnings.append("AUC 计算失败，可能是测试集中只有一个类别。")
                result.add_table("阈值分析", self.threshold_analysis(y_true, positive_scores))
                result.add_table("概率校准表", self.calibration_table(y_true, positive_scores))
                result.add_table("提升表", self.lift_table(y_true, positive_scores))
            elif proba_arr.ndim == 1 and len(labels) == 2:
                result.add_table("阈值分析", self.threshold_analysis(y_true, proba_arr))

        class_counts = pd.Series(y_true).value_counts(normalize=True)
        if not class_counts.empty and class_counts.iloc[0] > 0.9:
            result.warnings.append("目标类别高度不均衡，单看 Accuracy 可能偏乐观。")
        if result.metrics["f1"] < 0.6:
            result.warnings.append("F1 分数偏低，模型可能无法稳定识别关键类别。")

    def _evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        result: DiagnosticResult,
    ) -> None:
        residuals = y_true.astype(float) - y_pred.astype(float)
        abs_error = np.abs(residuals)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        result.metrics.update({
            "r2": r2_score(y_true, y_pred),
            "mse": float(mse),
            "rmse": rmse,
            "mae": mae,
            "median_absolute_error": float(np.median(abs_error)),
            "max_absolute_error": float(np.max(abs_error)) if len(abs_error) else 0.0,
            "mean_residual": float(np.mean(residuals)) if len(residuals) else 0.0,
            "residual_std": float(np.std(residuals)) if len(residuals) else 0.0,
        })
        result.add_table("残差统计", self.residual_summary(y_true, y_pred))
        result.add_table("误差分箱", self.error_band_table(y_true, y_pred))

        if result.metrics["r2"] < 0:
            result.warnings.append("R² 小于 0，模型效果可能差于直接使用均值预测。")
        if result.metrics["mean_residual"] != 0 and abs(result.metrics["mean_residual"]) > max(mae * 0.2, 1e-9):
            result.warnings.append("平均残差偏离 0，模型可能存在系统性高估或低估。")

    def _per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[Any]) -> pd.DataFrame:
        rows = []
        for label in labels:
            true_binary = (y_true == label).astype(int)
            pred_binary = (y_pred == label).astype(int)
            support = int(true_binary.sum())
            rows.append({
                "类别": label,
                "样本数": support,
                "Precision": precision_score(true_binary, pred_binary, zero_division=0),
                "Recall": recall_score(true_binary, pred_binary, zero_division=0),
                "F1": f1_score(true_binary, pred_binary, zero_division=0),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _confusion_matrix_df(matrix: np.ndarray, labels: Sequence[Any]) -> pd.DataFrame:
        df = pd.DataFrame(matrix, index=[f"真实_{label}" for label in labels])
        df.columns = [f"预测_{label}" for label in labels]
        return df.reset_index().rename(columns={"index": "真实/预测"})

    def calibration_table(
        self,
        y_true: Sequence[Any],
        positive_proba: Sequence[float],
        bins: int = 10,
    ) -> pd.DataFrame:
        """生成概率校准表。"""
        y_true_arr = np.asarray(y_true)
        proba_arr = np.asarray(positive_proba, dtype=float)
        positive_label = self._positive_label(y_true_arr)
        y_binary = (y_true_arr == positive_label).astype(int)
        df = pd.DataFrame({"y": y_binary, "proba": proba_arr})
        df["bin"] = pd.cut(df["proba"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
        grouped = df.groupby("bin", observed=False).agg(
            样本数=("y", "size"),
            平均预测概率=("proba", "mean"),
            实际正例率=("y", "mean"),
        ).reset_index()
        grouped["校准偏差"] = grouped["平均预测概率"] - grouped["实际正例率"]
        return grouped

    def lift_table(
        self,
        y_true: Sequence[Any],
        positive_proba: Sequence[float],
        groups: int = 10,
    ) -> pd.DataFrame:
        """生成提升表。"""
        y_true_arr = np.asarray(y_true)
        proba_arr = np.asarray(positive_proba, dtype=float)
        positive_label = self._positive_label(y_true_arr)
        y_binary = (y_true_arr == positive_label).astype(int)
        df = pd.DataFrame({"y": y_binary, "proba": proba_arr})
        df = df.sort_values("proba", ascending=False).reset_index(drop=True)
        df["分组"] = pd.qcut(df.index + 1, q=groups, labels=False, duplicates="drop") + 1
        base_rate = max(float(df["y"].mean()), 1e-12)
        table = df.groupby("分组").agg(
            样本数=("y", "size"),
            平均预测概率=("proba", "mean"),
            正例数=("y", "sum"),
            正例率=("y", "mean"),
        ).reset_index()
        table["提升倍数"] = table["正例率"] / base_rate
        table["累计正例数"] = table["正例数"].cumsum()
        table["累计召回率"] = table["累计正例数"] / max(float(df["y"].sum()), 1e-12)
        return table

    def residual_summary(self, y_true: Sequence[Any], y_pred: Sequence[Any]) -> pd.DataFrame:
        """生成回归残差统计。"""
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        residuals = y_true_arr - y_pred_arr
        abs_error = np.abs(residuals)
        rows = [
            {"统计项": "残差均值", "值": float(np.mean(residuals))},
            {"统计项": "残差标准差", "值": float(np.std(residuals))},
            {"统计项": "残差中位数", "值": float(np.median(residuals))},
            {"统计项": "绝对误差均值", "值": float(np.mean(abs_error))},
            {"统计项": "绝对误差P90", "值": float(np.quantile(abs_error, 0.90))},
            {"统计项": "绝对误差P95", "值": float(np.quantile(abs_error, 0.95))},
            {"统计项": "最大绝对误差", "值": float(np.max(abs_error))},
        ]
        return pd.DataFrame(rows)

    def error_band_table(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        bands: Optional[Sequence[float]] = None,
    ) -> pd.DataFrame:
        """按绝对误差分箱统计样本占比。"""
        if bands is None:
            bands = [0, 1, 5, 10, 20, 50, np.inf]
        residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        abs_error = np.abs(residuals)
        labels = []
        for left, right in zip(bands[:-1], bands[1:]):
            if np.isinf(right):
                labels.append(f">={left}")
            else:
                labels.append(f"{left}-{right}")
        bins = pd.cut(abs_error, bins=bands, labels=labels, include_lowest=True)
        counts = pd.Series(bins).value_counts(sort=False).reset_index()
        counts.columns = ["误差区间", "样本数"]
        counts["占比"] = counts["样本数"] / max(len(abs_error), 1)
        return counts

    def _build_recommendations(self, result: DiagnosticResult) -> List[str]:
        recommendations: List[str] = []
        if self.problem_type == "classification":
            f1 = result.metrics.get("f1")
            if self._is_number(f1) and f1 < 0.7:
                recommendations.append("分类 F1 偏低，建议检查类别不均衡、特征表达和阈值设置。")
            if "阈值分析" in result.tables:
                recommendations.append("可根据业务成本在阈值分析表中选择更合适的 Precision/Recall 平衡点。")
            if "概率校准表" in result.tables:
                recommendations.append("若预测概率用于业务排序或定价，建议关注校准偏差并考虑概率校准。")
        else:
            r2 = result.metrics.get("r2")
            if self._is_number(r2) and r2 < 0.5:
                recommendations.append("回归解释度偏低，建议增加特征工程、处理异常值或尝试非线性模型。")
            mean_residual = result.metrics.get("mean_residual")
            mae = result.metrics.get("mae")
            if self._is_number(mean_residual) and self._is_number(mae) and abs(mean_residual) > mae * 0.2:
                recommendations.append("残差存在系统偏移，建议检查样本分布或增加校准后处理。")
        if not recommendations:
            recommendations.append("核心诊断未发现明显异常，可结合业务指标继续验证。")
        return recommendations

    @staticmethod
    def _positive_label(y_true: np.ndarray) -> Any:
        labels = np.unique(y_true)
        if len(labels) == 0:
            return 1
        if len(labels) == 1:
            return labels[0]
        return labels[-1]

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value)

    @staticmethod
    def _is_simple_value(value: Any) -> bool:
        return isinstance(value, (str, int, float, np.integer, np.floating)) or value is None

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    @staticmethod
    def _gap_level(metric_name: str, gap: float) -> str:
        lower_name = metric_name.lower()
        if any(token in lower_name for token in ["rmse", "mse", "mae"]):
            if gap < -0.2:
                return "测试误差明显高于训练误差"
            return "正常"
        if gap > 0.15:
            return "可能过拟合"
        if gap < -0.05:
            return "测试集表现优于训练集，需检查划分"
        return "正常"

    @staticmethod
    def _table_preview(table: pd.DataFrame, max_rows: int = 20) -> str:
        if table is None or table.empty:
            return "_无数据_"
        view = table.head(max_rows)
        try:
            return view.to_markdown(index=False)
        except Exception:
            return view.to_string(index=False)
