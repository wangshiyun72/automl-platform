"""
数据画像与质量诊断模块

该模块负责在建模前对数据集进行系统化扫描，输出列级画像、缺失值
概览、重复样本、相关性、目标变量类型、潜在数据泄露与自动处理建议。
它不修改原始数据，只提供供界面、报告和建模流程使用的诊断结果。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    """列级画像信息。"""

    name: str
    dtype: str
    inferred_type: str
    null_count: int
    null_ratio: float
    unique_count: int
    unique_ratio: float
    memory_bytes: int
    examples: List[str] = field(default_factory=list)
    numeric_stats: Dict[str, Any] = field(default_factory=dict)
    categorical_stats: Dict[str, Any] = field(default_factory=dict)
    datetime_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        item = {
            "列名": self.name,
            "原始类型": self.dtype,
            "推断类型": self.inferred_type,
            "缺失数": self.null_count,
            "缺失率": self.null_ratio,
            "唯一值数": self.unique_count,
            "唯一值比例": self.unique_ratio,
            "内存占用": self.memory_bytes,
            "样例": ", ".join(self.examples),
        }
        item.update({f"数值_{k}": v for k, v in self.numeric_stats.items()})
        item.update({f"分类_{k}": v for k, v in self.categorical_stats.items()})
        item.update({f"日期_{k}": v for k, v in self.datetime_stats.items()})
        return item


@dataclass
class DataQualityIssue:
    """数据质量问题。"""

    level: str
    column: Optional[str]
    issue: str
    detail: str
    suggestion: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "级别": self.level,
            "列名": self.column or "",
            "问题": self.issue,
            "详情": self.detail,
            "建议": self.suggestion,
        }


class DataProfiler:
    """
    数据画像器。

    使用方式:
        profiler = DataProfiler()
        profiler.profile(df, target_column="label")
        overview = profiler.get_overview()
        issues = profiler.get_quality_issues()
    """

    def __init__(
        self,
        missing_warning_threshold: float = 0.30,
        missing_error_threshold: float = 0.80,
        high_cardinality_threshold: float = 0.80,
        correlation_threshold: float = 0.95,
        imbalance_threshold: float = 0.90,
        datetime_parse_threshold: float = 0.80,
        max_examples: int = 5,
    ):
        self.missing_warning_threshold = missing_warning_threshold
        self.missing_error_threshold = missing_error_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.correlation_threshold = correlation_threshold
        self.imbalance_threshold = imbalance_threshold
        self.datetime_parse_threshold = datetime_parse_threshold
        self.max_examples = max_examples

        self.fitted = False
        self.shape: Tuple[int, int] = (0, 0)
        self.target_column: Optional[str] = None
        self.target_type: Optional[str] = None
        self.column_profiles: Dict[str, ColumnProfile] = {}
        self.quality_issues: List[DataQualityIssue] = []
        self.correlation_pairs: List[Dict[str, Any]] = []
        self.duplicate_summary: Dict[str, Any] = {}
        self.leakage_candidates: List[Dict[str, Any]] = []
        self.recommendations: List[str] = []

    def profile(self, df: pd.DataFrame, target_column: Optional[str] = None) -> "DataProfiler":
        """执行完整数据画像。"""
        self._validate_dataframe(df)
        self._reset()
        self.shape = df.shape
        self.target_column = target_column if target_column in df.columns else None

        for column in df.columns:
            self.column_profiles[column] = self._profile_column(df[column])

        if self.target_column:
            self.target_type = self._infer_target_type(df[self.target_column])

        self.duplicate_summary = self._profile_duplicates(df)
        self.correlation_pairs = self._profile_correlations(df)
        self.leakage_candidates = self._detect_leakage_candidates(df)
        self.quality_issues = self._build_quality_issues(df)
        self.recommendations = self._build_recommendations()
        self.fitted = True
        return self

    def get_overview(self) -> Dict[str, Any]:
        """返回数据集整体概览。"""
        self._ensure_fitted()
        row_count, column_count = self.shape
        total_missing = sum(item.null_count for item in self.column_profiles.values())
        total_cells = max(row_count * column_count, 1)
        return {
            "row_count": row_count,
            "column_count": column_count,
            "total_missing": total_missing,
            "missing_ratio": total_missing / total_cells,
            "target_column": self.target_column,
            "target_type": self.target_type,
            "duplicate_rows": self.duplicate_summary.get("duplicate_rows", 0),
            "duplicate_ratio": self.duplicate_summary.get("duplicate_ratio", 0.0),
            "issue_count": len(self.quality_issues),
        }

    def get_column_profile_df(self) -> pd.DataFrame:
        """返回列级画像表。"""
        self._ensure_fitted()
        return pd.DataFrame([profile.to_dict() for profile in self.column_profiles.values()])

    def get_missing_report(self) -> pd.DataFrame:
        """返回缺失值报告。"""
        self._ensure_fitted()
        rows = []
        for profile in self.column_profiles.values():
            if profile.null_count > 0:
                rows.append({
                    "列名": profile.name,
                    "缺失数": profile.null_count,
                    "缺失率": profile.null_ratio,
                    "推断类型": profile.inferred_type,
                })
        return pd.DataFrame(rows).sort_values("缺失率", ascending=False) if rows else pd.DataFrame()

    def get_numeric_report(self) -> pd.DataFrame:
        """返回数值列统计报告。"""
        self._ensure_fitted()
        rows = []
        for profile in self.column_profiles.values():
            if profile.numeric_stats:
                item = {"列名": profile.name, "缺失率": profile.null_ratio}
                item.update(profile.numeric_stats)
                rows.append(item)
        return pd.DataFrame(rows)

    def get_categorical_report(self) -> pd.DataFrame:
        """返回分类列统计报告。"""
        self._ensure_fitted()
        rows = []
        for profile in self.column_profiles.values():
            if profile.categorical_stats:
                item = {"列名": profile.name, "缺失率": profile.null_ratio}
                item.update(profile.categorical_stats)
                rows.append(item)
        return pd.DataFrame(rows)

    def get_correlation_report(self) -> pd.DataFrame:
        """返回高相关特征对。"""
        self._ensure_fitted()
        return pd.DataFrame(self.correlation_pairs)

    def get_quality_issues(self) -> pd.DataFrame:
        """返回数据质量问题表。"""
        self._ensure_fitted()
        return pd.DataFrame([issue.to_dict() for issue in self.quality_issues])

    def get_leakage_candidates(self) -> pd.DataFrame:
        """返回潜在泄露特征表。"""
        self._ensure_fitted()
        return pd.DataFrame(self.leakage_candidates)

    def get_recommendations(self) -> List[str]:
        """返回自动处理建议。"""
        self._ensure_fitted()
        return list(self.recommendations)

    def render_markdown(self) -> str:
        """生成 Markdown 格式画像摘要。"""
        self._ensure_fitted()
        overview = self.get_overview()
        lines = [
            "# 数据画像报告",
            "",
            "## 数据概览",
            f"- 样本数: {overview['row_count']}",
            f"- 字段数: {overview['column_count']}",
            f"- 总缺失率: {overview['missing_ratio']:.2%}",
            f"- 重复样本: {overview['duplicate_rows']} ({overview['duplicate_ratio']:.2%})",
        ]
        if overview["target_column"]:
            lines.append(f"- 目标列: {overview['target_column']} ({overview['target_type']})")

        lines.extend(["", "## 字段类型分布"])
        type_counts = self.get_column_profile_df()["推断类型"].value_counts()
        for col_type, count in type_counts.items():
            lines.append(f"- {col_type}: {count}")

        if self.quality_issues:
            lines.extend(["", "## 主要质量问题"])
            for issue in self.quality_issues[:20]:
                prefix = f"[{issue.level}]"
                column = f"{issue.column}: " if issue.column else ""
                lines.append(f"- {prefix} {column}{issue.issue}，{issue.suggestion}")

        if self.recommendations:
            lines.extend(["", "## 处理建议"])
            for suggestion in self.recommendations:
                lines.append(f"- {suggestion}")

        return "\n".join(lines)

    def _reset(self) -> None:
        self.fitted = False
        self.shape = (0, 0)
        self.target_column = None
        self.target_type = None
        self.column_profiles = {}
        self.quality_issues = []
        self.correlation_pairs = []
        self.duplicate_summary = {}
        self.leakage_candidates = []
        self.recommendations = []

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 必须是 pandas.DataFrame")
        if df.empty:
            raise ValueError("数据集为空，无法画像")

    def _ensure_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("DataProfiler 尚未执行 profile")

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        row_count = max(len(series), 1)
        non_null = series.dropna()
        examples = [str(value)[:80] for value in non_null.head(self.max_examples).tolist()]
        inferred_type = self._infer_column_type(series)

        profile = ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            inferred_type=inferred_type,
            null_count=int(series.isna().sum()),
            null_ratio=float(series.isna().mean()),
            unique_count=int(series.nunique(dropna=True)),
            unique_ratio=float(series.nunique(dropna=True) / row_count),
            memory_bytes=int(series.memory_usage(deep=True)),
            examples=examples,
        )

        if inferred_type == "numeric":
            profile.numeric_stats = self._numeric_stats(series)
        elif inferred_type == "datetime":
            profile.datetime_stats = self._datetime_stats(series)
        else:
            profile.categorical_stats = self._categorical_stats(series)

        return profile

    def _infer_column_type(self, series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if self._looks_like_datetime(series):
            return "datetime"
        if self._looks_like_text(series):
            return "text"
        return "categorical"

    def _infer_target_type(self, series: pd.Series) -> str:
        unique_count = series.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(series) and unique_count > 20:
            return "regression"
        return "classification"

    def _numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        values = pd.to_numeric(series, errors="coerce")
        clean = values.dropna()
        if clean.empty:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "q1": None,
                "median": None,
                "q3": None,
                "max": None,
                "skew": None,
                "zero_ratio": None,
                "negative_ratio": None,
            }

        return {
            "mean": float(clean.mean()),
            "std": float(clean.std()) if len(clean) > 1 else 0.0,
            "min": float(clean.min()),
            "q1": float(clean.quantile(0.25)),
            "median": float(clean.median()),
            "q3": float(clean.quantile(0.75)),
            "max": float(clean.max()),
            "skew": float(clean.skew()) if len(clean) > 2 else 0.0,
            "kurtosis": float(clean.kurtosis()) if len(clean) > 3 else 0.0,
            "zero_ratio": float((clean == 0).mean()),
            "negative_ratio": float((clean < 0).mean()),
            "outlier_ratio": self._outlier_ratio(clean),
        }

    def _categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        normalized = series.astype("object").where(series.notna(), "__missing__").astype(str)
        counts = normalized.value_counts(dropna=False)
        total = max(int(counts.sum()), 1)
        top_value = str(counts.index[0]) if not counts.empty else None
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        return {
            "top_value": top_value,
            "top_ratio": top_count / total,
            "entropy": self._entropy(counts),
            "rare_value_count": int((counts / total < 0.01).sum()),
            "high_cardinality": bool(series.nunique(dropna=True) / total >= self.high_cardinality_threshold),
        }

    def _datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        valid = parsed.dropna()
        if valid.empty:
            return {
                "parse_ratio": 0.0,
                "min": None,
                "max": None,
                "span_days": None,
            }
        span = valid.max() - valid.min()
        return {
            "parse_ratio": float(parsed.notna().mean()),
            "min": str(valid.min()),
            "max": str(valid.max()),
            "span_days": float(span.total_seconds() / 86400.0),
            "year_count": int(valid.dt.year.nunique()),
        }

    def _profile_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        duplicate_mask = df.duplicated()
        duplicate_rows = int(duplicate_mask.sum())
        return {
            "duplicate_rows": duplicate_rows,
            "duplicate_ratio": duplicate_rows / max(len(df), 1),
            "unique_rows": int(len(df) - duplicate_rows),
        }

    def _profile_correlations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return []
        corr = numeric_df.corr().abs()
        pairs: List[Dict[str, Any]] = []
        for i, left in enumerate(corr.columns):
            for right in corr.columns[i + 1:]:
                value = corr.loc[left, right]
                if pd.notna(value) and value >= self.correlation_threshold:
                    pairs.append({
                        "特征A": left,
                        "特征B": right,
                        "相关系数": float(value),
                        "建议": "考虑删除其中一个高相关特征，降低冗余和过拟合风险",
                    })
        return sorted(pairs, key=lambda item: item["相关系数"], reverse=True)

    def _detect_leakage_candidates(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if not self.target_column:
            return []

        target = df[self.target_column]
        candidates: List[Dict[str, Any]] = []
        for column in df.columns:
            if column == self.target_column:
                continue
            series = df[column]
            if series.equals(target):
                candidates.append({
                    "列名": column,
                    "风险": "完全等于目标列",
                    "证据": "列值与目标列逐行一致",
                    "建议": "从训练特征中移除该列",
                })
                continue

            if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target):
                corr = pd.to_numeric(series, errors="coerce").corr(pd.to_numeric(target, errors="coerce"))
                if pd.notna(corr) and abs(corr) >= 0.995:
                    candidates.append({
                        "列名": column,
                        "风险": "与目标列近乎完全相关",
                        "证据": f"相关系数 {corr:.4f}",
                        "建议": "检查该列是否为目标泄露或后验字段",
                    })

            if self._name_suggests_leakage(column):
                candidates.append({
                    "列名": column,
                    "风险": "字段名疑似后验信息",
                    "证据": "字段名包含 label、target、result、score 等关键词",
                    "建议": "确认该字段在预测时是否可提前获得",
                })

        return candidates

    def _build_quality_issues(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        issues: List[DataQualityIssue] = []
        for profile in self.column_profiles.values():
            if profile.null_ratio >= self.missing_error_threshold:
                issues.append(DataQualityIssue(
                    level="高",
                    column=profile.name,
                    issue="缺失率过高",
                    detail=f"缺失率为 {profile.null_ratio:.2%}",
                    suggestion="建议删除该列或确认数据采集流程",
                ))
            elif profile.null_ratio >= self.missing_warning_threshold:
                issues.append(DataQualityIssue(
                    level="中",
                    column=profile.name,
                    issue="缺失率偏高",
                    detail=f"缺失率为 {profile.null_ratio:.2%}",
                    suggestion="建议使用中位数、众数或业务规则填充",
                ))

            if profile.unique_count <= 1:
                issues.append(DataQualityIssue(
                    level="中",
                    column=profile.name,
                    issue="常量列",
                    detail="该列只有一个有效取值",
                    suggestion="建议从训练特征中删除",
                ))

            if profile.categorical_stats.get("high_cardinality"):
                issues.append(DataQualityIssue(
                    level="中",
                    column=profile.name,
                    issue="高基数分类列",
                    detail=f"唯一值比例为 {profile.unique_ratio:.2%}",
                    suggestion="建议使用频率编码、目标编码或聚合稀有类别",
                ))

            if profile.numeric_stats.get("outlier_ratio", 0) >= 0.05:
                issues.append(DataQualityIssue(
                    level="低",
                    column=profile.name,
                    issue="异常值比例较高",
                    detail=f"异常值比例约 {profile.numeric_stats['outlier_ratio']:.2%}",
                    suggestion="建议使用稳健缩放、分箱或异常值截尾",
                ))

        if self.duplicate_summary.get("duplicate_ratio", 0) > 0:
            issues.append(DataQualityIssue(
                level="低",
                column=None,
                issue="存在重复样本",
                detail=f"重复率 {self.duplicate_summary['duplicate_ratio']:.2%}",
                suggestion="训练前可按业务主键或全字段去重",
            ))

        if self.target_column:
            issues.extend(self._target_quality_issues(df[self.target_column]))

        return issues

    def _target_quality_issues(self, target: pd.Series) -> List[DataQualityIssue]:
        issues: List[DataQualityIssue] = []
        if target.isna().any():
            issues.append(DataQualityIssue(
                level="高",
                column=self.target_column,
                issue="目标列存在缺失",
                detail=f"缺失数 {int(target.isna().sum())}",
                suggestion="建议删除目标缺失样本或补充标签",
            ))

        if self.target_type == "classification":
            counts = target.value_counts(normalize=True, dropna=True)
            if not counts.empty and counts.iloc[0] >= self.imbalance_threshold:
                issues.append(DataQualityIssue(
                    level="中",
                    column=self.target_column,
                    issue="类别分布不均衡",
                    detail=f"最大类别占比 {counts.iloc[0]:.2%}",
                    suggestion="建议使用分层抽样、类别权重或重采样策略",
                ))
        return issues

    def _build_recommendations(self) -> List[str]:
        recommendations: List[str] = []
        if any(issue.issue == "缺失率过高" for issue in self.quality_issues):
            recommendations.append("对缺失率过高的列进行删除或单独建模评估，避免噪声进入训练。")
        if any(issue.issue == "高基数分类列" for issue in self.quality_issues):
            recommendations.append("高基数分类列优先使用频率编码或稀有类别合并，避免独热编码膨胀。")
        if self.correlation_pairs:
            recommendations.append("存在高相关特征对，可在特征工程阶段保留信息量更高的一列。")
        if self.leakage_candidates:
            recommendations.append("潜在泄露字段需要人工确认，泄露字段会导致离线指标虚高。")
        if self.target_type == "classification":
            recommendations.append("分类任务建议使用准确率、F1、AUC、混淆矩阵综合评估。")
        elif self.target_type == "regression":
            recommendations.append("回归任务建议同时关注 R²、RMSE、MAE 与残差分布。")
        if not recommendations:
            recommendations.append("数据质量未发现明显高风险项，可进入特征工程和模型训练阶段。")
        return recommendations

    def _looks_like_datetime(self, series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_numeric_dtype(series):
            return pd.api.types.is_datetime64_any_dtype(series)
        non_null = series.dropna()
        if non_null.empty:
            return False
        sample = non_null.astype(str).head(min(200, len(non_null)))
        parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        return bool(parsed.notna().mean() >= self.datetime_parse_threshold)

    @staticmethod
    def _looks_like_text(series: pd.Series) -> bool:
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            return False
        non_null = series.dropna().astype(str)
        if non_null.empty:
            return False
        avg_len = non_null.str.len().mean()
        with_space = non_null.str.contains(r"\s+", regex=True).mean()
        return bool(avg_len >= 80 and with_space > 0.2)

    @staticmethod
    def _outlier_ratio(values: pd.Series) -> float:
        if values.empty:
            return 0.0
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            return 0.0
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return float(((values < lower) | (values > upper)).mean())

    @staticmethod
    def _entropy(counts: pd.Series) -> float:
        total = counts.sum()
        if total <= 0:
            return 0.0
        probabilities = counts / total
        probabilities = probabilities[probabilities > 0]
        return float(-(probabilities * np.log2(probabilities)).sum())

    @staticmethod
    def _name_suggests_leakage(column: str) -> bool:
        keywords = ["label", "target", "result", "score", "是否通过", "最终", "预测结果"]
        lower_name = column.lower()
        return any(keyword.lower() in lower_name for keyword in keywords)
