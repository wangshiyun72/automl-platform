"""
自动特征工程模块

该模块提供面向 AutoML 工作流的特征识别、清洗、构造与筛选能力。
设计目标是把常见的数据准备逻辑沉淀为可复用组件，减少主界面里的
过程性代码，并为后续报告、批量预测、模型复现实验提供稳定的特征
处理配置。
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureSummary:
    """记录单列在拟合阶段的画像信息。"""

    name: str
    dtype: str
    role: str
    null_count: int
    null_ratio: float
    unique_count: int
    unique_ratio: float
    sample_values: List[str] = field(default_factory=list)
    drop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "role": self.role,
            "null_count": self.null_count,
            "null_ratio": self.null_ratio,
            "unique_count": self.unique_count,
            "unique_ratio": self.unique_ratio,
            "sample_values": self.sample_values,
            "drop_reason": self.drop_reason,
        }


@dataclass
class NumericFeatureMeta:
    """数值列的拟合元数据。"""

    median: float
    mean: float
    std: float
    min_value: float
    max_value: float
    q1: float
    q3: float
    skew: float
    bin_edges: List[float] = field(default_factory=list)
    use_log: bool = False
    use_outlier_flag: bool = False


@dataclass
class CategoricalFeatureMeta:
    """分类列的拟合元数据。"""

    fill_value: str
    top_categories: List[str]
    frequency_map: Dict[str, float]
    count_map: Dict[str, int]
    rare_categories: List[str]
    one_hot_categories: List[str]


@dataclass
class DatetimeFeatureMeta:
    """日期时间列的拟合元数据。"""

    min_timestamp: Optional[pd.Timestamp]
    max_timestamp: Optional[pd.Timestamp]
    parse_ratio: float


class FeatureEngineer:
    """
    自动特征工程器。

    典型用法:
        engineer = FeatureEngineer(create_interactions=True)
        new_df = engineer.fit_transform(df, target_column="label")
        feature_columns = engineer.get_output_columns()

    输出全部为数值列，便于直接交给后续模型训练器或再经过 DataProcessor。
    """

    def __init__(
        self,
        missing_threshold: float = 0.95,
        unique_threshold: float = 0.98,
        variance_threshold: float = 0.0,
        corr_threshold: float = 0.98,
        max_one_hot_levels: int = 12,
        max_category_levels: int = 50,
        min_category_frequency: float = 0.01,
        datetime_parse_threshold: float = 0.80,
        bin_count: int = 10,
        create_missing_indicators: bool = True,
        create_datetime_features: bool = True,
        create_numeric_features: bool = True,
        create_interactions: bool = False,
        max_interactions: int = 20,
        max_text_length: int = 120,
        random_state: int = 42,
    ):
        self.missing_threshold = missing_threshold
        self.unique_threshold = unique_threshold
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.max_one_hot_levels = max_one_hot_levels
        self.max_category_levels = max_category_levels
        self.min_category_frequency = min_category_frequency
        self.datetime_parse_threshold = datetime_parse_threshold
        self.bin_count = bin_count
        self.create_missing_indicators = create_missing_indicators
        self.create_datetime_features = create_datetime_features
        self.create_numeric_features = create_numeric_features
        self.create_interactions = create_interactions
        self.max_interactions = max_interactions
        self.max_text_length = max_text_length
        self.random_state = random_state

        self.fitted = False
        self.target_column: Optional[str] = None
        self.input_columns: List[str] = []
        self.output_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.datetime_columns: List[str] = []
        self.boolean_columns: List[str] = []
        self.dropped_columns: Dict[str, str] = {}
        self.column_summaries: Dict[str, FeatureSummary] = {}
        self.numeric_meta: Dict[str, NumericFeatureMeta] = {}
        self.categorical_meta: Dict[str, CategoricalFeatureMeta] = {}
        self.datetime_meta: Dict[str, DatetimeFeatureMeta] = {}
        self.interaction_pairs: List[Tuple[str, str]] = []
        self.correlation_drop_columns: List[str] = []
        self.audit_log: List[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> "FeatureEngineer":
        """
        拟合特征工程规则。

        参数:
            df: 原始数据集
            target_column: 目标列，拟合特征时会自动排除
            feature_columns: 指定参与处理的特征列；为空时使用除目标列外的所有列
        """
        self._validate_dataframe(df)
        self._reset_state()
        self.target_column = target_column

        candidate_columns = self._resolve_feature_columns(df, target_column, feature_columns)
        self.input_columns = list(candidate_columns)
        self.column_summaries = self._profile_columns(df, candidate_columns)
        kept_columns = self._select_usable_columns(df, candidate_columns)

        self._infer_column_groups(df, kept_columns)
        self._fit_numeric_columns(df)
        self._fit_categorical_columns(df)
        self._fit_datetime_columns(df)
        self._fit_interactions()

        generated = self._build_feature_frame(df, fitting=True)
        self.correlation_drop_columns = self._find_correlated_columns(generated)
        for name in self.correlation_drop_columns:
            self.dropped_columns[name] = f"与其他生成特征相关系数超过 {self.corr_threshold}"

        self.output_columns = [
            name for name in generated.columns
            if name not in set(self.correlation_drop_columns)
        ]
        self.fitted = True
        self.audit_log.append(f"拟合完成，输入列 {len(self.input_columns)} 个，输出特征 {len(self.output_columns)} 个")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用拟合好的规则转换新数据。
        """
        if not self.fitted:
            raise RuntimeError("FeatureEngineer 尚未拟合，请先调用 fit 或 fit_transform")
        self._validate_dataframe(df)
        missing_columns = [col for col in self.input_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"新数据缺少训练时使用的列: {missing_columns}")

        transformed = self._build_feature_frame(df, fitting=False)
        for col in self.output_columns:
            if col not in transformed.columns:
                transformed[col] = 0
        return transformed[self.output_columns].copy()

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """拟合并转换数据。"""
        self.fit(df, target_column=target_column, feature_columns=feature_columns)
        return self.transform(df)

    def get_output_columns(self) -> List[str]:
        """返回转换后的特征列名。"""
        return list(self.output_columns)

    def get_dropped_columns(self) -> Dict[str, str]:
        """返回被删除的原始列或生成列及原因。"""
        return dict(self.dropped_columns)

    def get_feature_report(self) -> pd.DataFrame:
        """返回拟合阶段的列画像与处理决策。"""
        rows = []
        for summary in self.column_summaries.values():
            item = summary.to_dict()
            item["is_input"] = summary.name in self.input_columns
            item["is_numeric"] = summary.name in self.numeric_columns
            item["is_categorical"] = summary.name in self.categorical_columns
            item["is_datetime"] = summary.name in self.datetime_columns
            item["is_boolean"] = summary.name in self.boolean_columns
            rows.append(item)
        return pd.DataFrame(rows)

    def export_config(self) -> Dict[str, Any]:
        """导出可序列化的特征工程配置。"""
        return {
            "target_column": self.target_column,
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "boolean_columns": self.boolean_columns,
            "dropped_columns": self.dropped_columns,
            "interaction_pairs": self.interaction_pairs,
            "correlation_drop_columns": self.correlation_drop_columns,
            "parameters": {
                "missing_threshold": self.missing_threshold,
                "unique_threshold": self.unique_threshold,
                "variance_threshold": self.variance_threshold,
                "corr_threshold": self.corr_threshold,
                "max_one_hot_levels": self.max_one_hot_levels,
                "max_category_levels": self.max_category_levels,
                "min_category_frequency": self.min_category_frequency,
                "datetime_parse_threshold": self.datetime_parse_threshold,
                "bin_count": self.bin_count,
                "create_missing_indicators": self.create_missing_indicators,
                "create_datetime_features": self.create_datetime_features,
                "create_numeric_features": self.create_numeric_features,
                "create_interactions": self.create_interactions,
                "max_interactions": self.max_interactions,
            },
            "audit_log": self.audit_log,
        }

    def _reset_state(self) -> None:
        self.fitted = False
        self.input_columns = []
        self.output_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.boolean_columns = []
        self.dropped_columns = {}
        self.column_summaries = {}
        self.numeric_meta = {}
        self.categorical_meta = {}
        self.datetime_meta = {}
        self.interaction_pairs = []
        self.correlation_drop_columns = []
        self.audit_log = []

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 必须是 pandas.DataFrame")
        if df.empty:
            raise ValueError("数据集为空，无法进行特征工程")

    @staticmethod
    def _resolve_feature_columns(
        df: pd.DataFrame,
        target_column: Optional[str],
        feature_columns: Optional[Sequence[str]],
    ) -> List[str]:
        if feature_columns is None:
            columns = list(df.columns)
            if target_column in columns:
                columns.remove(target_column)
            return columns

        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"指定的特征列不存在: {missing}")
        return list(feature_columns)

    def _profile_columns(self, df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, FeatureSummary]:
        profiles: Dict[str, FeatureSummary] = {}
        row_count = max(len(df), 1)
        for col in columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            unique_count = int(series.nunique(dropna=True))
            samples = [
                str(value)[:60]
                for value in series.dropna().head(5).tolist()
            ]
            role = self._guess_column_role(series)
            profiles[col] = FeatureSummary(
                name=col,
                dtype=str(series.dtype),
                role=role,
                null_count=null_count,
                null_ratio=null_count / row_count,
                unique_count=unique_count,
                unique_ratio=unique_count / row_count,
                sample_values=samples,
            )
        return profiles

    def _guess_column_role(self, series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if self._looks_like_datetime(series):
            return "datetime"
        return "categorical"

    def _select_usable_columns(self, df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
        kept: List[str] = []
        row_count = max(len(df), 1)
        for col in columns:
            series = df[col]
            summary = self.column_summaries[col]
            reason = None

            if summary.null_ratio >= self.missing_threshold:
                reason = f"缺失率 {summary.null_ratio:.2%} 超过阈值"
            elif summary.unique_count <= 1:
                reason = "常量列或仅有一个有效取值"
            elif summary.unique_ratio >= self.unique_threshold and not pd.api.types.is_numeric_dtype(series):
                reason = "近似唯一标识列，泛化价值较低"
            elif self._looks_like_long_text(series):
                reason = "长文本列暂不参与结构化建模"
            elif pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                variance = float(numeric.var(skipna=True)) if numeric.notna().any() else 0.0
                if variance <= self.variance_threshold:
                    reason = "数值方差过低"

            if reason:
                self.dropped_columns[col] = reason
                summary.drop_reason = reason
                self.audit_log.append(f"删除列 {col}: {reason}")
            else:
                kept.append(col)

        self.audit_log.append(f"可用原始列 {len(kept)} 个，删除 {len(columns) - len(kept)} 个")
        return kept

    def _infer_column_groups(self, df: pd.DataFrame, columns: Sequence[str]) -> None:
        for col in columns:
            series = df[col]
            if pd.api.types.is_bool_dtype(series):
                self.boolean_columns.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                self.numeric_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(series) or self._looks_like_datetime(series):
                self.datetime_columns.append(col)
            else:
                self.categorical_columns.append(col)

        self.audit_log.append(
            "列类型识别: "
            f"数值 {len(self.numeric_columns)}，分类 {len(self.categorical_columns)}，"
            f"日期 {len(self.datetime_columns)}，布尔 {len(self.boolean_columns)}"
        )

    def _fit_numeric_columns(self, df: pd.DataFrame) -> None:
        for col in self.numeric_columns:
            values = pd.to_numeric(df[col], errors="coerce")
            clean = values.dropna()
            if clean.empty:
                meta = NumericFeatureMeta(
                    median=0.0,
                    mean=0.0,
                    std=1.0,
                    min_value=0.0,
                    max_value=0.0,
                    q1=0.0,
                    q3=0.0,
                    skew=0.0,
                    bin_edges=[],
                )
                self.numeric_meta[col] = meta
                continue

            q1 = float(clean.quantile(0.25))
            q3 = float(clean.quantile(0.75))
            std = float(clean.std()) if clean.std() and not np.isnan(clean.std()) else 1.0
            skew = float(clean.skew()) if len(clean) > 2 else 0.0
            bin_edges = self._safe_quantile_edges(clean, self.bin_count)
            meta = NumericFeatureMeta(
                median=float(clean.median()),
                mean=float(clean.mean()),
                std=std,
                min_value=float(clean.min()),
                max_value=float(clean.max()),
                q1=q1,
                q3=q3,
                skew=skew,
                bin_edges=bin_edges,
                use_log=abs(skew) >= 1.0,
                use_outlier_flag=(q3 - q1) > 0,
            )
            self.numeric_meta[col] = meta

    def _fit_categorical_columns(self, df: pd.DataFrame) -> None:
        for col in self.categorical_columns:
            normalized = self._normalize_category_series(df[col])
            counts = normalized.value_counts(dropna=False)
            total = max(int(counts.sum()), 1)
            top_categories = counts.head(self.max_category_levels).index.astype(str).tolist()
            frequency_map = {
                str(key): float(value / total)
                for key, value in counts.items()
            }
            count_map = {
                str(key): int(value)
                for key, value in counts.items()
            }
            rare_categories = [
                str(key)
                for key, value in counts.items()
                if value / total < self.min_category_frequency
            ]
            one_hot_categories: List[str] = []
            if 1 < len(counts) <= self.max_one_hot_levels:
                one_hot_categories = counts.index.astype(str).tolist()
            elif len(counts) > self.max_one_hot_levels:
                one_hot_categories = counts.head(self.max_one_hot_levels).index.astype(str).tolist()

            self.categorical_meta[col] = CategoricalFeatureMeta(
                fill_value="__missing__",
                top_categories=top_categories,
                frequency_map=frequency_map,
                count_map=count_map,
                rare_categories=rare_categories,
                one_hot_categories=one_hot_categories,
            )

    def _fit_datetime_columns(self, df: pd.DataFrame) -> None:
        for col in self.datetime_columns:
            parsed = self._parse_datetime(df[col])
            valid = parsed.dropna()
            parse_ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
            min_timestamp = valid.min() if not valid.empty else None
            max_timestamp = valid.max() if not valid.empty else None
            self.datetime_meta[col] = DatetimeFeatureMeta(
                min_timestamp=min_timestamp,
                max_timestamp=max_timestamp,
                parse_ratio=parse_ratio,
            )

    def _fit_interactions(self) -> None:
        if not self.create_interactions:
            self.interaction_pairs = []
            return

        available = [
            col for col in self.numeric_columns
            if col in self.numeric_meta and self.numeric_meta[col].std > 0
        ]
        if len(available) < 2:
            self.interaction_pairs = []
            return

        ranked = sorted(
            available,
            key=lambda name: abs(self.numeric_meta[name].skew) + self.numeric_meta[name].std,
            reverse=True,
        )
        pairs = list(combinations(ranked[: min(len(ranked), 12)], 2))
        self.interaction_pairs = pairs[: self.max_interactions]
        self.audit_log.append(f"生成交互特征组合 {len(self.interaction_pairs)} 组")

    def _build_feature_frame(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []

        for col in self.numeric_columns:
            frames.append(self._transform_numeric_column(df[col], col))
        for col in self.boolean_columns:
            frames.append(self._transform_boolean_column(df[col], col))
        for col in self.categorical_columns:
            frames.append(self._transform_categorical_column(df[col], col))
        for col in self.datetime_columns:
            frames.append(self._transform_datetime_column(df[col], col))

        if frames:
            result = pd.concat(frames, axis=1)
        else:
            result = pd.DataFrame(index=df.index)

        if self.interaction_pairs:
            result = self._append_interactions(result)

        result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
        result = result.astype(float, copy=False)

        if not fitting and self.correlation_drop_columns:
            result = result.drop(columns=[c for c in self.correlation_drop_columns if c in result], errors="ignore")
        return result

    def _transform_numeric_column(self, series: pd.Series, col: str) -> pd.DataFrame:
        meta = self.numeric_meta[col]
        values = pd.to_numeric(series, errors="coerce")
        missing_mask = values.isna()
        filled = values.fillna(meta.median).astype(float)
        data: Dict[str, Any] = {col: filled}

        if self.create_missing_indicators and missing_mask.any():
            data[f"{col}__missing"] = missing_mask.astype(int)

        if self.create_numeric_features:
            data[f"{col}__zscore"] = (filled - meta.mean) / (meta.std if meta.std else 1.0)
            data[f"{col}__rank_pct"] = filled.rank(method="average", pct=True).fillna(0)
            data[f"{col}__iqr_flag"] = self._outlier_flags(filled, meta)

            if meta.use_log:
                data[f"{col}__log_abs"] = np.sign(filled) * np.log1p(np.abs(filled))

            if meta.bin_edges:
                data[f"{col}__bin"] = self._apply_bins(filled, meta.bin_edges)

        return pd.DataFrame(data, index=series.index)

    def _transform_boolean_column(self, series: pd.Series, col: str) -> pd.DataFrame:
        values = series.fillna(False)
        mapped = values.map(self._to_bool_value).astype(int)
        data: Dict[str, Any] = {col: mapped}
        if self.create_missing_indicators and series.isna().any():
            data[f"{col}__missing"] = series.isna().astype(int)
        return pd.DataFrame(data, index=series.index)

    def _transform_categorical_column(self, series: pd.Series, col: str) -> pd.DataFrame:
        meta = self.categorical_meta[col]
        normalized = self._normalize_category_series(series)
        is_missing = normalized.eq(meta.fill_value)
        frequency = normalized.map(meta.frequency_map).fillna(0.0)
        counts = normalized.map(meta.count_map).fillna(0).astype(float)
        is_rare = normalized.isin(meta.rare_categories).astype(int)

        data: Dict[str, Any] = {
            f"{col}__freq": frequency,
            f"{col}__count": counts,
            f"{col}__rare": is_rare,
        }
        if self.create_missing_indicators and is_missing.any():
            data[f"{col}__missing"] = is_missing.astype(int)

        for category in meta.one_hot_categories:
            safe_name = self._safe_feature_name(category)
            data[f"{col}__is__{safe_name}"] = normalized.eq(category).astype(int)

        return pd.DataFrame(data, index=series.index)

    def _transform_datetime_column(self, series: pd.Series, col: str) -> pd.DataFrame:
        parsed = self._parse_datetime(series)
        meta = self.datetime_meta[col]
        baseline = meta.min_timestamp or parsed.dropna().min()
        if pd.isna(baseline):
            baseline = pd.Timestamp("1970-01-01")

        data: Dict[str, Any] = {}
        if self.create_missing_indicators and parsed.isna().any():
            data[f"{col}__missing"] = parsed.isna().astype(int)

        if self.create_datetime_features:
            data[f"{col}__year"] = parsed.dt.year.fillna(0)
            data[f"{col}__month"] = parsed.dt.month.fillna(0)
            data[f"{col}__day"] = parsed.dt.day.fillna(0)
            data[f"{col}__dayofweek"] = parsed.dt.dayofweek.fillna(0)
            data[f"{col}__quarter"] = parsed.dt.quarter.fillna(0)
            data[f"{col}__is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).fillna(False).astype(int)
            data[f"{col}__is_month_start"] = parsed.dt.is_month_start.fillna(False).astype(int)
            data[f"{col}__is_month_end"] = parsed.dt.is_month_end.fillna(False).astype(int)
            elapsed = (parsed - baseline).dt.total_seconds() / 86400.0
            data[f"{col}__elapsed_days"] = elapsed.fillna(0)
        else:
            elapsed = (parsed - baseline).dt.total_seconds() / 86400.0
            data[f"{col}__elapsed_days"] = elapsed.fillna(0)

        return pd.DataFrame(data, index=series.index)

    def _append_interactions(self, result: pd.DataFrame) -> pd.DataFrame:
        interaction_data: Dict[str, Any] = {}
        for left, right in self.interaction_pairs:
            if left not in result.columns or right not in result.columns:
                continue
            left_values = result[left].astype(float)
            right_values = result[right].astype(float)
            interaction_data[f"{left}__x__{right}"] = left_values * right_values
            denominator = right_values.replace(0, np.nan)
            ratio = (left_values / denominator).replace([np.inf, -np.inf], np.nan).fillna(0)
            interaction_data[f"{left}__div__{right}"] = ratio
        if interaction_data:
            return pd.concat([result, pd.DataFrame(interaction_data, index=result.index)], axis=1)
        return result

    def _find_correlated_columns(self, df: pd.DataFrame) -> List[str]:
        if df.shape[1] < 2 or self.corr_threshold >= 1:
            return []

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return []

        corr = numeric_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop: List[str] = []
        for column in upper.columns:
            if any(upper[column] > self.corr_threshold):
                to_drop.append(column)
        return to_drop

    def _looks_like_datetime(self, series: pd.Series) -> bool:
        if not self.create_datetime_features:
            return False
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if pd.api.types.is_numeric_dtype(series):
            return False
        non_null = series.dropna()
        if non_null.empty:
            return False
        sample = non_null.astype(str).head(min(200, len(non_null)))
        parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        return bool(parsed.notna().mean() >= self.datetime_parse_threshold)

    def _looks_like_long_text(self, series: pd.Series) -> bool:
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            return False
        non_null = series.dropna().astype(str)
        if non_null.empty:
            return False
        avg_length = non_null.str.len().mean()
        whitespace_ratio = non_null.str.contains(r"\s+", regex=True).mean()
        return bool(avg_length >= self.max_text_length and whitespace_ratio > 0.3)

    @staticmethod
    def _parse_datetime(series: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.to_datetime(series, errors="coerce")
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

    @staticmethod
    def _normalize_category_series(series: pd.Series) -> pd.Series:
        normalized = series.astype("object").where(series.notna(), "__missing__")
        normalized = normalized.astype(str).str.strip()
        normalized = normalized.replace({"": "__missing__", "nan": "__missing__", "None": "__missing__"})
        return normalized

    @staticmethod
    def _safe_quantile_edges(values: pd.Series, bin_count: int) -> List[float]:
        if bin_count <= 1 or values.nunique() <= 1:
            return []
        quantiles = np.linspace(0, 1, bin_count + 1)
        edges = values.quantile(quantiles).drop_duplicates().astype(float).tolist()
        if len(edges) <= 2:
            return []
        return edges

    @staticmethod
    def _apply_bins(values: pd.Series, edges: Sequence[float]) -> pd.Series:
        if not edges:
            return pd.Series(0, index=values.index)
        bins = pd.cut(values, bins=list(edges), include_lowest=True, labels=False, duplicates="drop")
        return bins.fillna(0).astype(int)

    @staticmethod
    def _outlier_flags(values: pd.Series, meta: NumericFeatureMeta) -> pd.Series:
        iqr = meta.q3 - meta.q1
        if iqr <= 0:
            return pd.Series(0, index=values.index)
        lower = meta.q1 - 1.5 * iqr
        upper = meta.q3 + 1.5 * iqr
        return ((values < lower) | (values > upper)).astype(int)

    @staticmethod
    def _to_bool_value(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "是", "对"}
        return bool(value)

    @staticmethod
    def _safe_feature_name(value: Any) -> str:
        text = str(value).strip()
        if not text:
            return "empty"
        cleaned = []
        for char in text:
            if char.isalnum() or char in {"_", "-"}:
                cleaned.append(char)
            else:
                cleaned.append("_")
        name = "".join(cleaned).strip("_")
        return name[:60] or "category"
