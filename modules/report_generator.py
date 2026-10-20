"""
模型报告生成模块

提供训练过程、数据画像、模型对比、特征工程、SHAP 解释和预测结果的
统一报告能力。报告可以导出为 Markdown、HTML 或 Excel，便于项目归档、
实验复现和软著材料整理。
"""

from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from typing import Any, Dict, Iterable, List, Optional, Sequence

import os
import pandas as pd


@dataclass
class ReportSection:
    """报告章节。"""

    title: str
    content: str = ""
    table: Optional[pd.DataFrame] = None
    level: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_table(self) -> bool:
        return self.table is not None and not self.table.empty


class ModelReportGenerator:
    """
    AutoML 报告生成器。

    该类采用追加章节的方式构建报告，每个章节可以包含纯文本和 DataFrame。
    最终可渲染为 Markdown、HTML 或写入 Excel 工作簿。
    """

    def __init__(
        self,
        title: str = "AutoML 建模报告",
        author: str = "AutoML Platform",
        version: str = "1.0",
        created_at: Optional[datetime] = None,
    ):
        self.title = title
        self.author = author
        self.version = version
        self.created_at = created_at or datetime.now()
        self.sections: List[ReportSection] = []
        self.summary_items: Dict[str, Any] = {}

    def add_summary_item(self, key: str, value: Any) -> None:
        """添加报告摘要字段。"""
        self.summary_items[key] = value

    def add_section(
        self,
        title: str,
        content: str = "",
        table: Optional[pd.DataFrame] = None,
        level: int = 2,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReportSection:
        """追加一个报告章节。"""
        section = ReportSection(
            title=title,
            content=content,
            table=table,
            level=level,
            metadata=metadata or {},
        )
        self.sections.append(section)
        return section

    def add_dataset_overview(self, profiler: Any) -> None:
        """从 DataProfiler 对象添加数据概览章节。"""
        overview = profiler.get_overview()
        self.add_summary_item("样本数", overview.get("row_count"))
        self.add_summary_item("字段数", overview.get("column_count"))
        self.add_summary_item("目标列", overview.get("target_column"))
        self.add_summary_item("任务类型", overview.get("target_type"))

        content = "\n".join([
            f"- 样本数: {overview.get('row_count')}",
            f"- 字段数: {overview.get('column_count')}",
            f"- 总缺失率: {overview.get('missing_ratio', 0):.2%}",
            f"- 重复样本数: {overview.get('duplicate_rows')}",
            f"- 数据质量问题数: {overview.get('issue_count')}",
        ])
        self.add_section("数据集概览", content)

        profile_df = profiler.get_column_profile_df()
        self.add_section("字段画像", "以下为字段级类型、缺失和唯一值统计。", profile_df)

        missing_df = profiler.get_missing_report()
        if not missing_df.empty:
            self.add_section("缺失值报告", "仅展示存在缺失值的字段。", missing_df)

        issues_df = profiler.get_quality_issues()
        if not issues_df.empty:
            self.add_section("数据质量问题", "系统识别出的潜在数据质量风险。", issues_df)

        leakage_df = profiler.get_leakage_candidates()
        if not leakage_df.empty:
            self.add_section("潜在数据泄露", "以下字段需要人工确认是否可在预测时提前获得。", leakage_df)

    def add_feature_engineering(self, engineer: Any) -> None:
        """添加特征工程配置与处理结果。"""
        config = engineer.export_config()
        self.add_summary_item("输入特征数", len(config.get("input_columns", [])))
        self.add_summary_item("输出特征数", len(config.get("output_columns", [])))

        content = "\n".join([
            f"- 输入特征数: {len(config.get('input_columns', []))}",
            f"- 输出特征数: {len(config.get('output_columns', []))}",
            f"- 数值列: {len(config.get('numeric_columns', []))}",
            f"- 分类列: {len(config.get('categorical_columns', []))}",
            f"- 日期列: {len(config.get('datetime_columns', []))}",
            f"- 布尔列: {len(config.get('boolean_columns', []))}",
            f"- 删除列/特征: {len(config.get('dropped_columns', {}))}",
        ])
        self.add_section("特征工程摘要", content)

        report = engineer.get_feature_report()
        if not report.empty:
            self.add_section("特征工程明细", "记录每个原始字段的画像与处理决策。", report)

        dropped = config.get("dropped_columns", {})
        if dropped:
            dropped_df = pd.DataFrame([
                {"名称": name, "原因": reason}
                for name, reason in dropped.items()
            ])
            self.add_section("删除字段与原因", "以下字段或生成特征未进入最终训练矩阵。", dropped_df)

        audit_log = config.get("audit_log", [])
        if audit_log:
            self.add_section("特征工程日志", "\n".join([f"- {line}" for line in audit_log]))

    def add_model_comparison(self, comparison_df: pd.DataFrame, problem_type: str) -> None:
        """添加模型对比章节。"""
        if comparison_df is None or comparison_df.empty:
            self.add_section("模型对比", "暂无模型对比数据。")
            return

        best_model = self._pick_best_model(comparison_df, problem_type)
        if best_model:
            self.add_summary_item("最优模型", best_model.get("模型"))
            self.add_summary_item("最优指标", best_model.get("metric_value"))

        metric_text = self._metric_guidance(problem_type)
        content = "模型对比结果按照任务类型选择核心指标进行排序。\n\n" + metric_text
        self.add_section("模型对比", content, comparison_df)

    def add_training_results(self, results: Dict[str, Dict[str, Dict[str, Any]]], problem_type: str) -> None:
        """添加训练/测试指标详情。"""
        if not results:
            self.add_section("训练结果", "暂无训练结果。")
            return

        rows = []
        for model_name, result in results.items():
            train = result.get("train", {})
            test = result.get("test", {})
            row = {"模型": model_name}
            if problem_type == "classification":
                row.update({
                    "训练Accuracy": train.get("accuracy"),
                    "测试Accuracy": test.get("accuracy"),
                    "训练F1": train.get("f1"),
                    "测试F1": test.get("f1"),
                    "测试AUC": test.get("auc"),
                })
            else:
                row.update({
                    "训练R2": train.get("r2"),
                    "测试R2": test.get("r2"),
                    "训练RMSE": train.get("rmse"),
                    "测试RMSE": test.get("rmse"),
                    "测试MAE": test.get("mae"),
                })
            rows.append(row)

        self.add_section("训练指标详情", "训练集与测试集指标用于观察泛化能力。", pd.DataFrame(rows))

    def add_shap_summary(self, shap_analyzer: Any) -> None:
        """添加 SHAP 特征重要性摘要。"""
        if shap_analyzer is None:
            self.add_section("SHAP 解释", "尚未生成 SHAP 分析。")
            return
        try:
            importance = shap_analyzer.get_feature_importance()
        except Exception as exc:
            self.add_section("SHAP 解释", f"SHAP 摘要生成失败: {exc}")
            return

        content = "SHAP 重要性反映特征对模型输出的平均影响强度。"
        self.add_section("SHAP 特征重要性", content, importance.head(30))

    def add_prediction_summary(self, predictions_df: pd.DataFrame, target_name: str = "预测结果") -> None:
        """添加批量预测结果摘要。"""
        if predictions_df is None or predictions_df.empty:
            self.add_section("预测结果", "暂无预测结果。")
            return

        content = f"- 预测样本数: {len(predictions_df)}"
        self.add_section("预测结果概览", content)

        if target_name in predictions_df.columns:
            if pd.api.types.is_numeric_dtype(predictions_df[target_name]):
                stats = predictions_df[target_name].describe().reset_index()
                stats.columns = ["统计项", "数值"]
                self.add_section("预测值分布", "数值型预测结果的描述性统计。", stats)
            else:
                counts = predictions_df[target_name].value_counts().reset_index()
                counts.columns = ["预测类别", "样本数"]
                counts["占比"] = counts["样本数"] / counts["样本数"].sum()
                self.add_section("预测类别分布", "分类预测结果的类别分布。", counts)

    def add_artifacts(self, artifact_paths: Dict[str, str]) -> None:
        """添加模型与报告产物清单。"""
        if not artifact_paths:
            return
        rows = [
            {"产物类型": key, "路径": value}
            for key, value in artifact_paths.items()
        ]
        self.add_section("产物清单", "以下文件由平台保存，可用于复现或交付。", pd.DataFrame(rows))

    def render_markdown(self) -> str:
        """渲染 Markdown 报告。"""
        lines = [
            f"# {self.title}",
            "",
            f"- 生成时间: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 生成者: {self.author}",
            f"- 报告版本: {self.version}",
        ]

        if self.summary_items:
            lines.extend(["", "## 摘要"])
            for key, value in self.summary_items.items():
                lines.append(f"- {key}: {self._format_value(value)}")

        for section in self.sections:
            prefix = "#" * max(section.level, 1)
            lines.extend(["", f"{prefix} {section.title}"])
            if section.content:
                lines.extend(["", section.content])
            if section.has_table():
                lines.extend(["", self._dataframe_to_markdown(section.table)])

        return "\n".join(lines).strip() + "\n"

    def render_html(self) -> str:
        """渲染 HTML 报告。"""
        body = [
            "<!doctype html>",
            "<html lang=\"zh-CN\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            f"<title>{escape(self.title)}</title>",
            self._html_style(),
            "</head>",
            "<body>",
            f"<h1>{escape(self.title)}</h1>",
            "<section class=\"meta\">",
            f"<p>生成时间: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>",
            f"<p>生成者: {escape(self.author)}</p>",
            f"<p>报告版本: {escape(self.version)}</p>",
            "</section>",
        ]

        if self.summary_items:
            body.append("<h2>摘要</h2>")
            body.append("<table>")
            body.append("<thead><tr><th>项目</th><th>值</th></tr></thead><tbody>")
            for key, value in self.summary_items.items():
                body.append(f"<tr><td>{escape(str(key))}</td><td>{escape(self._format_value(value))}</td></tr>")
            body.append("</tbody></table>")

        for section in self.sections:
            level = min(max(section.level, 1), 6)
            body.append(f"<h{level}>{escape(section.title)}</h{level}>")
            if section.content:
                body.append(self._markdownish_text_to_html(section.content))
            if section.has_table():
                body.append(section.table.to_html(index=False, escape=True, border=0))

        body.extend(["</body>", "</html>"])
        return "\n".join(body)

    def export_markdown(self, path: str) -> str:
        """写入 Markdown 文件。"""
        self._ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.render_markdown())
        return path

    def export_html(self, path: str) -> str:
        """写入 HTML 文件。"""
        self._ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.render_html())
        return path

    def export_excel(self, path: str) -> str:
        """写入 Excel 工作簿。"""
        self._ensure_parent_dir(path)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            summary_rows = [
                {"项目": "报告标题", "值": self.title},
                {"项目": "生成时间", "值": self.created_at.strftime("%Y-%m-%d %H:%M:%S")},
                {"项目": "生成者", "值": self.author},
                {"项目": "报告版本", "值": self.version},
            ]
            for key, value in self.summary_items.items():
                summary_rows.append({"项目": key, "值": self._format_value(value)})
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="报告摘要", index=False)

            for index, section in enumerate(self.sections, start=1):
                sheet_name = self._safe_sheet_name(f"{index}_{section.title}")
                if section.has_table():
                    section.table.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    rows = [{"内容": line} for line in section.content.splitlines() if line.strip()]
                    pd.DataFrame(rows or [{"内容": ""}]).to_excel(writer, sheet_name=sheet_name, index=False)
        return path

    @staticmethod
    def _ensure_parent_dir(path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    @staticmethod
    def _pick_best_model(comparison_df: pd.DataFrame, problem_type: str) -> Optional[Dict[str, Any]]:
        if comparison_df.empty or "模型" not in comparison_df.columns:
            return None
        if problem_type == "classification":
            preferred = ["测试集F1", "测试集准确率", "测试Accuracy", "测试F1"]
            ascending = False
        else:
            preferred = ["测试集R²", "测试R2", "测试集RMSE", "测试RMSE"]
            ascending = "RMSE" in next((name for name in preferred if name in comparison_df.columns), "")

        metric_col = next((name for name in preferred if name in comparison_df.columns), None)
        if not metric_col:
            return None
        ordered = comparison_df.sort_values(metric_col, ascending=ascending)
        best = ordered.iloc[0].to_dict()
        best["metric_name"] = metric_col
        best["metric_value"] = best.get(metric_col)
        return best

    @staticmethod
    def _metric_guidance(problem_type: str) -> str:
        if problem_type == "classification":
            return (
                "- Accuracy 衡量整体分类正确率。\n"
                "- F1 综合考虑精确率与召回率，适合类别不均衡场景。\n"
                "- AUC 用于二分类排序能力评估。"
            )
        return (
            "- R² 衡量模型对目标变量方差的解释能力。\n"
            "- RMSE 对大误差更敏感，适合关注预测偏差的场景。\n"
            "- MAE 更直观稳定，表示平均绝对预测误差。"
        )

    @staticmethod
    def _dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
        if df is None or df.empty:
            return "_无数据_"
        view = df.head(max_rows).copy()
        columns = [str(col) for col in view.columns]
        rows = [columns]
        for _, row in view.iterrows():
            rows.append([ModelReportGenerator._format_cell(row[col]) for col in view.columns])

        widths = [
            max(len(str(row[index])) for row in rows)
            for index in range(len(columns))
        ]
        header = "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(rows[0])) + " |"
        sep = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
        body = [
            "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"
            for row in rows[1:]
        ]
        suffix = ""
        if len(df) > max_rows:
            suffix = f"\n\n_仅展示前 {max_rows} 行，共 {len(df)} 行。_"
        return "\n".join([header, sep] + body) + suffix

    @staticmethod
    def _format_cell(value: Any) -> str:
        if isinstance(value, (list, tuple, set, dict)):
            text = str(value)
        else:
            try:
                if pd.isna(value):
                    return ""
            except (TypeError, ValueError):
                text = str(value)
            else:
                if isinstance(value, float):
                    return f"{value:.6g}"
                text = str(value)
        text = text.replace("\n", " ")
        if len(text) > 120:
            text = text[:117] + "..."
        return text

    @staticmethod
    def _markdownish_text_to_html(text: str) -> str:
        lines = text.splitlines()
        html_lines: List[str] = []
        in_list = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                continue
            if stripped.startswith("- "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append(f"<li>{escape(stripped[2:])}</li>")
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<p>{escape(stripped)}</p>")
        if in_list:
            html_lines.append("</ul>")
        return "\n".join(html_lines)

    @staticmethod
    def _safe_sheet_name(name: str) -> str:
        invalid = set(r"[]:*?/\\")
        cleaned = "".join("_" if char in invalid else char for char in name)
        return cleaned[:31] or "Sheet"

    @staticmethod
    def _html_style() -> str:
        return """
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 32px;
    color: #1f2933;
    line-height: 1.6;
}
h1, h2, h3 {
    color: #0f4c81;
}
.meta {
    color: #5f6c7b;
    border-bottom: 1px solid #d9e2ec;
    margin-bottom: 24px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0 24px;
    font-size: 13px;
}
th, td {
    border: 1px solid #d9e2ec;
    padding: 8px 10px;
    text-align: left;
}
th {
    background: #f0f4f8;
}
tr:nth-child(even) {
    background: #fbfdff;
}
</style>
"""


def build_standard_report(
    title: str,
    problem_type: str,
    profiler: Optional[Any] = None,
    feature_engineer: Optional[Any] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    training_results: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    shap_analyzer: Optional[Any] = None,
    predictions_df: Optional[pd.DataFrame] = None,
    artifact_paths: Optional[Dict[str, str]] = None,
) -> ModelReportGenerator:
    """
    一次性构建标准 AutoML 报告。
    """
    report = ModelReportGenerator(title=title)
    report.add_summary_item("任务类型", problem_type)

    if profiler is not None:
        report.add_dataset_overview(profiler)
    if feature_engineer is not None:
        report.add_feature_engineering(feature_engineer)
    if comparison_df is not None:
        report.add_model_comparison(comparison_df, problem_type)
    if training_results is not None:
        report.add_training_results(training_results, problem_type)
    if shap_analyzer is not None:
        report.add_shap_summary(shap_analyzer)
    if predictions_df is not None:
        report.add_prediction_summary(predictions_df)
    if artifact_paths:
        report.add_artifacts(artifact_paths)

    return report
