# modules/__init__.py
"""
AutoML 平台模块初始化
"""

from .data_processor import DataProcessor
from .data_profiler import DataProfiler, ColumnProfile, DataQualityIssue
from .feature_engineer import FeatureEngineer, FeatureSummary
from .model_trainer import ModelTrainer
from .model_diagnostics import ModelDiagnostics, DiagnosticResult
from .predictor import (
    BatchPredictor,
    ModelArtifactManager,
    PredictionResult,
    PredictionSchema,
    load_prediction_file,
)
from .report_generator import ModelReportGenerator, ReportSection, build_standard_report
from .shap_analyzer import SHAPAnalyzer
from .utils import (
    load_data,
    detect_problem_type,
    get_column_types,
    save_model,
    format_metrics,
    Logger
)

__all__ = [
    'DataProcessor',
    'DataProfiler',
    'ColumnProfile',
    'DataQualityIssue',
    'FeatureEngineer',
    'FeatureSummary',
    'ModelTrainer',
    'ModelDiagnostics',
    'DiagnosticResult',
    'BatchPredictor',
    'ModelArtifactManager',
    'PredictionResult',
    'PredictionSchema',
    'load_prediction_file',
    'ModelReportGenerator',
    'ReportSection',
    'build_standard_report',
    'SHAPAnalyzer',
    'load_data',
    'detect_problem_type',
    'get_column_types',
    'save_model',
    'format_metrics',
    'Logger'
]
