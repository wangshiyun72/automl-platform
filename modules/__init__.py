# modules/__init__.py
"""
AutoML 平台模块初始化
"""

from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
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
    'ModelTrainer',
    'SHAPAnalyzer',
    'load_data',
    'detect_problem_type',
    'get_column_types',
    'save_model',
    'format_metrics',
    'Logger'
]