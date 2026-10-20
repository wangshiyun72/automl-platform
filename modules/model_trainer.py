# modules/model_trainer.py
"""
模型训练模块
包含模型创建、训练、评估、超参数优化等功能
"""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# 分类模型
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# 回归模型
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# 可选的高级模型
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# scikit-optimize 0.9 仍在使用 numpy 1.24 已移除的 np.int，这里做兼容处理
if not hasattr(np, "int"):
    np.int = int

try:
    from skopt import BayesSearchCV

    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

import warnings

warnings.filterwarnings('ignore')

from config import PARAM_GRIDS


class ModelTrainer:
    """
    模型训练器类
    """

    def __init__(self, problem_type="classification", random_state=42):
        """
        初始化模型训练器

        参数:
            problem_type: "classification" 或 "regression"
            random_state: 随机种子
        """
        self.problem_type = problem_type
        self.random_state = random_state
        self.models = {}  # 存储训练好的模型
        self.results = {}  # 存储评估结果
        self.best_params = {}  # 存储最佳参数

    def get_available_models(self):
        """
        获取可用的模型列表
        """
        if self.problem_type == "classification":
            models = {
                "RandomForest": ("随机森林", "⭐ 推荐算法", True),
                "GradientBoosting": ("梯度提升", "集成学习", True),
                "AdaBoost": ("AdaBoost", "集成学习", True),
                "ExtraTrees": ("极端随机树", "集成学习", True),
                "LogisticRegression": ("逻辑回归", "线性模型", True),
                "SVM": ("支持向量机", "其他算法", True),
                "KNN": ("K近邻", "其他算法", True),
                "DecisionTree": ("决策树", "其他算法", True),
                "NaiveBayes": ("朴素贝叶斯", "其他算法", True),
                "NeuralNetwork": ("神经网络(MLP)", "其他算法", True),
            }

            if XGBOOST_AVAILABLE:
                models["XGBoost"] = ("XGBoost", "⭐ 推荐算法", True)
            if LIGHTGBM_AVAILABLE:
                models["LightGBM"] = ("LightGBM", "⭐ 推荐算法", True)
            if CATBOOST_AVAILABLE:
                models["CatBoost"] = ("CatBoost", "⭐ 推荐算法", True)

        else:  # regression
            models = {
                "RandomForest": ("随机森林", "⭐ 推荐算法", True),
                "GradientBoosting": ("梯度提升", "集成学习", True),
                "AdaBoost": ("AdaBoost", "集成学习", True),
                "ExtraTrees": ("极端随机树", "集成学习", True),
                "LinearRegression": ("线性回归", "线性模型", True),
                "Ridge": ("岭回归", "线性模型", True),
                "Lasso": ("Lasso回归", "线性模型", True),
                "ElasticNet": ("弹性网络", "线性模型", True),
                "SVR": ("支持向量回归", "其他算法", True),
                "KNN": ("K近邻回归", "其他算法", True),
                "DecisionTree": ("决策树", "其他算法", True),
                "NeuralNetwork": ("神经网络(MLP)", "其他算法", True),
            }

            if XGBOOST_AVAILABLE:
                models["XGBoost"] = ("XGBoost", "⭐ 推荐算法", True)
            if LIGHTGBM_AVAILABLE:
                models["LightGBM"] = ("LightGBM", "⭐ 推荐算法", True)
            if CATBOOST_AVAILABLE:
                models["CatBoost"] = ("CatBoost", "⭐ 推荐算法", True)

        return models

    def create_model(self, model_name):
        """
        创建模型实例
        """
        if self.problem_type == "classification":
            model_dict = {
                "RandomForest": RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                "GradientBoosting": GradientBoostingClassifier(random_state=self.random_state),
                "AdaBoost": AdaBoostClassifier(random_state=self.random_state),
                "ExtraTrees": ExtraTreesClassifier(random_state=self.random_state, n_jobs=-1),
                "LogisticRegression": LogisticRegression(random_state=self.random_state, max_iter=1000),
                "SVM": SVC(random_state=self.random_state, probability=True),
                "KNN": KNeighborsClassifier(n_jobs=-1),
                "DecisionTree": DecisionTreeClassifier(random_state=self.random_state),
                "NaiveBayes": GaussianNB(),
                "NeuralNetwork": MLPClassifier(random_state=self.random_state, max_iter=1000),
            }

            if XGBOOST_AVAILABLE:
                model_dict["XGBoost"] = XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    n_jobs=-1
                )
            if LIGHTGBM_AVAILABLE:
                model_dict["LightGBM"] = LGBMClassifier(
                    random_state=self.random_state,
                    verbose=-1,
                    n_jobs=-1
                )
            if CATBOOST_AVAILABLE:
                model_dict["CatBoost"] = CatBoostClassifier(
                    random_state=self.random_state,
                    verbose=False,
                    allow_writing_files=False
                )

        else:  # regression
            model_dict = {
                "RandomForest": RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
                "GradientBoosting": GradientBoostingRegressor(random_state=self.random_state),
                "AdaBoost": AdaBoostRegressor(random_state=self.random_state),
                "ExtraTrees": ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1),
                "LinearRegression": LinearRegression(n_jobs=-1),
                "Ridge": Ridge(random_state=self.random_state),
                "Lasso": Lasso(random_state=self.random_state),
                "ElasticNet": ElasticNet(random_state=self.random_state),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor(n_jobs=-1),
                "DecisionTree": DecisionTreeRegressor(random_state=self.random_state),
                "NeuralNetwork": MLPRegressor(random_state=self.random_state, max_iter=1000),
            }

            if XGBOOST_AVAILABLE:
                model_dict["XGBoost"] = XGBRegressor(
                    random_state=self.random_state,
                    n_jobs=-1
                )
            if LIGHTGBM_AVAILABLE:
                model_dict["LightGBM"] = LGBMRegressor(
                    random_state=self.random_state,
                    verbose=-1,
                    n_jobs=-1
                )
            if CATBOOST_AVAILABLE:
                model_dict["CatBoost"] = CatBoostRegressor(
                    random_state=self.random_state,
                    verbose=False,
                    allow_writing_files=False
                )

        return model_dict.get(model_name)

    def get_param_grid(self, model_name):
        """
        获取模型的参数网格
        """
        suffix = "_clf" if self.problem_type == "classification" else "_reg"
        key = f"{model_name}{suffix}"
        return PARAM_GRIDS.get(key, {})

    def train_model(self, model_name, X_train, y_train, X_test, y_test,
                    optimization="none", n_iter=20, cv=5):
        """
        训练单个模型

        参数:
            model_name: 模型名称
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            optimization: 优化方法 ("none", "grid", "random", "bayes")
            n_iter: 随机搜索/贝叶斯搜索的迭代次数
            cv: 交叉验证折数

        返回:
            model: 训练好的模型
            train_metrics: 训练集指标
            test_metrics: 测试集指标
            best_params: 最佳参数
        """
        model = self.create_model(model_name)
        if model is None:
            raise ValueError(f"未知的模型: {model_name}")

        best_params = None
        notes = []
        param_grid = self.get_param_grid(model_name)
        start_time = time.time()

        if optimization != "none" and not param_grid:
            notes.append("该模型未配置参数网格，已使用默认参数")

        if optimization != "none" and param_grid:
            scoring = 'accuracy' if self.problem_type == "classification" else 'r2'
            cv_splitter = self._make_cv(y_train, cv)
            search = self._make_search(model, param_grid, optimization, n_iter, cv_splitter, scoring)

            try:
                search.fit(X_train, y_train)
            except Exception as e:
                if optimization != "bayes":
                    raise
                # 贝叶斯搜索失败时退回随机搜索，保证流程可以继续
                notes.append(f"贝叶斯搜索失败（{e}），已改用随机搜索")
                search = self._make_search(model, param_grid, "random", n_iter, cv_splitter, scoring)
                search.fit(X_train, y_train)

            model = search.best_estimator_
            best_params = {k: self._to_builtin(v) for k, v in dict(search.best_params_).items()}
        else:
            model.fit(X_train, y_train)

        train_time = time.time() - start_time

        # 评估模型
        train_metrics = self.evaluate_model(model, X_train, y_train)
        test_metrics = self.evaluate_model(model, X_test, y_test)

        # 存储结果
        self.models[model_name] = model
        self.results[model_name] = {
            "train": train_metrics,
            "test": test_metrics,
            "train_time": train_time,
            "notes": notes,
        }
        self.best_params[model_name] = best_params

        return model, train_metrics, test_metrics, best_params

    def _make_cv(self, y_train, cv):
        """根据样本情况调整交叉验证折数，避免类别样本过少导致报错"""
        cv = max(2, int(cv))
        if self.problem_type == "classification":
            min_class_count = int(np.bincount(np.asarray(y_train).astype(int)).min())
            if min_class_count >= 2:
                return StratifiedKFold(n_splits=min(cv, min_class_count), shuffle=True,
                                       random_state=self.random_state)
        return KFold(n_splits=min(cv, len(y_train)), shuffle=True, random_state=self.random_state)

    def _make_search(self, model, param_grid, optimization, n_iter, cv, scoring):
        if optimization == "grid":
            return GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        if optimization == "bayes" and BAYESIAN_AVAILABLE:
            return BayesSearchCV(model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring,
                                 random_state=self.random_state, n_jobs=-1)
        # 随机搜索（或贝叶斯不可用时的后备）
        n_candidates = int(np.prod([len(v) for v in param_grid.values()]))
        return RandomizedSearchCV(model, param_grid, n_iter=min(n_iter, n_candidates), cv=cv,
                                  scoring=scoring, random_state=self.random_state, n_jobs=-1)

    @staticmethod
    def _to_builtin(value):
        """把 numpy 标量转为 Python 内置类型，便于 JSON 展示与保存"""
        if isinstance(value, np.generic):
            return value.item()
        return value

    def evaluate_model(self, model, X, y):
        """
        评估模型性能
        """
        y_pred = np.asarray(model.predict(X)).ravel()

        if self.problem_type == "classification":
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y, y_pred, average='weighted', zero_division=0),
                "f1": f1_score(y, y_pred, average='weighted', zero_division=0),
            }

            # 计算 AUC（仅二分类）
            if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X)[:, 1]
                    metrics["auc"] = roc_auc_score(y, y_proba)
                except Exception:
                    pass

            # 混淆矩阵
            metrics["confusion_matrix"] = confusion_matrix(y, y_pred)

        else:  # regression
            metrics = {
                "r2": r2_score(y, y_pred),
                "mse": mean_squared_error(y, y_pred),
                "rmse": np.sqrt(mean_squared_error(y, y_pred)),
                "mae": mean_absolute_error(y, y_pred),
            }

        metrics["predictions"] = y_pred
        return metrics

    def get_model(self, model_name):
        """获取训练好的模型"""
        return self.models.get(model_name)

    def get_results(self, model_name=None):
        """获取评估结果"""
        if model_name:
            return self.results.get(model_name)
        return self.results

    def get_comparison_df(self):
        """
        获取模型对比数据框
        """
        data = []
        for model_name, result in self.results.items():
            if self.problem_type == "classification":
                data.append({
                    "模型": model_name,
                    "训练集准确率": result["train"]["accuracy"],
                    "测试集准确率": result["test"]["accuracy"],
                    "训练集F1": result["train"]["f1"],
                    "测试集F1": result["test"]["f1"],
                    "测试集AUC": result["test"].get("auc", np.nan),
                    "训练耗时(秒)": round(result.get("train_time", 0.0), 2),
                })
            else:
                data.append({
                    "模型": model_name,
                    "训练集R²": result["train"]["r2"],
                    "测试集R²": result["test"]["r2"],
                    "训练集RMSE": result["train"]["rmse"],
                    "测试集RMSE": result["test"]["rmse"],
                    "测试集MAE": result["test"]["mae"],
                    "训练耗时(秒)": round(result.get("train_time", 0.0), 2),
                })

        df = pd.DataFrame(data)
        if "测试集AUC" in df.columns and df["测试集AUC"].isna().all():
            df = df.drop(columns="测试集AUC")
        return df