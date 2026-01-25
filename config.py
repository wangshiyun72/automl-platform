# config.py
"""
AutoML平台配置文件
"""

import os
from datetime import datetime

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "shap_results")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# 创建结果目录
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ==================== 模型配置 ====================
# 分类模型
CLASSIFICATION_MODELS = {
    "RandomForest": {
        "name": "随机森林",
        "category": "⭐ 推荐算法",
        "need_scaling": False
    },
    "XGBoost": {
        "name": "XGBoost",
        "category": "⭐ 推荐算法",
        "need_scaling": False
    },
    "LightGBM": {
        "name": "LightGBM",
        "category": "⭐ 推荐算法",
        "need_scaling": False
    },
    "CatBoost": {
        "name": "CatBoost",
        "category": "⭐ 推荐算法",
        "need_scaling": False
    },
    "GradientBoosting": {
        "name": "梯度提升",
        "category": "集成学习",
        "need_scaling": False
    },
    "AdaBoost": {
        "name": "AdaBoost",
        "category": "集成学习",
        "need_scaling": False
    },
    "ExtraTrees": {
        "name": "极端随机树",
        "category": "集成学习",
        "need_scaling": False
    },
    "LogisticRegression": {
        "name": "逻辑回归",
        "category": "线性模型",
        "need_scaling": True
    },
    "SVM": {
        "name": "支持向量机",
        "category": "其他算法",
        "need_scaling": True
    },
    "KNN": {
        "name": "K近邻",
        "category": "其他算法",
        "need_scaling": True
    },
    "DecisionTree": {
        "name": "决策树",
        "category": "其他算法",
        "need_scaling": False
    },
    "NaiveBayes": {
        "name": "朴素贝叶斯",
        "category": "其他算法",
        "need_scaling": False
    },
    "NeuralNetwork": {
        "name": "神经网络(MLP)",
        "category": "其他算法",
        "need_scaling": True
    }
}

# 回归模型
REGRESSION_MODELS = {
    "RandomForest": {
        "name": "随机森林",
        "category": "⭐ 推荐算法",
        "need_scaling": False
    },
    "XGBoost": {
        "name": "XGBoost",
        "category": "⭐ 推荐算法",
        "need_scaling": False
    },
    "LightGBM": {
        "name": "LightGBM",
        "category": "⭐ 推荐算法",
        "need_scaling": False
    },
    "CatBoost": {
        "name": "CatBoost",
        "category": "⭐ 推荐算法",
        "need_scaling": False
    },
    "GradientBoosting": {
        "name": "梯度提升",
        "category": "集成学习",
        "need_scaling": False
    },
    "AdaBoost": {
        "name": "AdaBoost",
        "category": "集成学习",
        "need_scaling": False
    },
    "ExtraTrees": {
        "name": "极端随机树",
        "category": "集成学习",
        "need_scaling": False
    },
    "LinearRegression": {
        "name": "线性回归",
        "category": "线性模型",
        "need_scaling": False
    },
    "Ridge": {
        "name": "岭回归",
        "category": "线性模型",
        "need_scaling": True
    },
    "Lasso": {
        "name": "Lasso回归",
        "category": "线性模型",
        "need_scaling": True
    },
    "ElasticNet": {
        "name": "弹性网络",
        "category": "线性模型",
        "need_scaling": True
    },
    "SVR": {
        "name": "支持向量回归",
        "category": "其他算法",
        "need_scaling": True
    },
    "KNN": {
        "name": "K近邻回归",
        "category": "其他算法",
        "need_scaling": True
    },
    "DecisionTree": {
        "name": "决策树",
        "category": "其他算法",
        "need_scaling": False
    },
    "NeuralNetwork": {
        "name": "神经网络(MLP)",
        "category": "其他算法",
        "need_scaling": True
    }
}

# ==================== 超参数网格 ====================
PARAM_GRIDS = {
    # 分类模型参数
    "RandomForest_clf": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "XGBoost_clf": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.8, 1.0]
    },
    "LightGBM_clf": {
        "n_estimators": [50, 100, 200],
        "max_depth": [-1, 10, 20],
        "learning_rate": [0.01, 0.1, 0.3],
        "num_leaves": [31, 50, 100]
    },
    "CatBoost_clf": {
        "iterations": [100, 200],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1, 0.3]
    },
    "LogisticRegression_clf": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "max_iter": [1000]
    },
    "SVM_clf": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"]
    },
    "KNN_clf": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    },

    # 回归模型参数
    "RandomForest_reg": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "XGBoost_reg": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.8, 1.0]
    },
    "LightGBM_reg": {
        "n_estimators": [50, 100, 200],
        "max_depth": [-1, 10, 20],
        "learning_rate": [0.01, 0.1, 0.3],
        "num_leaves": [31, 50, 100]
    },
    "Ridge_reg": {
        "alpha": [0.01, 0.1, 1, 10, 100]
    },
    "Lasso_reg": {
        "alpha": [0.01, 0.1, 1, 10, 100]
    },
    "SVR_reg": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"]
    }
}

# ==================== 默认设置 ====================
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_SHAP_SAMPLE_RATIO = 0.2
DEFAULT_RANDOM_STATE = 42


# ==================== 日志格式 ====================
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_file_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")