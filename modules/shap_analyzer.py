# modules/shap_analyzer.py
"""
SHAP 分析模块
提供模型可解释性分析功能
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False

from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    """
    SHAP 分析器类
    """

    def __init__(self, model, X, feature_names, problem_type="classification"):
        """
        初始化 SHAP 分析器

        参数:
            model: 训练好的模型
            X: 用于计算 SHAP 值的数据
            feature_names: 特征名称列表
            problem_type: 问题类型
        """
        self.model = model
        self.X = X
        self.feature_names = feature_names
        self.problem_type = problem_type
        self.shap_values = None
        self.explainer = None
        self.expected_value = None

    def compute_shap_values(self, sample_ratio=0.2, max_samples=500):
        """
        计算 SHAP 值

        参数:
            sample_ratio: 采样比例
            max_samples: 最大样本数
        """
        # 采样
        n_samples = min(int(len(self.X) * sample_ratio), max_samples)
        n_samples = max(n_samples, min(100, len(self.X)))  # 至少100个样本

        if n_samples < len(self.X):
            indices = np.random.choice(len(self.X), n_samples, replace=False)
            X_sample = self.X[indices]
        else:
            X_sample = self.X

        # 创建解释器
        model_name = type(self.model).__name__

        # 根据模型类型选择合适的解释器
        tree_models = ['RandomForest', 'GradientBoosting', 'XGB', 'LGBM',
                       'CatBoost', 'ExtraTrees', 'DecisionTree', 'AdaBoost']

        is_tree_model = any(name in model_name for name in tree_models)

        try:
            if is_tree_model:
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(X_sample)
            else:
                # 使用 KernelExplainer 作为后备
                if hasattr(self.model, 'predict_proba') and self.problem_type == "classification":
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        shap.sample(self.X, min(100, len(self.X)))
                    )
                else:
                    self.explainer = shap.KernelExplainer(
                        self.model.predict,
                        shap.sample(self.X, min(100, len(self.X)))
                    )
                self.shap_values = self.explainer.shap_values(X_sample, nsamples=100)
        except Exception as e:
            # 最后的后备方案
            self.explainer = shap.Explainer(self.model.predict, X_sample)
            self.shap_values = self.explainer(X_sample).values

        self.expected_value = self.explainer.expected_value
        self.X_sample = X_sample

        return self.shap_values

    def get_shap_values_for_class(self, class_idx=None):
        """
        获取特定类别的 SHAP 值（多分类问题）
        """
        if self.shap_values is None:
            raise ValueError("请先计算 SHAP 值")

        if isinstance(self.shap_values, list):
            if class_idx is not None and class_idx < len(self.shap_values):
                return self.shap_values[class_idx]
            return self.shap_values[0]  # 默认返回第一个类别

        return self.shap_values

    def plot_summary(self, class_idx=None, plot_type="dot", max_display=20):
        """
        绘制摘要图

        参数:
            class_idx: 类别索引（多分类）
            plot_type: "dot" 或 "bar"
            max_display: 显示的最大特征数
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        shap_vals = self.get_shap_values_for_class(class_idx)

        if plot_type == "bar":
            shap.summary_plot(
                shap_vals,
                self.X_sample,
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                shap_vals,
                self.X_sample,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )

        plt.tight_layout()
        return fig

    def plot_bar(self, class_idx=None, max_display=20):
        """
        绘制条形图（特征重要性）
        """
        return self.plot_summary(class_idx, plot_type="bar", max_display=max_display)

    def plot_waterfall(self, sample_idx=0, class_idx=None, max_display=15):
        """
        绘制瀑布图（单样本解释）
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        shap_vals = self.get_shap_values_for_class(class_idx)

        # 获取期望值
        if isinstance(self.expected_value, (list, np.ndarray)):
            if class_idx is not None:
                exp_val = self.expected_value[class_idx]
            else:
                exp_val = self.expected_value[0]
        else:
            exp_val = self.expected_value

        # 创建 Explanation 对象
        explanation = shap.Explanation(
            values=shap_vals[sample_idx],
            base_values=exp_val,
            data=self.X_sample[sample_idx],
            feature_names=self.feature_names
        )

        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        return fig

    def plot_force(self, sample_idx=0, class_idx=None):
        """
        绘制力图（单样本）
        """
        shap_vals = self.get_shap_values_for_class(class_idx)

        if isinstance(self.expected_value, (list, np.ndarray)):
            if class_idx is not None:
                exp_val = self.expected_value[class_idx]
            else:
                exp_val = self.expected_value[0]
        else:
            exp_val = self.expected_value

        # 生成力图 HTML
        force_plot = shap.force_plot(
            exp_val,
            shap_vals[sample_idx],
            self.X_sample[sample_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )

        fig = plt.gcf()
        fig.set_size_inches(12, 3)
        plt.tight_layout()
        return fig

    def plot_dependence(self, feature_idx, class_idx=None, interaction_idx="auto"):
        """
        绘制依赖图（带置信区间）
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        shap_vals = self.get_shap_values_for_class(class_idx)

        if isinstance(feature_idx, str):
            feature_idx = self.feature_names.index(feature_idx)

        feature_name = self.feature_names[feature_idx]
        x_values = self.X_sample[:, feature_idx]
        y_values = shap_vals[:, feature_idx]

        # 绘制散点图
        scatter = ax.scatter(x_values, y_values, alpha=0.5, c='steelblue', s=20)

        # 添加趋势线和置信区间
        try:
            # 排序
            sort_idx = np.argsort(x_values)
            x_sorted = x_values[sort_idx]
            y_sorted = y_values[sort_idx]

            # 使用 LOWESS 平滑
            from scipy.ndimage import uniform_filter1d
            window_size = max(len(x_sorted) // 10, 5)
            y_smooth = uniform_filter1d(y_sorted, size=window_size)

            ax.plot(x_sorted, y_smooth, color='red', linewidth=2, label='趋势线')

            # 计算置信区间
            residuals = y_sorted - y_smooth
            std = np.std(residuals)
            ax.fill_between(x_sorted, y_smooth - 1.96 * std, y_smooth + 1.96 * std,
                            alpha=0.2, color='red', label='95% 置信区间')
        except:
            pass

        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel(f'SHAP value for {feature_name}', fontsize=12)
        ax.set_title(f'SHAP 依赖图: {feature_name}', fontsize=14)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_heatmap(self, class_idx=None, max_display=20):
        """
        绘制热力图
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        shap_vals = self.get_shap_values_for_class(class_idx)

        # 计算特征重要性排序
        importance = np.abs(shap_vals).mean(axis=0)
        top_indices = np.argsort(importance)[-max_display:][::-1]

        # 准备数据
        shap_subset = shap_vals[:, top_indices]
        feature_subset = [self.feature_names[i] for i in top_indices]

        # 绘制热力图
        im = ax.imshow(shap_subset.T, aspect='auto', cmap='RdBu_r')

        ax.set_yticks(range(len(feature_subset)))
        ax.set_yticklabels(feature_subset)
        ax.set_xlabel('样本索引', fontsize=12)
        ax.set_ylabel('特征', fontsize=12)
        ax.set_title('SHAP 值热力图', fontsize=14)

        plt.colorbar(im, ax=ax, label='SHAP value')
        plt.tight_layout()
        return fig

    def get_feature_importance(self):
        """
        获取特征重要性
        """
        if self.shap_values is None:
            raise ValueError("请先计算 SHAP 值")

        if isinstance(self.shap_values, list):
            # 多分类：取所有类别的平均
            shap_vals = np.abs(np.array(self.shap_values)).mean(axis=0)
        else:
            shap_vals = np.abs(self.shap_values)

        importance = shap_vals.mean(axis=0)

        df = pd.DataFrame({
            '特征': self.feature_names,
            '重要性': importance
        }).sort_values('重要性', ascending=False)

        return df

    def export_shap_values(self):
        """
        导出 SHAP 值为 DataFrame
        """
        if self.shap_values is None:
            raise ValueError("请先计算 SHAP 值")

        shap_vals = self.get_shap_values_for_class(0)

        df = pd.DataFrame(shap_vals, columns=self.feature_names)
        df['样本索引'] = range(len(df))

        # 添加原始特征值
        for i, name in enumerate(self.feature_names):
            df[f'{name}_原始值'] = self.X_sample[:, i]

        return df