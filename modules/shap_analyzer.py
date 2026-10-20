# modules/shap_analyzer.py
"""
SHAP 分析模块
提供模型可解释性分析功能
"""

import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import warnings

warnings.filterwarnings('ignore')

# 可由 TreeExplainer 精确计算的模型（按类名匹配）
TREE_MODELS = ('RandomForest', 'GradientBoosting', 'XGB', 'LGBM', 'CatBoost', 'ExtraTrees', 'DecisionTree')
LINEAR_MODELS = ('LogisticRegression', 'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet')


class SHAPAnalyzer:
    """
    SHAP 分析器类

    内部统一把 SHAP 值保存为:
        - 回归 / 单输出: ndarray (n_samples, n_features)
        - 多输出（多分类、二分类概率）: list[ndarray]，每个类别一个
    """

    def __init__(self, model, X, feature_names, problem_type="classification", background=None,
                 random_state=42):
        """
        初始化 SHAP 分析器

        参数:
            model: 训练好的模型
            X: 用于计算 SHAP 值的数据（通常为测试集）
            feature_names: 特征名称列表
            problem_type: 问题类型
            background: 背景数据（通常为训练集），用于 Kernel/Linear 解释器
        """
        self.model = model
        self.X = np.asarray(X)
        self.background = np.asarray(background) if background is not None else self.X
        self.feature_names = list(feature_names)
        self.problem_type = problem_type
        self.random_state = random_state
        self.shap_values = None
        self.explainer = None
        self.expected_value = None
        self.explainer_type = None
        self.X_sample = None

    # ------------------------------------------------------------------
    # 计算
    # ------------------------------------------------------------------
    def compute_shap_values(self, sample_ratio=0.2, max_samples=500):
        """
        计算 SHAP 值

        参数:
            sample_ratio: 采样比例
            max_samples: 最大样本数
        """
        rng = np.random.default_rng(self.random_state)
        model_name = type(self.model).__name__
        is_tree = any(name in model_name for name in TREE_MODELS)

        # Kernel 解释器很慢，非树模型限制样本数
        limit = max_samples if is_tree else min(max_samples, 150)
        n_samples = int(len(self.X) * sample_ratio)
        n_samples = max(n_samples, min(50, len(self.X)))
        n_samples = min(n_samples, limit, len(self.X))

        if n_samples < len(self.X):
            indices = rng.choice(len(self.X), n_samples, replace=False)
            X_sample = self.X[np.sort(indices)]
        else:
            X_sample = self.X

        background = self.background
        if len(background) > 100:
            background = background[rng.choice(len(background), 100, replace=False)]

        shap_values, expected_value = None, None
        errors = []

        if is_tree:
            try:
                self.explainer = shap.TreeExplainer(self.model)
                shap_values = self.explainer.shap_values(X_sample, check_additivity=False)
                expected_value = self.explainer.expected_value
                self.explainer_type = "TreeExplainer"
            except Exception as e:
                errors.append(f"TreeExplainer: {e}")
                shap_values = None

        if shap_values is None and any(name in model_name for name in LINEAR_MODELS):
            try:
                self.explainer = shap.LinearExplainer(self.model, background)
                shap_values = self.explainer.shap_values(X_sample)
                expected_value = self.explainer.expected_value
                self.explainer_type = "LinearExplainer"
            except Exception as e:
                errors.append(f"LinearExplainer: {e}")
                shap_values = None

        if shap_values is None:
            # 通用后备：KernelExplainer（与模型无关）
            try:
                use_proba = self.problem_type == "classification" and hasattr(self.model, 'predict_proba')
                predict_fn = self.model.predict_proba if use_proba else self.model.predict
                kernel_bg = shap.kmeans(background, min(20, len(background))) if len(background) > 20 else background
                self.explainer = shap.KernelExplainer(predict_fn, kernel_bg)
                shap_values = self.explainer.shap_values(X_sample, nsamples=min(2 * X_sample.shape[1] + 200, 500),
                                                         silent=True)
                expected_value = self.explainer.expected_value
                self.explainer_type = "KernelExplainer"
            except Exception as e:
                errors.append(f"KernelExplainer: {e}")
                raise RuntimeError("SHAP 值计算失败: " + " | ".join(errors))

        self.shap_values = self._normalize_shap_values(shap_values)
        self.expected_value = expected_value
        self.X_sample = X_sample
        return self.shap_values

    @staticmethod
    def _normalize_shap_values(values):
        """把不同解释器返回的形状统一成 ndarray 或 list[ndarray]"""
        if isinstance(values, list):
            values = [np.asarray(v) for v in values]
            return values[0] if len(values) == 1 else values
        values = np.asarray(values)
        if values.ndim == 3:  # (n_samples, n_features, n_outputs)
            if values.shape[2] == 1:
                return values[:, :, 0]
            return [values[:, :, i] for i in range(values.shape[2])]
        return values

    # ------------------------------------------------------------------
    # 取值
    # ------------------------------------------------------------------
    @property
    def n_outputs(self):
        return len(self.shap_values) if isinstance(self.shap_values, list) else 1

    def _resolve_class_idx(self, class_idx):
        """多输出时的类别索引；二分类默认解释正类 (索引 1)"""
        if not isinstance(self.shap_values, list):
            return None
        if class_idx is None:
            return 1 if len(self.shap_values) == 2 else 0
        return int(min(max(class_idx, 0), len(self.shap_values) - 1))

    def get_shap_values_for_class(self, class_idx=None):
        """
        获取特定类别的 SHAP 值（多分类问题）
        """
        if self.shap_values is None:
            raise ValueError("请先计算 SHAP 值")

        idx = self._resolve_class_idx(class_idx)
        if idx is None:
            return self.shap_values
        return self.shap_values[idx]

    def get_expected_value(self, class_idx=None):
        exp_val = self.expected_value
        if isinstance(exp_val, (list, np.ndarray)) and np.ndim(exp_val) > 0:
            exp_arr = np.asarray(exp_val).ravel()
            if len(exp_arr) == 1:
                return float(exp_arr[0])
            idx = self._resolve_class_idx(class_idx)
            return float(exp_arr[idx if idx is not None else 0])
        return float(exp_val)

    def _sample_frame(self):
        return pd.DataFrame(self.X_sample, columns=self.feature_names)

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------
    def plot_summary(self, class_idx=None, plot_type="dot", max_display=20):
        """
        绘制摘要图

        参数:
            class_idx: 类别索引（多分类）
            plot_type: "dot" 或 "bar"
            max_display: 显示的最大特征数
        """
        plt.close('all')
        shap_vals = self.get_shap_values_for_class(class_idx)
        fig = plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_vals,
            self._sample_frame(),
            plot_type="bar" if plot_type == "bar" else "dot",
            max_display=max_display,
            show=False,
            plot_size=None,
        )
        fig = plt.gcf()
        fig.tight_layout()
        return fig

    def plot_bar(self, class_idx=None, max_display=20):
        """
        绘制条形图（特征重要性）
        """
        return self.plot_summary(class_idx, plot_type="bar", max_display=max_display)

    def _explanation(self, sample_idx, class_idx):
        shap_vals = self.get_shap_values_for_class(class_idx)
        sample_idx = int(min(max(sample_idx, 0), len(shap_vals) - 1))
        return shap.Explanation(
            values=shap_vals[sample_idx],
            base_values=self.get_expected_value(class_idx),
            data=self.X_sample[sample_idx],
            feature_names=self.feature_names
        )

    def plot_waterfall(self, sample_idx=0, class_idx=None, max_display=15):
        """
        绘制瀑布图（单样本解释）
        """
        plt.close('all')
        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(self._explanation(sample_idx, class_idx), max_display=max_display, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        return fig

    def plot_force(self, sample_idx=0, class_idx=None):
        """
        绘制力图（单样本）
        """
        plt.close('all')
        explanation = self._explanation(sample_idx, class_idx)
        shap.force_plot(
            explanation.base_values,
            explanation.values,
            np.round(explanation.data, 3),
            feature_names=self.feature_names,
            matplotlib=True,
            show=False,
            text_rotation=15,
        )
        fig = plt.gcf()
        fig.set_size_inches(14, 3.6)
        return fig

    def plot_dependence(self, feature_idx, class_idx=None, interaction_idx="auto"):
        """
        绘制依赖图（带趋势线与置信带）
        """
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 6))

        shap_vals = self.get_shap_values_for_class(class_idx)

        if isinstance(feature_idx, str):
            feature_idx = self.feature_names.index(feature_idx)

        feature_name = self.feature_names[feature_idx]
        x_values = self.X_sample[:, feature_idx].astype(float)
        y_values = shap_vals[:, feature_idx]

        scatter = ax.scatter(x_values, y_values, c=y_values, cmap='coolwarm', alpha=0.75, s=28,
                             edgecolors='white', linewidths=0.4)
        fig.colorbar(scatter, ax=ax, label='SHAP 值')

        # 滑动平均趋势线 + 95% 置信带
        if len(np.unique(x_values)) > 3:
            from scipy.ndimage import uniform_filter1d
            sort_idx = np.argsort(x_values)
            x_sorted, y_sorted = x_values[sort_idx], y_values[sort_idx]
            window_size = max(len(x_sorted) // 10, 5)
            y_smooth = uniform_filter1d(y_sorted, size=window_size, mode='nearest')
            std = np.std(y_sorted - y_smooth)
            ax.plot(x_sorted, y_smooth, color='#4F46E5', linewidth=2.2, label='趋势线')
            ax.fill_between(x_sorted, y_smooth - 1.96 * std, y_smooth + 1.96 * std,
                            alpha=0.15, color='#4F46E5', label='95% 置信区间')
            ax.legend(frameon=False)

        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel(f'SHAP 值 ({feature_name})', fontsize=12)
        ax.set_title(f'SHAP 依赖图: {feature_name}', fontsize=14)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)

        fig.tight_layout()
        return fig

    def plot_heatmap(self, class_idx=None, max_display=20):
        """
        绘制热力图
        """
        plt.close('all')
        shap_vals = self.get_shap_values_for_class(class_idx)

        importance = np.abs(shap_vals).mean(axis=0)
        top_indices = np.argsort(importance)[-max_display:][::-1]
        shap_subset = shap_vals[:, top_indices]
        feature_subset = [self.feature_names[i] for i in top_indices]

        # 按样本的总贡献排序，使热力图的模式更清晰
        order = np.argsort(shap_subset.sum(axis=1))
        limit = np.abs(shap_subset).max() or 1.0

        fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(feature_subset) + 2)))
        im = ax.imshow(shap_subset[order].T, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)

        ax.set_yticks(range(len(feature_subset)))
        ax.set_yticklabels(feature_subset)
        ax.set_xlabel('样本（按总贡献排序）', fontsize=12)
        ax.set_ylabel('特征', fontsize=12)
        ax.set_title('SHAP 值热力图', fontsize=14)

        fig.colorbar(im, ax=ax, label='SHAP 值')
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def get_feature_importance(self, class_idx=None):
        """
        获取特征重要性（平均 |SHAP|）。
        class_idx 为 None 且为多分类时，取所有类别的平均。
        """
        if self.shap_values is None:
            raise ValueError("请先计算 SHAP 值")

        if isinstance(self.shap_values, list) and class_idx is None:
            shap_vals = np.mean([np.abs(v) for v in self.shap_values], axis=0)
        else:
            shap_vals = np.abs(self.get_shap_values_for_class(class_idx))

        return pd.DataFrame({
            '特征': self.feature_names,
            '重要性': shap_vals.mean(axis=0)
        }).sort_values('重要性', ascending=False).reset_index(drop=True)

    def export_shap_values(self, class_idx=None):
        """
        导出 SHAP 值为 DataFrame
        """
        if self.shap_values is None:
            raise ValueError("请先计算 SHAP 值")

        shap_vals = self.get_shap_values_for_class(class_idx)

        df = pd.DataFrame(shap_vals, columns=[f'{name}_SHAP' for name in self.feature_names])
        df.insert(0, '样本索引', range(len(df)))

        for i, name in enumerate(self.feature_names):
            df[f'{name}_特征值'] = self.X_sample[:, i]

        return df
