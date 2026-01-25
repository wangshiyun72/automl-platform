# app.py
"""
AutoML 平台主程序
基于 Streamlit 的自动化机器学习平台
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import io
import json
import joblib
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from modules import (
    DataProcessor, ModelTrainer, SHAPAnalyzer,
    load_data, detect_problem_type, get_column_types,
    save_model, format_metrics, Logger
)
from config import RESULTS_DIR, get_timestamp, get_file_timestamp

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="AutoML 自动化机器学习平台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义样式 ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.1rem;
        font-weight: bold;
        color: #333;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-color: #ffeeba;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 初始化 Session State ====================
def init_session_state():
    """初始化会话状态"""
    defaults = {
        'data': None,
        'data_processed': False,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'feature_names': [],
        'target_column': None,
        'problem_type': None,
        'data_processor': None,
        'model_trainer': None,
        'shap_analyzer': None,
        'trained_models': {},
        'current_model': None,
        'shap_computed': False,
        'logger': Logger(),
        'processing_log': [],
        'predictions': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ==================== 辅助函数 ====================
def log_message(message, level="INFO"):
    """记录日志消息"""
    st.session_state.logger.log(message, level)


def reset_downstream_state():
    """重置下游状态（当上游设置改变时）"""
    st.session_state.data_processed = False
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.trained_models = {}
    st.session_state.current_model = None
    st.session_state.shap_computed = False
    st.session_state.shap_analyzer = None


# ==================== 主界面 ====================
def main():
    # 标题
    st.markdown('<p class="main-header">🤖 AutoML 自动化机器学习平台</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">零代码构建机器学习模型 | 深度集成 SHAP 可解释性分析</p>', unsafe_allow_html=True)

    # 创建左右两栏布局
    col_left, col_right = st.columns([1, 2])

    # ==================== 左侧控制台 ====================
    with col_left:
        st.markdown("## 📋 控制台")

        # ========== 步骤 1: 数据加载 ==========
        st.markdown('<p class="step-header">📁 步骤 1: 数据加载</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "选择数据文件",
            type=['csv', 'xlsx', 'xls'],
            help="支持 CSV, XLSX, XLS 格式"
        )

        if uploaded_file is not None:
            if st.session_state.data is None or uploaded_file.name != st.session_state.get('uploaded_filename'):
                df, error = load_data(uploaded_file)
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.session_state.data = df
                    st.session_state.uploaded_filename = uploaded_file.name
                    reset_downstream_state()
                    log_message(f"数据加载成功: {uploaded_file.name}, 维度: {df.shape}")
                    st.success(f"✅ 已加载: {uploaded_file.name} ({df.shape[0]} 行 × {df.shape[1]} 列)")
            else:
                st.success(
                    f"✅ 已加载: {uploaded_file.name} ({st.session_state.data.shape[0]} 行 × {st.session_state.data.shape[1]} 列)")

        # 如果数据已加载，显示后续步骤
        if st.session_state.data is not None:
            df = st.session_state.data

            # ========== 步骤 2: 选择目标列 ==========
            st.markdown('<p class="step-header">🎯 步骤 2: 选择目标列</p>', unsafe_allow_html=True)

            target_column = st.selectbox(
                "选择要预测的目标列",
                options=df.columns.tolist(),
                index=len(df.columns) - 1,
                help="选择您希望模型预测的列（Label/Target）"
            )

            if target_column != st.session_state.target_column:
                st.session_state.target_column = target_column
                reset_downstream_state()

            # 显示目标列信息
            col_info = get_column_types(df)
            target_info = col_info[target_column]
            st.info(
                f"📊 目标列信息: {target_info['type']} | 唯一值: {target_info['unique_count']} | 缺失: {target_info['null_ratio']}")

            # ========== 步骤 3: 特征工程 ==========
            st.markdown('<p class="step-header">✅ 步骤 3: 特征工程</p>', unsafe_allow_html=True)

            # 特征选择
            available_features = [col for col in df.columns if col != target_column]

            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                if st.button("全选", key="select_all"):
                    st.session_state.selected_features = available_features
            with col_sel2:
                if st.button("全不选", key="select_none"):
                    st.session_state.selected_features = []

            if 'selected_features' not in st.session_state:
                st.session_state.selected_features = available_features

            selected_features = st.multiselect(
                "选择特征列",
                options=available_features,
                default=st.session_state.selected_features,
                help="选择用于训练模型的特征列"
            )
            st.session_state.selected_features = selected_features

            if len(selected_features) == 0:
                st.warning("⚠️ 请至少选择一个特征列")
            else:
                st.success(f"✅ 已选择 {len(selected_features)} 个特征")

            # 步骤 3.1: 训练集比例
            st.markdown("**📊 步骤 3.1: 训练集比例**")
            train_ratio = st.slider(
                "训练集比例",
                min_value=0.5,
                max_value=0.9,
                value=0.7,
                step=0.05,
                help="设置用于训练的数据比例，剩余部分用于测试"
            )

            # 步骤 3.2: 数据标准化
            st.markdown("**📐 步骤 3.2: 数据标准化**")
            scaling_method = st.selectbox(
                "标准化方法",
                options=["none", "zscore", "minmax"],
                format_func=lambda x: {
                    "none": "无标准化（推荐树模型）",
                    "zscore": "Z-score 标准化（推荐SVM/KNN/神经网络）",
                    "minmax": "MinMax 归一化 [0,1]"
                }[x],
                help="选择数据标准化方法"
            )

            # SHAP 样本设置
            st.markdown("**📝 SHAP 样本设置**")
            shap_sample_ratio = st.slider(
                "SHAP 计算样本比例",
                min_value=0.05,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="从测试集中抽取的样本比例用于计算 SHAP 值"
            )

            # ========== 步骤 4: 问题类型 ==========
            st.markdown('<p class="step-header">🔍 步骤 4: 问题类型</p>', unsafe_allow_html=True)

            # 自动检测
            auto_detected = detect_problem_type(df[target_column])
            auto_label = "分类问题" if auto_detected == "classification" else "回归问题"

            st.info(f"🤖 自动检测: {auto_label}")

            problem_type = st.radio(
                "确认问题类型",
                options=["classification", "regression"],
                format_func=lambda x: "分类问题" if x == "classification" else "回归问题",
                index=0 if auto_detected == "classification" else 1,
                horizontal=True
            )

            if problem_type != st.session_state.problem_type:
                st.session_state.problem_type = problem_type
                reset_downstream_state()

            # ========== 步骤 5: 算法选择 ==========
            st.markdown('<p class="step-header">⚙️ 步骤 5: 算法选择</p>', unsafe_allow_html=True)

            # 创建模型训练器以获取可用模型
            temp_trainer = ModelTrainer(problem_type=problem_type)
            available_models = temp_trainer.get_available_models()

            # 按类别分组显示
            categories = {}
            for model_key, (model_name, category, available) in available_models.items():
                if category not in categories:
                    categories[category] = []
                categories[category].append((model_key, model_name))

            selected_models = []
            for category, models in categories.items():
                st.markdown(f"**{category}**")
                cols = st.columns(3)
                for i, (model_key, model_name) in enumerate(models):
                    with cols[i % 3]:
                        if st.checkbox(model_name, value=(category == "⭐ 推荐算法"), key=f"model_{model_key}"):
                            selected_models.append(model_key)

            if len(selected_models) == 0:
                st.warning("⚠️ 请至少选择一个算法")
            else:
                st.success(f"✅ 已选择 {len(selected_models)} 个算法")

            # ========== 步骤 6: 参数优化 ==========
            st.markdown('<p class="step-header">🔧 步骤 6: 参数优化</p>', unsafe_allow_html=True)

            optimization = st.selectbox(
                "超参数优化方法",
                options=["none", "grid", "random", "bayes"],
                format_func=lambda x: {
                    "none": "无优化（使用默认参数，最快）",
                    "grid": "网格搜索（最准，最慢）",
                    "random": "随机搜索（速度与效果平衡）",
                    "bayes": "贝叶斯搜索（智能推荐，高效）"
                }[x]
            )

            if optimization in ["random", "bayes"]:
                n_iter = st.slider("搜索迭代次数", 10, 100, 20)
            else:
                n_iter = 20

            cv_folds = st.slider("交叉验证折数", 2, 10, 5)

            # ========== 步骤 7: 训练模型 ==========
            st.markdown('<p class="step-header">🚀 步骤 7: 训练模型</p>', unsafe_allow_html=True)

            if st.button("🚀 开始训练", type="primary",
                         disabled=(len(selected_features) == 0 or len(selected_models) == 0)):
                with st.spinner("正在处理数据和训练模型..."):
                    try:
                        # 数据处理
                        log_message("开始数据预处理...")
                        data_processor = DataProcessor()
                        X, y, processing_log = data_processor.fit_transform(
                            df, target_column, selected_features, scaling_method
                        )

                        for log in processing_log:
                            log_message(log)

                        st.session_state.data_processor = data_processor
                        st.session_state.feature_names = selected_features

                        # 划分数据集
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, train_size=train_ratio, random_state=42
                        )

                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test

                        log_message(f"数据划分完成: 训练集 {X_train.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本")

                        # 训练模型
                        model_trainer = ModelTrainer(problem_type=problem_type)
                        st.session_state.model_trainer = model_trainer

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i, model_name in enumerate(selected_models):
                            status_text.text(f"正在训练: {model_name} ({i + 1}/{len(selected_models)})")
                            log_message(f"开始训练模型: {model_name}")

                            try:
                                model, train_metrics, test_metrics, best_params = model_trainer.train_model(
                                    model_name, X_train, y_train, X_test, y_test,
                                    optimization=optimization, n_iter=n_iter, cv=cv_folds
                                )

                                st.session_state.trained_models[model_name] = {
                                    'model': model,
                                    'train_metrics': train_metrics,
                                    'test_metrics': test_metrics,
                                    'best_params': best_params
                                }

                                if problem_type == "classification":
                                    log_message(f"{model_name} 训练完成 - 测试集准确率: {test_metrics['accuracy']:.4f}",
                                                "SUCCESS")
                                else:
                                    log_message(f"{model_name} 训练完成 - 测试集R²: {test_metrics['r2']:.4f}",
                                                "SUCCESS")

                            except Exception as e:
                                log_message(f"{model_name} 训练失败: {str(e)}", "ERROR")

                            progress_bar.progress((i + 1) / len(selected_models))

                        status_text.text("训练完成！")
                        st.session_state.data_processed = True

                        # 设置默认当前模型
                        if st.session_state.trained_models:
                            st.session_state.current_model = list(st.session_state.trained_models.keys())[0]

                        st.success(f"✅ 成功训练 {len(st.session_state.trained_models)} 个模型！")
                        st.rerun()

                    except Exception as e:
                        log_message(f"训练过程出错: {str(e)}", "ERROR")
                        st.error(f"❌ 训练失败: {str(e)}")

            # ========== 步骤 8: 选择模型分析 ==========
            if st.session_state.trained_models:
                st.markdown('<p class="step-header">🎯 步骤 8: 选择模型分析</p>', unsafe_allow_html=True)

                model_options = list(st.session_state.trained_models.keys())
                current_model = st.selectbox(
                    "选择要分析的模型",
                    options=model_options,
                    index=model_options.index(
                        st.session_state.current_model) if st.session_state.current_model in model_options else 0
                )
                st.session_state.current_model = current_model

                # 多分类时选择类别
                if problem_type == "classification":
                    target_classes = st.session_state.data_processor.get_target_classes()
                    if target_classes is not None and len(target_classes) > 2:
                        st.markdown("**🎨 步骤 8.5: 选择分析类别**")
                        class_idx = st.selectbox(
                            "选择要分析的类别",
                            options=range(len(target_classes)),
                            format_func=lambda x: f"类别 {x}: {target_classes[x]}"
                        )
                        st.session_state.class_idx = class_idx
                    else:
                        st.session_state.class_idx = None
                else:
                    st.session_state.class_idx = None

                # ========== 步骤 9: SHAP 可视化 ==========
                st.markdown('<p class="step-header">📊 步骤 9: SHAP 可视化</p>', unsafe_allow_html=True)

                shap_plot_type = st.selectbox(
                    "选择图表类型",
                    options=["summary", "bar", "waterfall", "force", "dependence", "heatmap"],
                    format_func=lambda x: {
                        "summary": "📊 摘要图 (Summary Plot)",
                        "bar": "📊 条形图 (Bar Plot)",
                        "waterfall": "🌊 瀑布图 (Waterfall Plot)",
                        "force": "💪 力图 (Force Plot)",
                        "dependence": "📈 依赖图 (Dependence Plot)",
                        "heatmap": "🔥 热力图 (Heatmap)"
                    }[x]
                )

                # 依赖图需要选择特征
                if shap_plot_type == "dependence":
                    dep_feature = st.selectbox(
                        "选择特征",
                        options=st.session_state.feature_names
                    )
                else:
                    dep_feature = None

                if st.button("🎨 生成 SHAP 图表"):
                    with st.spinner("正在计算 SHAP 值并生成图表..."):
                        try:
                            model_data = st.session_state.trained_models[current_model]
                            model = model_data['model']

                            # 创建 SHAP 分析器
                            shap_analyzer = SHAPAnalyzer(
                                model=model,
                                X=st.session_state.X_test,
                                feature_names=st.session_state.feature_names,
                                problem_type=problem_type
                            )

                            # 计算 SHAP 值
                            log_message("开始计算 SHAP 值...")
                            shap_analyzer.compute_shap_values(sample_ratio=shap_sample_ratio)
                            st.session_state.shap_analyzer = shap_analyzer
                            st.session_state.shap_computed = True
                            log_message("SHAP 值计算完成", "SUCCESS")

                            # 生成图表
                            class_idx = st.session_state.get('class_idx')

                            if shap_plot_type == "summary":
                                fig = shap_analyzer.plot_summary(class_idx=class_idx)
                            elif shap_plot_type == "bar":
                                fig = shap_analyzer.plot_bar(class_idx=class_idx)
                            elif shap_plot_type == "waterfall":
                                fig = shap_analyzer.plot_waterfall(class_idx=class_idx)
                            elif shap_plot_type == "force":
                                fig = shap_analyzer.plot_force(class_idx=class_idx)
                            elif shap_plot_type == "dependence":
                                fig = shap_analyzer.plot_dependence(dep_feature, class_idx=class_idx)
                            elif shap_plot_type == "heatmap":
                                fig = shap_analyzer.plot_heatmap(class_idx=class_idx)

                            st.session_state.current_shap_fig = fig
                            st.success("✅ 图表生成成功！请在右侧查看")

                        except Exception as e:
                            log_message(f"SHAP 分析失败: {str(e)}", "ERROR")
                            st.error(f"❌ SHAP 分析失败: {str(e)}")

                # ========== 步骤 10: 结果导出 ==========
                st.markdown('<p class="step-header">💾 步骤 10: 结果导出</p>', unsafe_allow_html=True)

                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    if st.button("📊 保存模型对比"):
                        try:
                            comparison_df = st.session_state.model_trainer.get_comparison_df()
                            timestamp = get_file_timestamp()
                            save_path = os.path.join(RESULTS_DIR, f"model_comparison_{timestamp}.csv")
                            comparison_df.to_csv(save_path, index=False, encoding='utf-8-sig')
                            st.success(f"✅ 已保存到: {save_path}")
                        except Exception as e:
                            st.error(f"❌ 保存失败: {str(e)}")

                with col_exp2:
                    if st.button("💾 导出模型参数"):
                        try:
                            model_data = st.session_state.trained_models[current_model]
                            timestamp = get_file_timestamp()
                            save_dir = os.path.join(RESULTS_DIR, f"{current_model}_{timestamp}")

                            paths = save_model(
                                model=model_data['model'],
                                model_name=current_model,
                                save_dir=save_dir,
                                scaler=st.session_state.data_processor.scaler,
                                label_encoders=st.session_state.data_processor.label_encoders,
                                feature_names=st.session_state.feature_names,
                                model_params=model_data.get('best_params')
                            )
                            st.success(f"✅ 模型已保存到: {save_dir}")
                        except Exception as e:
                            st.error(f"❌ 保存失败: {str(e)}")

                if st.session_state.shap_computed and st.button("📈 导出 SHAP 值"):
                    try:
                        shap_df = st.session_state.shap_analyzer.export_shap_values()
                        timestamp = get_file_timestamp()
                        save_path = os.path.join(RESULTS_DIR, f"shap_values_{timestamp}.xlsx")
                        shap_df.to_excel(save_path, index=False)
                        st.success(f"✅ SHAP 值已保存到: {save_path}")
                    except Exception as e:
                        st.error(f"❌ 保存失败: {str(e)}")

            # ========== 预测新数据 ==========
            st.markdown("---")
            st.markdown('<p class="step-header">🔮 预测新数据</p>', unsafe_allow_html=True)

            if st.session_state.trained_models:
                predict_file = st.file_uploader(
                    "上传新数据进行预测",
                    type=['csv', 'xlsx', 'xls'],
                    key="predict_file"
                )

                if predict_file is not None:
                    predict_model = st.selectbox(
                        "选择预测模型",
                        options=list(st.session_state.trained_models.keys()),
                        key="predict_model_select"
                    )

                    if st.button("🔮 开始预测"):
                        with st.spinner("正在预测..."):
                            try:
                                new_df, error = load_data(predict_file)
                                if error:
                                    st.error(f"❌ {error}")
                                else:
                                    # 处理新数据
                                    X_new = st.session_state.data_processor.transform(new_df)

                                    # 预测
                                    model = st.session_state.trained_models[predict_model]['model']
                                    predictions = model.predict(X_new)

                                    # 如果是分类问题，获取概率
                                    if problem_type == "classification" and hasattr(model, 'predict_proba'):
                                        probabilities = model.predict_proba(X_new)
                                    else:
                                        probabilities = None

                                    # 反转标签编码
                                    if st.session_state.data_processor.target_encoder is not None:
                                        predictions = st.session_state.data_processor.inverse_transform_target(
                                            predictions)

                                    # 创建结果 DataFrame
                                    result_df = new_df.copy()
                                    result_df['预测结果'] = predictions

                                    if probabilities is not None:
                                        classes = st.session_state.data_processor.get_target_classes()
                                        for i, cls in enumerate(classes):
                                            result_df[f'概率_{cls}'] = probabilities[:, i]

                                    st.session_state.predictions = result_df
                                    st.success("✅ 预测完成！请在右侧查看结果")

                            except Exception as e:
                                st.error(f"❌ 预测失败: {str(e)}")
                                log_message(f"预测失败: {str(e)}", "ERROR")

    # ==================== 右侧展示板 ====================
    with col_right:
        st.markdown("## 📊 展示板")

        # 创建标签页
        tabs = st.tabs([
            "📊 SHAP 可视化",
            "📈 模型对比",
            "📋 模型评估",
            "👁️ 数据预览",
            "📋 模型详情",
            "📝 运行日志",
            "🔮 预测结果"
        ])

        # Tab 1: SHAP 可视化
        with tabs[0]:
            st.markdown("### 📊 SHAP 可视化")
            if 'current_shap_fig' in st.session_state and st.session_state.current_shap_fig is not None:
                st.pyplot(st.session_state.current_shap_fig)

                # 保存选项
                col_save1, col_save2, col_save3 = st.columns(3)
                with col_save1:
                    if st.button("💾 保存 PNG"):
                        timestamp = get_file_timestamp()
                        save_path = os.path.join(RESULTS_DIR, f"shap_plot_{timestamp}.png")
                        st.session_state.current_shap_fig.savefig(save_path, dpi=300, bbox_inches='tight')
                        st.success(f"✅ 已保存: {save_path}")
                with col_save2:
                    if st.button("💾 保存 SVG"):
                        timestamp = get_file_timestamp()
                        save_path = os.path.join(RESULTS_DIR, f"shap_plot_{timestamp}.svg")
                        st.session_state.current_shap_fig.savefig(save_path, format='svg', bbox_inches='tight')
                        st.success(f"✅ 已保存: {save_path}")
                with col_save3:
                    if st.button("💾 保存 PDF"):
                        timestamp = get_file_timestamp()
                        save_path = os.path.join(RESULTS_DIR, f"shap_plot_{timestamp}.pdf")
                        st.session_state.current_shap_fig.savefig(save_path, format='pdf', bbox_inches='tight')
                        st.success(f"✅ 已保存: {save_path}")
            else:
                st.info("👈 请在左侧控制台选择模型并生成 SHAP 图表")

                # Tab 2: 模型对比
                with tabs[1]:
                    st.markdown("### 📈 模型对比")
                    if st.session_state.trained_models:
                        comparison_df = st.session_state.model_trainer.get_comparison_df()

                        # 显示对比表格
                        st.dataframe(comparison_df, use_container_width=True)

                        # 绘制对比图
                        problem_type = st.session_state.problem_type

                        if problem_type == "classification":
                            # 准确率对比图
                            fig = make_subplots(rows=1, cols=2, subplot_titles=('准确率对比', 'F1分数对比'))

                            fig.add_trace(
                                go.Bar(name='训练集', x=comparison_df['模型'], y=comparison_df['训练集准确率'],
                                       marker_color='steelblue'),
                                row=1, col=1
                            )
                            fig.add_trace(
                                go.Bar(name='测试集', x=comparison_df['模型'], y=comparison_df['测试集准确率'],
                                       marker_color='coral'),
                                row=1, col=1
                            )

                            fig.add_trace(
                                go.Bar(name='训练集', x=comparison_df['模型'], y=comparison_df['训练集F1'],
                                       marker_color='steelblue', showlegend=False),
                                row=1, col=2
                            )
                            fig.add_trace(
                                go.Bar(name='测试集', x=comparison_df['模型'], y=comparison_df['测试集F1'],
                                       marker_color='coral', showlegend=False),
                                row=1, col=2
                            )

                            fig.update_layout(barmode='group', height=400)
                            st.plotly_chart(fig, use_container_width=True)

                        else:
                            # R² 和 RMSE 对比图
                            fig = make_subplots(rows=1, cols=2, subplot_titles=('R² 分数对比', 'RMSE 对比'))

                            fig.add_trace(
                                go.Bar(name='训练集', x=comparison_df['模型'], y=comparison_df['训练集R²'],
                                       marker_color='steelblue'),
                                row=1, col=1
                            )
                            fig.add_trace(
                                go.Bar(name='测试集', x=comparison_df['模型'], y=comparison_df['测试集R²'],
                                       marker_color='coral'),
                                row=1, col=1
                            )

                            fig.add_trace(
                                go.Bar(name='训练集', x=comparison_df['模型'], y=comparison_df['训练集RMSE'],
                                       marker_color='steelblue', showlegend=False),
                                row=1, col=2
                            )
                            fig.add_trace(
                                go.Bar(name='测试集', x=comparison_df['模型'], y=comparison_df['测试集RMSE'],
                                       marker_color='coral', showlegend=False),
                                row=1, col=2
                            )

                            fig.update_layout(barmode='group', height=400)
                            st.plotly_chart(fig, use_container_width=True)

                        # 过拟合分析
                        st.markdown("#### 📊 过拟合分析")
                        if problem_type == "classification":
                            comparison_df['过拟合程度'] = comparison_df['训练集准确率'] - comparison_df['测试集准确率']
                        else:
                            comparison_df['过拟合程度'] = comparison_df['训练集R²'] - comparison_df['测试集R²']

                        for _, row in comparison_df.iterrows():
                            overfit = row['过拟合程度']
                            if overfit > 0.1:
                                st.warning(f"⚠️ {row['模型']}: 过拟合程度 {overfit:.4f} (较高)")
                            elif overfit > 0.05:
                                st.info(f"ℹ️ {row['模型']}: 过拟合程度 {overfit:.4f} (中等)")
                            else:
                                st.success(f"✅ {row['模型']}: 过拟合程度 {overfit:.4f} (良好)")
                    else:
                        st.info("👈 请先在左侧训练模型")

                # Tab 3: 模型评估
                with tabs[2]:
                    st.markdown("### 📋 模型评估详情")
                    if st.session_state.trained_models and st.session_state.current_model:
                        current_model = st.session_state.current_model
                        model_data = st.session_state.trained_models[current_model]
                        problem_type = st.session_state.problem_type

                        st.markdown(f"**当前模型: {current_model}**")

                        col_metric1, col_metric2 = st.columns(2)

                        with col_metric1:
                            st.markdown("##### 训练集指标")
                            train_metrics = model_data['train_metrics']
                            if problem_type == "classification":
                                st.metric("准确率", f"{train_metrics['accuracy']:.4f}")
                                st.metric("精确率", f"{train_metrics['precision']:.4f}")
                                st.metric("召回率", f"{train_metrics['recall']:.4f}")
                                st.metric("F1分数", f"{train_metrics['f1']:.4f}")
                            else:
                                st.metric("R² 分数", f"{train_metrics['r2']:.4f}")
                                st.metric("MSE", f"{train_metrics['mse']:.4f}")
                                st.metric("RMSE", f"{train_metrics['rmse']:.4f}")
                                st.metric("MAE", f"{train_metrics['mae']:.4f}")

                        with col_metric2:
                            st.markdown("##### 测试集指标")
                            test_metrics = model_data['test_metrics']
                            if problem_type == "classification":
                                st.metric("准确率", f"{test_metrics['accuracy']:.4f}")
                                st.metric("精确率", f"{test_metrics['precision']:.4f}")
                                st.metric("召回率", f"{test_metrics['recall']:.4f}")
                                st.metric("F1分数", f"{test_metrics['f1']:.4f}")
                            else:
                                st.metric("R² 分数", f"{test_metrics['r2']:.4f}")
                                st.metric("MSE", f"{test_metrics['mse']:.4f}")
                                st.metric("RMSE", f"{test_metrics['rmse']:.4f}")
                                st.metric("MAE", f"{test_metrics['mae']:.4f}")

                        # 分类问题：混淆矩阵和ROC曲线
                        if problem_type == "classification":
                            st.markdown("---")
                            st.markdown("##### 混淆矩阵")

                            cm = test_metrics['confusion_matrix']
                            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                            ax_cm.set_xlabel('预测标签')
                            ax_cm.set_ylabel('真实标签')
                            ax_cm.set_title(f'{current_model} - 混淆矩阵')
                            st.pyplot(fig_cm)
                            plt.close()

                            # ROC曲线（二分类）
                            if 'auc' in test_metrics:
                                st.markdown("##### ROC 曲线")
                                from sklearn.metrics import roc_curve

                                model = model_data['model']
                                if hasattr(model, 'predict_proba'):
                                    y_proba = model.predict_proba(st.session_state.X_test)[:, 1]
                                    fpr, tpr, _ = roc_curve(st.session_state.y_test, y_proba)

                                    fig_roc = go.Figure()
                                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                                                 name=f'ROC (AUC={test_metrics["auc"]:.4f})'))
                                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='随机猜测',
                                                                 line=dict(dash='dash')))
                                    fig_roc.update_layout(
                                        title='ROC 曲线',
                                        xaxis_title='假阳性率 (FPR)',
                                        yaxis_title='真阳性率 (TPR)',
                                        height=400
                                    )
                                    st.plotly_chart(fig_roc, use_container_width=True)

                        # 回归问题：残差图
                        else:
                            st.markdown("---")
                            st.markdown("##### 残差分析")

                            y_pred = test_metrics['predictions']
                            y_true = st.session_state.y_test
                            residuals = y_true - y_pred

                            fig_res = make_subplots(rows=1, cols=2, subplot_titles=('预测值 vs 真实值', '残差分布'))

                            # 预测值 vs 真实值
                            fig_res.add_trace(
                                go.Scatter(x=y_true, y=y_pred, mode='markers', name='数据点', marker=dict(opacity=0.5)),
                                row=1, col=1
                            )
                            min_val = min(y_true.min(), y_pred.min())
                            max_val = max(y_true.max(), y_pred.max())
                            fig_res.add_trace(
                                go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='理想线',
                                           line=dict(dash='dash', color='red')),
                                row=1, col=1
                            )

                            # 残差分布
                            fig_res.add_trace(
                                go.Histogram(x=residuals, name='残差', nbinsx=30),
                                row=1, col=2
                            )

                            fig_res.update_layout(height=400)
                            st.plotly_chart(fig_res, use_container_width=True)

                        # 最佳参数
                        if model_data.get('best_params'):
                            st.markdown("---")
                            st.markdown("##### 最佳超参数")
                            st.json(model_data['best_params'])

                        # 高级保存选项
                        st.markdown("---")
                        st.markdown("##### 高级保存选项")
                        save_format = st.selectbox(
                            "选择保存格式",
                            options=["PNG (300 DPI)", "SVG (矢量图)", "PDF (矢量图)", "EPS (矢量图)"],
                            key="eval_save_format"
                        )

                        if st.button("💾 保存评估图表", key="save_eval"):
                            timestamp = get_file_timestamp()
                            format_map = {
                                "PNG (300 DPI)": ("png", 300),
                                "SVG (矢量图)": ("svg", None),
                                "PDF (矢量图)": ("pdf", None),
                                "EPS (矢量图)": ("eps", None)
                            }
                            fmt, dpi = format_map[save_format]
                            save_path = os.path.join(RESULTS_DIR, f"evaluation_{current_model}_{timestamp}.{fmt}")

                            if problem_type == "classification":
                                fig_cm.savefig(save_path, format=fmt, dpi=dpi, bbox_inches='tight')
                            st.success(f"✅ 已保存: {save_path}")
                    else:
                        st.info("👈 请先在左侧训练模型")

                # Tab 4: 数据预览
                with tabs[3]:
                    st.markdown("### 👁️ 数据预览")
                    if st.session_state.data is not None:
                        df = st.session_state.data

                        # 基本信息
                        col_info1, col_info2, col_info3 = st.columns(3)
                        with col_info1:
                            st.metric("行数", df.shape[0])
                        with col_info2:
                            st.metric("列数", df.shape[1])
                        with col_info3:
                            st.metric("缺失值", df.isnull().sum().sum())

                        # 数据表格
                        st.markdown("##### 数据表格")
                        st.dataframe(df.head(100), use_container_width=True)

                        # 列信息
                        st.markdown("##### 列信息")
                        col_info = get_column_types(df)
                        col_info_df = pd.DataFrame([
                            {
                                "列名": col,
                                "类型": info["type"],
                                "数据类型": info["dtype"],
                                "唯一值数": info["unique_count"],
                                "缺失值数": info["null_count"],
                                "缺失比例": info["null_ratio"]
                            }
                            for col, info in col_info.items()
                        ])
                        st.dataframe(col_info_df, use_container_width=True)

                        # 统计描述
                        st.markdown("##### 统计描述")
                        st.dataframe(df.describe(), use_container_width=True)
                    else:
                        st.info("👈 请先在左侧上传数据")

                # Tab 5: 模型详情
                with tabs[4]:
                    st.markdown("### 📋 模型详情")
                    if st.session_state.trained_models:
                        for model_name, model_data in st.session_state.trained_models.items():
                            with st.expander(f"📦 {model_name}",
                                             expanded=(model_name == st.session_state.current_model)):
                                st.markdown(f"**模型类型:** {type(model_data['model']).__name__}")

                                # 训练集指标
                                st.markdown("**训练集指标:**")
                                train_metrics = model_data['train_metrics']
                                metrics_text = format_metrics(train_metrics, st.session_state.problem_type)
                                st.text(metrics_text)

                                # 测试集指标
                                st.markdown("**测试集指标:**")
                                test_metrics = model_data['test_metrics']
                                metrics_text = format_metrics(test_metrics, st.session_state.problem_type)
                                st.text(metrics_text)

                                # 最佳参数
                                if model_data.get('best_params'):
                                    st.markdown("**最佳参数:**")
                                    st.json(model_data['best_params'])

                                # 特征重要性（如果可用）
                                model = model_data['model']
                                if hasattr(model, 'feature_importances_'):
                                    st.markdown("**特征重要性:**")
                                    importance_df = pd.DataFrame({
                                        '特征': st.session_state.feature_names,
                                        '重要性': model.feature_importances_
                                    }).sort_values('重要性', ascending=False)

                                    fig_imp = px.bar(
                                        importance_df.head(15),
                                        x='重要性',
                                        y='特征',
                                        orientation='h',
                                        title='Top 15 特征重要性'
                                    )
                                    fig_imp.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                                    st.plotly_chart(fig_imp, use_container_width=True)
                    else:
                        st.info("👈 请先在左侧训练模型")

                # Tab 6: 运行日志
                with tabs[5]:
                    st.markdown("### 📝 运行日志")
                    logs = st.session_state.logger.get_logs()
                    if logs:
                        # 添加颜色标记
                        log_html = ""
                        for line in logs.split('\n'):
                            if '[ERROR]' in line:
                                log_html += f'<p style="color: red;">{line}</p>'
                            elif '[WARNING]' in line:
                                log_html += f'<p style="color: orange;">{line}</p>'
                            elif '[SUCCESS]' in line:
                                log_html += f'<p style="color: green;">{line}</p>'
                            else:
                                log_html += f'<p style="color: black;">{line}</p>'

                        st.markdown(
                            f'<div style="height: 500px; overflow-y: scroll; background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace;">{log_html}</div>',
                            unsafe_allow_html=True)

                        if st.button("🗑️ 清空日志"):
                            st.session_state.logger.clear()
                            st.rerun()
                    else:
                        st.info("暂无日志记录")

                # Tab 7: 预测结果
                with tabs[6]:
                    st.markdown("### 🔮 预测结果")
                    if st.session_state.predictions is not None:
                        predictions_df = st.session_state.predictions

                        st.markdown(f"**预测样本数:** {len(predictions_df)}")
                        st.dataframe(predictions_df, use_container_width=True)

                        # 保存预测结果
                        if st.button("💾 保存预测结果"):
                            timestamp = get_file_timestamp()
                            save_path = os.path.join(RESULTS_DIR, f"predictions_{timestamp}.xlsx")
                            predictions_df.to_excel(save_path, index=False)
                            st.success(f"✅ 已保存: {save_path}")

                        # 预测结果统计
                        st.markdown("##### 预测结果统计")
                        if st.session_state.problem_type == "classification":
                            value_counts = predictions_df['预测结果'].value_counts()
                            fig_pred = px.pie(
                                values=value_counts.values,
                                names=value_counts.index,
                                title='预测类别分布'
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                        else:
                            fig_pred = px.histogram(
                                predictions_df,
                                x='预测结果',
                                title='预测值分布',
                                nbins=30
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                    else:
                        st.info("👈 请在左侧上传新数据并进行预测")


# ==================== 运行主程序 ====================
if __name__ == "__main__":
    main()