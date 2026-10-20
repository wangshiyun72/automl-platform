# app.py
"""
AutoML 平台主程序
基于 Streamlit 的自动化机器学习平台
"""

import os
import html
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, f1_score, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

from modules import (
    DataProcessor, ModelTrainer, SHAPAnalyzer, DataProfiler, ModelDiagnostics,
    build_standard_report, load_data, detect_problem_type, get_column_types,
    save_model, Logger
)
from modules.utils import make_arrow_safe, fig_to_bytes, df_to_excel_bytes, zip_paths
from config import RESULTS_DIR, ASSETS_DIR, get_file_timestamp

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="AutoML 自动化机器学习平台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = ["#4F46E5", "#06B6D4", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#EC4899", "#64748B"]
TRAIN_COLOR, TEST_COLOR = "#A5B4FC", "#4F46E5"

OPTIMIZATION_LABELS = {
    "none": "不优化（默认参数，最快）",
    "random": "随机搜索（速度与效果平衡）",
    "bayes": "贝叶斯搜索（智能高效）",
    "grid": "网格搜索（最全面，最慢）",
}
SCALING_LABELS = {
    "none": "不标准化（推荐树模型）",
    "zscore": "Z-score 标准化（推荐 SVM / KNN / 神经网络）",
    "minmax": "MinMax 归一化到 [0, 1]",
}
SHAP_PLOTS = {
    "summary": ("🐝 摘要图", "每个点是一个样本，横轴为 SHAP 值，颜色表示特征值高低，可同时看出重要性和影响方向。"),
    "bar": ("📊 重要性条形图", "按平均 |SHAP| 排序的全局特征重要性。"),
    "waterfall": ("🌊 瀑布图", "单个样本的预测是如何从基准值一步步被各特征推高或拉低的。"),
    "force": ("💪 力图", "单个样本中推动预测升高（红）和降低（蓝）的特征。"),
    "dependence": ("📈 依赖图", "某个特征的取值与其 SHAP 值的关系，可观察非线性效应。"),
    "heatmap": ("🔥 热力图", "样本 × 特征的 SHAP 值矩阵，揭示样本群体的解释模式。"),
}
SAMPLE_DATASETS = {
    "breast_cancer": "乳腺癌诊断 · 二分类",
    "iris": "鸢尾花品种 · 多分类",
    "wine": "葡萄酒产地 · 多分类",
    "diabetes": "糖尿病进展 · 回归",
}


# ==================== 样式 ====================
def inject_css():
    css_path = os.path.join(ASSETS_DIR, "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def esc(value):
    return html.escape(str(value))


def section_title(text, caption=None):
    st.markdown(f'<div class="section-title"><span class="bar"></span>{text}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def note(text, kind="info"):
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def empty_state(icon, title, desc):
    st.markdown(
        f'<div class="empty-state"><div class="icon">{icon}</div>'
        f'<div class="title">{title}</div><div class="desc">{desc}</div></div>',
        unsafe_allow_html=True,
    )


def card():
    """带白色背景的卡片容器（通过 card-marker 被 CSS 识别）"""
    container = st.container(border=True)
    container.markdown('<div class="card-marker"></div>', unsafe_allow_html=True)
    return container


def side_step(num, text, done=False):
    cls = "side-step done" if done else "side-step"
    mark = "✓" if done else num
    st.markdown(f'<div class="{cls}"><span class="num">{mark}</span>{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=380, title=None):
    if title:
        fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)))
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=56 if title or fig.layout.annotations else 30, b=10),
        font=dict(family="Inter, PingFang SC, Microsoft YaHei, sans-serif", size=13, color="#1F2937"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        colorway=PALETTE,
    )
    fig.update_xaxes(gridcolor="#EEF0F5", zerolinecolor="#E5E7EB")
    fig.update_yaxes(gridcolor="#EEF0F5", zerolinecolor="#E5E7EB")
    return fig


# ==================== Session State ====================
def init_session_state():
    defaults = {
        'data': None,
        'data_name': None,
        'data_sig': None,
        'data_version': 0,
        'trained': None,
        'shap_cache': None,
        'fig_cache': None,
        'predictions': None,
        'profiler_cache': None,
        'model_zip': None,
        'report_html': None,
        'logger': Logger(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log_message(message, level="INFO"):
    st.session_state.logger.log(message, level)


def set_new_data(df, name, sig):
    """载入新数据并清空所有下游结果"""
    st.session_state.data = df
    st.session_state.data_name = name
    st.session_state.data_sig = sig
    st.session_state.data_version += 1
    for key in ('trained', 'shap_cache', 'fig_cache', 'predictions', 'profiler_cache', 'model_zip', 'report_html'):
        st.session_state[key] = None
    log_message(f"数据加载成功: {name}, 维度: {df.shape}", "SUCCESS")


@st.cache_data(show_spinner=False)
def load_sample_dataset(key):
    from sklearn import datasets
    if key == "breast_cancer":
        d = datasets.load_breast_cancer(as_frame=True)
        df = d.frame.rename(columns={"target": "诊断结果"})
        df["诊断结果"] = df["诊断结果"].map({0: "恶性", 1: "良性"})
    elif key == "iris":
        d = datasets.load_iris(as_frame=True)
        df = d.frame.rename(columns={"target": "品种"})
        df["品种"] = df["品种"].map(dict(enumerate(d.target_names)))
    elif key == "wine":
        d = datasets.load_wine(as_frame=True)
        df = d.frame.rename(columns={"target": "产地"})
        df["产地"] = df["产地"].map({0: "产区A", 1: "产区B", 2: "产区C"})
    else:
        d = datasets.load_diabetes(as_frame=True, scaled=False)
        df = d.frame.rename(columns={"target": "病情进展指标"})
    return df


@st.cache_data(show_spinner=False)
def model_catalog(problem_type):
    """{model_key: (中文名, 类别)}"""
    models = ModelTrainer(problem_type=problem_type).get_available_models()
    return {key: (name, category) for key, (name, category, available) in models.items() if available}


def model_label(key, problem_type):
    name = model_catalog(problem_type).get(key, (key, ""))[0]
    return key if name == key else f"{name} · {key}"


# ==================== 侧边栏：配置 ====================
def render_sidebar():
    """渲染侧边栏配置，返回 (配置字典, 是否点击训练)"""
    cfg, train_clicked = None, False

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><div class="logo">🤖</div>'
            '<div><div class="name">AutoML Studio</div><div class="tag">深度集成SHAP可解释</div></div></div>',
            unsafe_allow_html=True,
        )

        # ---------- 1. 数据 ----------
        side_step(1, "加载数据", st.session_state.data is not None)
        source = st.radio("数据来源", ["上传文件", "示例数据"], horizontal=True, label_visibility="collapsed")

        if source == "上传文件":
            uploaded = st.file_uploader("上传 CSV / Excel 文件", type=['csv', 'xlsx', 'xls'],
                                        help="支持 CSV、XLSX、XLS；CSV 自动识别 UTF-8 / GBK 编码与分隔符")
            if uploaded is not None:
                sig = ("file", uploaded.name, uploaded.size)
                if sig != st.session_state.data_sig:
                    df, error = load_data(uploaded)
                    if error:
                        st.error(f"❌ {error}")
                        log_message(f"数据加载失败: {error}", "ERROR")
                    else:
                        set_new_data(df, uploaded.name, sig)
        else:
            sample_key = st.selectbox("选择示例数据集", list(SAMPLE_DATASETS),
                                      format_func=SAMPLE_DATASETS.get)
            if st.button("📥 加载示例数据", use_container_width=True):
                set_new_data(load_sample_dataset(sample_key).copy(), SAMPLE_DATASETS[sample_key],
                             ("sample", sample_key, st.session_state.data_version))

        df = st.session_state.data
        if df is None:
            st.caption("👆 请先上传数据或加载示例数据集")
            return None, False

        st.markdown(
            f'<span class="badge">📄 {esc(st.session_state.data_name)}</span>'
            f'<span class="badge gray">{df.shape[0]:,} 行 × {df.shape[1]} 列</span>',
            unsafe_allow_html=True,
        )

        v = st.session_state.data_version
        columns = df.columns.tolist()

        # ---------- 2. 目标与特征 ----------
        side_step(2, "目标与特征", True)
        target = st.selectbox("🎯 目标列（要预测的列）", columns, index=len(columns) - 1, key=f"target_{v}")

        target_series = df[target]
        auto_type = detect_problem_type(target_series)
        problem_type = st.radio(
            "问题类型",
            ["classification", "regression"],
            format_func=lambda x: "分类" if x == "classification" else "回归",
            index=0 if auto_type == "classification" else 1,
            horizontal=True,
            key=f"ptype_{v}_{target}",
            help=f"根据目标列自动识别为「{'分类' if auto_type == 'classification' else '回归'}」，可手动修改",
        )
        st.caption(f"唯一值 {target_series.nunique()} 个 · 缺失 {int(target_series.isna().sum())} 个 · "
                   f"类型 {target_series.dtype}")

        available = [c for c in columns if c != target]
        fkey = f"features_{v}"
        if fkey not in st.session_state:
            st.session_state[fkey] = available
        else:
            st.session_state[fkey] = [c for c in st.session_state[fkey] if c in available]

        c1, c2 = st.columns(2)
        c1.button("全选", key="sel_all", use_container_width=True,
                  on_click=lambda: st.session_state.update({fkey: available}))
        c2.button("清空", key="sel_none", use_container_width=True,
                  on_click=lambda: st.session_state.update({fkey: []}))
        features = st.multiselect("特征列", available, key=fkey, placeholder="选择用于训练的特征")
        st.caption(f"已选择 {len(features)} / {len(available)} 个特征")

        # ---------- 3. 预处理 ----------
        side_step(3, "数据预处理", True)
        train_ratio = st.slider("训练集比例", 0.5, 0.9, 0.7, 0.05, help="剩余部分作为测试集")
        scaling = st.selectbox("标准化方法", list(SCALING_LABELS), format_func=SCALING_LABELS.get)

        # ---------- 4. 算法 ----------
        side_step(4, "选择算法", True)
        catalog = model_catalog(problem_type)
        categories = {}
        for key, (name, category) in catalog.items():
            categories.setdefault(category, []).append((key, name))

        models = []
        for category in sorted(categories, key=lambda c: (c != "⭐ 推荐算法", c)):
            with st.expander(category, expanded=(category == "⭐ 推荐算法")):
                cols = st.columns(2)
                for i, (key, name) in enumerate(categories[category]):
                    with cols[i % 2]:
                        if st.checkbox(name, value=(category == "⭐ 推荐算法"), key=f"model_{problem_type}_{key}"):
                            models.append(key)
        st.caption(f"已选择 {len(models)} 个算法")

        # ---------- 5. 调参 ----------
        side_step(5, "超参数优化", True)
        optimization = st.selectbox("优化方法", list(OPTIMIZATION_LABELS), format_func=OPTIMIZATION_LABELS.get)
        n_iter, cv = 20, 5
        if optimization != "none":
            if optimization in ("random", "bayes"):
                n_iter = st.slider("搜索迭代次数", 5, 100, 20, 5)
            cv = st.slider("交叉验证折数", 2, 10, 5)

        cfg = {
            "target": target,
            "features": list(features),
            "problem_type": problem_type,
            "train_ratio": train_ratio,
            "scaling": scaling,
            "models": models,
            "optimization": optimization,
            "n_iter": n_iter,
            "cv": cv,
        }

        # ---------- 校验 ----------
        blockers = validate_config(df, cfg)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        for msg in blockers:
            st.warning(msg, icon="⚠️")
        train_clicked = st.button("🚀 开始训练", type="primary", use_container_width=True, disabled=bool(blockers))

    return cfg, train_clicked


def validate_config(df, cfg):
    blockers = []
    if not cfg["features"]:
        blockers.append("请至少选择一个特征列")
    if not cfg["models"]:
        blockers.append("请至少选择一个算法")

    y = df[cfg["target"]].dropna()
    if len(y) < 10:
        blockers.append("目标列有效样本少于 10 行")
    elif cfg["problem_type"] == "regression":
        if pd.to_numeric(y, errors="coerce").isna().mean() > 0.05:
            blockers.append("目标列不是数值型，无法做回归，请切换为「分类」")
    else:
        n_classes = y.nunique()
        if n_classes < 2:
            blockers.append("目标列只有一个取值，无法分类")
        elif n_classes > 100 or (n_classes > 30 and n_classes / len(y) > 0.3):
            blockers.append(f"目标列有 {n_classes} 个不同取值，更适合「回归」")
    return blockers


# ==================== 训练 ====================
def run_training(cfg):
    df = st.session_state.data
    problem_type = cfg["problem_type"]
    st.session_state.shap_cache = None
    st.session_state.fig_cache = None
    st.session_state.model_zip = None
    st.session_state.report_html = None

    with st.status("🚀 正在训练模型...", expanded=True) as status:
        try:
            log_message("开始数据预处理...")
            processor = DataProcessor()
            X, y, processing_log = processor.fit_transform(
                df, cfg["target"], cfg["features"], cfg["scaling"], problem_type
            )
            for line in processing_log:
                log_message(line)
            st.write(f"✅ 数据预处理完成：{X.shape[0]:,} 个样本，{X.shape[1]} 个特征")

            stratify = None
            if problem_type == "classification":
                counts = np.bincount(y.astype(int))
                n_test = int(round(len(y) * (1 - cfg["train_ratio"])))
                if counts.min() >= 2 and n_test >= len(counts):
                    stratify = y
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=cfg["train_ratio"], random_state=42, stratify=stratify
            )
            log_message(f"数据划分完成: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本"
                        + ("（分层抽样）" if stratify is not None else ""))
            st.write(f"✅ 数据划分完成：训练集 {len(X_train):,}，测试集 {len(X_test):,}"
                     + ("（分层抽样）" if stratify is not None else ""))
        except Exception as e:
            log_message(f"数据预处理失败: {e}", "ERROR")
            status.update(label=f"❌ 数据预处理失败：{e}", state="error", expanded=True)
            return

        trainer = ModelTrainer(problem_type=problem_type)
        trained_models, failed = {}, {}
        progress = st.progress(0.0)

        for i, name in enumerate(cfg["models"]):
            label = model_label(name, problem_type)
            progress.progress(i / len(cfg["models"]), text=f"正在训练 {label}（{i + 1}/{len(cfg['models'])}）")
            log_message(f"开始训练模型: {name}")
            try:
                model, train_metrics, test_metrics, best_params = trainer.train_model(
                    name, X_train, y_train, X_test, y_test,
                    optimization=cfg["optimization"], n_iter=cfg["n_iter"], cv=cfg["cv"]
                )
                result = trainer.results[name]
                trained_models[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'best_params': best_params,
                    'train_time': result["train_time"],
                    'notes': result["notes"],
                }
                for n in result["notes"]:
                    log_message(f"{name}: {n}", "WARNING")
                if problem_type == "classification":
                    score_text = f"测试集准确率 {test_metrics['accuracy']:.4f}"
                else:
                    score_text = f"测试集 R² {test_metrics['r2']:.4f}"
                log_message(f"{name} 训练完成 - {score_text}", "SUCCESS")
                st.write(f"✅ **{label}** — {score_text} · 用时 {result['train_time']:.1f}s")
            except Exception as e:
                failed[name] = str(e)
                log_message(f"{name} 训练失败: {e}", "ERROR")
                st.write(f"❌ **{label}** 训练失败：{e}")

        progress.progress(1.0, text="训练结束")

        if trained_models:
            status.update(label=f"✅ 训练完成：成功 {len(trained_models)} 个"
                                + (f"，失败 {len(failed)} 个" if failed else ""),
                          state="complete", expanded=bool(failed))
        else:
            status.update(label="❌ 所有模型均训练失败，请查看日志", state="error", expanded=True)

    if not trained_models:
        return

    class_names = None
    if problem_type == "classification":
        class_names = [str(c) for c in processor.get_target_classes()]

    st.session_state.trained = {
        'id': get_file_timestamp(),
        'config': cfg,
        'processor': processor,
        'trainer': trainer,
        'models': trained_models,
        'failed': failed,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(cfg["features"]),
        'target': cfg["target"],
        'problem_type': problem_type,
        'class_names': class_names,
    }
    st.toast(f"成功训练 {len(trained_models)} 个模型", icon="🎉")


def primary_metric(problem_type):
    return ("accuracy", "准确率") if problem_type == "classification" else ("r2", "R²")


def best_model_name(trained):
    key, _ = primary_metric(trained['problem_type'])
    return max(trained['models'], key=lambda m: trained['models'][m]['test_metrics'][key])


# ==================== 主区域：头部 ====================
def render_header(cfg):
    st.markdown(
        '<div class="hero"><h1>🤖 AutoML 自动化机器学习平台</h1>'
        '<p>上传数据 → 选择目标 → 一键训练多种算法，并用 SHAP 理解模型为什么这样预测</p>'
        '<div class="chips"><span class="chip">15+ 种算法</span><span class="chip">网格 / 随机 / 贝叶斯调参</span>'
        '<span class="chip">SHAP 可解释性</span><span class="chip">批量预测与导出</span></div></div>',
        unsafe_allow_html=True,
    )

    trained = st.session_state.trained
    steps = [
        ("加载数据", st.session_state.data is not None),
        ("配置特征", bool(cfg and cfg["features"])),
        ("训练模型", trained is not None),
        ("SHAP 解释", st.session_state.shap_cache is not None),
        ("新数据预测", st.session_state.predictions is not None),
    ]
    active_found = False
    parts = []
    for i, (label, done) in enumerate(steps, 1):
        cls = "done" if done else ""
        if not done and not active_found:
            cls, active_found = "active", True
        mark = "✓" if done else str(i)
        parts.append(f'<div class="step {cls}"><span class="dot">{mark}</span>{label}</div>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_kpis(cfg):
    df = st.session_state.data
    trained = st.session_state.trained

    cards = [
        ("📄 样本量", f"{df.shape[0]:,}", f"{df.shape[1]} 列 · 缺失 {int(df.isna().sum().sum()):,}", ""),
        ("🧩 特征数", f"{len(cfg['features'])}", f"目标列：{esc(cfg['target'])}", ""),
    ]
    if trained:
        n_ok, n_fail = len(trained['models']), len(trained['failed'])
        cards.append(("🧠 已训练模型", f"{n_ok}", f"失败 {n_fail} 个" if n_fail else "全部训练成功", ""))
        best = best_model_name(trained)
        key, label = primary_metric(trained['problem_type'])
        score = trained['models'][best]['test_metrics'][key]
        cards.append(("🏆 最佳模型", esc(best), f"测试集{label} {score:.4f}", "highlight"))
    else:
        task = "分类" if cfg["problem_type"] == "classification" else "回归"
        cards.append(("🧠 任务类型", task, f"已选 {len(cfg['models'])} 个算法", ""))
        cards.append(("🏆 最佳模型", "—", "训练后显示", ""))

    html_cards = "".join(
        f'<div class="kpi {extra}"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
        for label, value, sub, extra in cards
    )
    st.markdown(f'<div class="kpi-grid">{html_cards}</div>', unsafe_allow_html=True)

    if trained:
        old = trained['config']
        changed = [name for key, name in (("target", "目标列"), ("features", "特征"), ("problem_type", "问题类型"),
                                          ("scaling", "标准化"), ("train_ratio", "训练集比例"))
                   if old[key] != cfg[key]]
        if changed:
            note(f"⚠️ 左侧配置（{'、'.join(changed)}）已修改，当前展示的仍是上一次训练的结果，点击「开始训练」以更新。",
                 "warning")


def render_welcome():
    st.markdown(
        '<div class="feature-grid">'
        '<div class="feature-card"><div class="icon">📁</div><div class="title">1. 加载数据</div>'
        '<div class="desc">在左侧上传 CSV / Excel 文件，或直接加载内置示例数据集体验完整流程。</div></div>'
        '<div class="feature-card"><div class="icon">⚙️</div><div class="title">2. 配置并训练</div>'
        '<div class="desc">选择目标列与特征，勾选算法与调参方式，一键训练并自动对比多个模型。</div></div>'
        '<div class="feature-card"><div class="icon">🔍</div><div class="title">3. 解释与预测</div>'
        '<div class="desc">用 SHAP 图理解模型决策，上传新数据批量预测，结果与模型均可下载。</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    empty_state("👈", "从左侧开始", "支持缺失值自动填充、分类变量自动编码、日期列自动转换")


# ==================== Tab: 数据概览 ====================
def get_profiler(df, target):
    key = (st.session_state.data_version, target)
    cache = st.session_state.profiler_cache
    if cache and cache[0] == key:
        return cache[1]
    try:
        profiler = DataProfiler().profile(df, target_column=target)
    except Exception as e:
        log_message(f"数据画像失败: {e}", "WARNING")
        profiler = None
    st.session_state.profiler_cache = (key, profiler)
    return profiler


def render_data_tab(cfg):
    df = st.session_state.data
    target = cfg["target"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("行数", f"{df.shape[0]:,}")
    c2.metric("列数", df.shape[1])
    c3.metric("缺失值", f"{int(df.isna().sum().sum()):,}")
    c4.metric("重复行", f"{int(df.duplicated().sum()):,}")

    section_title("数据预览", "展示前 200 行")
    st.dataframe(make_arrow_safe(df.head(200)), use_container_width=True, height=320)

    left, right = st.columns([3, 2])
    with left:
        section_title("列信息")
        col_info = get_column_types(df)
        info_df = pd.DataFrame([
            {"列名": col, "类型": info["type"], "数据类型": info["dtype"], "唯一值": info["unique_count"],
             "缺失数": info["null_count"], "缺失比例": info["null_count"] / max(len(df), 1)}
            for col, info in col_info.items()
        ])
        st.dataframe(
            info_df, use_container_width=True, hide_index=True, height=320,
            column_config={"缺失比例": st.column_config.ProgressColumn("缺失比例", format="%.2f", min_value=0,
                                                                      max_value=1)},
        )
    with right:
        section_title(f"目标列分布 · {esc(target)}")
        y = df[target].dropna()
        if cfg["problem_type"] == "classification":
            counts = y.astype(str).value_counts().head(30)
            fig = px.bar(x=counts.index, y=counts.values, labels={"x": target, "y": "样本数"},
                         color_discrete_sequence=[PALETTE[0]])
        else:
            fig = px.histogram(pd.to_numeric(y, errors="coerce").dropna(), nbins=40,
                               color_discrete_sequence=[PALETTE[0]])
            fig.update_layout(showlegend=False, xaxis_title=target, yaxis_title="样本数")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    profiler = get_profiler(df, target)
    if profiler is not None:
        section_title("数据质量检查", "自动检测缺失、高基数、强相关、潜在泄露等问题")
        issues = profiler.get_quality_issues()
        if issues.empty:
            note("✅ 未发现明显的数据质量问题", "success")
        else:
            st.dataframe(issues, use_container_width=True, hide_index=True)
        leakage = profiler.get_leakage_candidates()
        if not leakage.empty:
            note("🚨 以下列与目标列高度相关或名称可疑，可能存在数据泄露，建议从特征中移除：", "danger")
            st.dataframe(leakage, use_container_width=True, hide_index=True)
        for rec in profiler.get_recommendations():
            note(f"💡 {esc(rec)}", "info")

    with st.expander("📐 数值统计描述"):
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            st.caption("没有数值列")
        else:
            st.dataframe(numeric.describe().T, use_container_width=True)


# ==================== Tab: 模型对比 ====================
def render_comparison_tab():
    trained = st.session_state.trained
    if not trained:
        empty_state("📈", "还没有训练结果", "在左侧完成配置后点击「开始训练」")
        return

    problem_type = trained['problem_type']
    comparison_df = trained['trainer'].get_comparison_df()
    comparison_df = comparison_df[comparison_df["模型"].isin(trained['models'])]
    key_col = "测试集准确率" if problem_type == "classification" else "测试集R²"
    comparison_df = comparison_df.sort_values(key_col, ascending=False).reset_index(drop=True)
    comparison_df.insert(0, "排名", ["🥇", "🥈", "🥉"][:len(comparison_df)] +
                         [str(i) for i in range(4, len(comparison_df) + 1)])

    section_title("模型排行榜", f"按{key_col}从高到低排序")
    column_config = {}
    for col in comparison_df.columns:
        if col in ("排名", "模型"):
            continue
        if problem_type == "classification" and col not in ("训练耗时(秒)",):
            column_config[col] = st.column_config.ProgressColumn(col, format="%.4f", min_value=0, max_value=1)
        else:
            column_config[col] = st.column_config.NumberColumn(col, format="%.4f")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True, column_config=column_config)

    if trained['failed']:
        with st.expander(f"❌ {len(trained['failed'])} 个模型训练失败"):
            for name, err in trained['failed'].items():
                st.markdown(f"**{name}**：`{err}`")

    if problem_type == "classification":
        pairs = [("准确率", "训练集准确率", "测试集准确率"), ("F1 分数", "训练集F1", "测试集F1")]
    else:
        pairs = [("R²", "训练集R²", "测试集R²"), ("RMSE（越低越好）", "训练集RMSE", "测试集RMSE")]

    fig = make_subplots(rows=1, cols=2, subplot_titles=[p[0] for p in pairs], horizontal_spacing=0.08)
    for col_idx, (_, train_col, test_col) in enumerate(pairs, 1):
        fig.add_trace(go.Bar(name="训练集", x=comparison_df["模型"], y=comparison_df[train_col],
                             marker_color=TRAIN_COLOR, showlegend=col_idx == 1), row=1, col=col_idx)
        fig.add_trace(go.Bar(name="测试集", x=comparison_df["模型"], y=comparison_df[test_col],
                             marker_color=TEST_COLOR, showlegend=col_idx == 1), row=1, col=col_idx)
    fig.update_layout(barmode="group", bargap=0.25)
    section_title("训练集 vs 测试集")
    st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    section_title("过拟合分析", "训练集与测试集核心指标的差距越大，模型越可能过拟合")
    train_col, test_col = pairs[0][1], pairs[0][2]
    overfit = comparison_df[["模型", train_col, test_col]].copy()
    overfit["差距"] = overfit[train_col] - overfit[test_col]
    overfit["评估"] = overfit["差距"].apply(
        lambda g: "🔴 较高" if g > 0.1 else ("🟡 中等" if g > 0.05 else "🟢 良好"))
    st.dataframe(overfit, use_container_width=True, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(c, format="%.4f")
                                for c in (train_col, test_col, "差距")})

    st.download_button(
        "⬇️ 下载对比结果 (CSV)",
        comparison_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"model_comparison_{trained['id']}.csv",
        mime="text/csv",
    )


# ==================== Tab: 模型评估 ====================
def model_selector(trained, key, label="选择模型"):
    names = list(trained['models'])
    best = best_model_name(trained)
    return st.selectbox(
        label, names, index=names.index(best), key=f"{key}_{trained['id']}",
        format_func=lambda m: ("🏆 " if m == best else "") + model_label(m, trained['problem_type']),
    )


def model_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.atleast_2d(model.coef_)).mean(axis=0)
    else:
        return None
    if len(values) != len(feature_names):
        return None
    return pd.DataFrame({"特征": feature_names, "重要性": values}).sort_values("重要性", ascending=False)


def render_evaluation_tab():
    trained = st.session_state.trained
    if not trained:
        empty_state("📋", "还没有训练结果", "训练完成后可在这里查看每个模型的详细评估")
        return

    problem_type = trained['problem_type']
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        name = model_selector(trained, "eval_model")
    data = trained['models'][name]
    model, train_m, test_m = data['model'], data['train_metrics'], data['test_metrics']
    y_test = trained['y_test']
    y_pred = np.asarray(test_m['predictions'])

    for n in data.get('notes', []):
        note(f"ℹ️ {esc(n)}", "warning")

    if problem_type == "classification":
        items = [("准确率", "accuracy"), ("精确率", "precision"), ("召回率", "recall"), ("F1 分数", "f1")]
        if "auc" in test_m:
            items.append(("AUC", "auc"))
    else:
        items = [("R²", "r2"), ("RMSE", "rmse"), ("MAE", "mae"), ("MSE", "mse")]

    section_title("测试集指标", "下方小字为与训练集的差值")
    cols = st.columns(len(items))
    for col, (label, key) in zip(cols, items):
        col.metric(label, f"{test_m[key]:.4f}", delta=f"{test_m[key] - train_m[key]:+.4f} vs 训练集",
                   delta_color="off")

    y_proba = None
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(trained['X_test'])
        except Exception:
            y_proba = None

    if problem_type == "classification":
        class_names = trained['class_names']
        labels = list(range(len(class_names)))
        left, right = st.columns(2)
        with left:
            section_title("混淆矩阵")
            normalize = st.toggle("按真实类别归一化", key="cm_norm")
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            z = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
            fig = px.imshow(z, x=class_names, y=class_names, text_auto=".2f" if normalize else "d",
                            color_continuous_scale=["#F5F7FF", "#4F46E5"], aspect="auto",
                            labels=dict(x="预测类别", y="真实类别", color="比例" if normalize else "数量"))
            fig.update_xaxes(type="category")
            fig.update_yaxes(type="category")
            st.plotly_chart(style_fig(fig, 400), use_container_width=True)
        with right:
            if y_proba is not None and len(class_names) == 2:
                section_title(f"ROC 曲线 · 正类「{esc(class_names[1])}」")
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", fill="tozeroy",
                                         fillcolor="rgba(79,70,229,0.12)", line=dict(color=PALETTE[0], width=2.5),
                                         name=f"AUC = {test_m.get('auc', float('nan')):.4f}"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="随机猜测",
                                         line=dict(dash="dash", color="#9CA3AF")))
                fig.update_layout(xaxis_title="假阳性率 (FPR)", yaxis_title="真阳性率 (TPR)")
                st.plotly_chart(style_fig(fig, 400), use_container_width=True)
            else:
                section_title("各类别指标")
                diag = ModelDiagnostics("classification").evaluate(
                    np.asarray(class_names)[y_test.astype(int)], np.asarray(class_names)[y_pred.astype(int)])
                st.dataframe(diag.tables["分类别指标"], use_container_width=True, hide_index=True)

        if len(class_names) == 2:
            with st.expander("📑 各类别指标"):
                diag = ModelDiagnostics("classification").evaluate(
                    np.asarray(class_names)[y_test.astype(int)], np.asarray(class_names)[y_pred.astype(int)])
                st.dataframe(diag.tables["分类别指标"], use_container_width=True, hide_index=True)
    else:
        section_title("残差分析")
        residuals = y_test - y_pred
        fig = make_subplots(rows=1, cols=2, subplot_titles=("预测值 vs 真实值", "残差分布"),
                            horizontal_spacing=0.08)
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name="样本",
                                 marker=dict(color=PALETTE[0], opacity=0.55, size=7)), row=1, col=1)
        lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="理想线",
                                 line=dict(dash="dash", color=PALETTE[4])), row=1, col=1)
        fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name="残差", marker_color=PALETTE[1]), row=1, col=2)
        fig.update_xaxes(title_text="真实值", row=1, col=1)
        fig.update_yaxes(title_text="预测值", row=1, col=1)
        fig.update_xaxes(title_text="残差", row=1, col=2)
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        importance = model_feature_importance(model, trained['feature_names'])
        if importance is not None:
            section_title("模型内置特征重要性", "Top 15，来自 feature_importances_ 或系数绝对值")
            top = importance.head(15)
            fig = px.bar(top, x="重要性", y="特征", orientation="h", color="重要性",
                         color_continuous_scale=["#C7D2FE", "#4F46E5"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, max(300, 26 * len(top) + 80)), use_container_width=True)
    with right:
        section_title("模型参数")
        st.markdown(f'<span class="badge">{esc(type(model).__name__)}</span>'
                    f'<span class="badge gray">训练用时 {data.get("train_time", 0):.2f}s</span>',
                    unsafe_allow_html=True)
        if data.get('best_params'):
            st.caption("超参数搜索得到的最佳参数：")
            st.json(data['best_params'])
        else:
            st.caption("使用默认参数训练")

        section_title("导出模型")
        zip_info = st.session_state.model_zip
        if st.button("📦 打包模型文件", use_container_width=True, key="pack_model"):
            try:
                save_dir = os.path.join(RESULTS_DIR, f"{name}_{get_file_timestamp()}")
                paths = save_model(
                    model=model, model_name=name, save_dir=save_dir,
                    scaler=trained['processor'].scaler,
                    label_encoders=trained['processor'].label_encoders,
                    feature_names=trained['feature_names'],
                    model_params=data.get('best_params'),
                    data_processor=trained['processor'],
                )
                st.session_state.model_zip = (trained['id'], name, zip_paths(paths.values()))
                zip_info = st.session_state.model_zip
                log_message(f"模型已保存到: {save_dir}", "SUCCESS")
            except Exception as e:
                st.error(f"❌ 打包失败: {e}")
                log_message(f"模型打包失败: {e}", "ERROR")
        if zip_info and zip_info[0] == trained['id'] and zip_info[1] == name:
            st.download_button("⬇️ 下载模型包 (.zip)", zip_info[2], file_name=f"{name}_{trained['id']}.zip",
                               mime="application/zip", use_container_width=True, type="primary")
            st.caption("包含模型 (.pkl/.joblib)、数据处理器、配置文件，同时已保存到 shap_results 目录")


# ==================== Tab: SHAP ====================
def render_shap_tab():
    trained = st.session_state.trained
    if not trained:
        empty_state("🧠", "还没有训练结果", "训练完成后可在这里用 SHAP 解释模型")
        return

    problem_type = trained['problem_type']
    class_names = trained['class_names']

    with card():
        c1, c2, c3 = st.columns([1.4, 1, 1])
        with c1:
            name = model_selector(trained, "shap_model")
        with c2:
            plot_type = st.selectbox("图表类型", list(SHAP_PLOTS), format_func=lambda k: SHAP_PLOTS[k][0],
                                     key="shap_plot_type")
        with c3:
            ratio = st.slider("采样比例（测试集）", 0.05, 1.0, 0.3, 0.05, key="shap_ratio",
                              help="比例越大越准确，但计算越慢；非树模型最多使用 150 个样本")

        class_idx = None
        c4, c5, c6 = st.columns([1.4, 1, 1])
        if problem_type == "classification" and class_names:
            with c4:
                class_idx = st.selectbox(
                    "解释的类别", range(len(class_names)), index=1 if len(class_names) == 2 else 0,
                    format_func=lambda i: class_names[i], key=f"shap_class_{trained['id']}",
                    help="SHAP 值表示各特征把预测推向该类别的程度",
                )

        cache_key = (trained['id'], name, ratio)
        cache = st.session_state.shap_cache
        analyzer = cache["analyzer"] if cache and cache["key"] == cache_key else None

        sample_idx, dep_feature = 0, None
        if plot_type in ("waterfall", "force"):
            with c5:
                max_idx = len(analyzer.X_sample) - 1 if analyzer is not None else 0
                sample_idx = st.number_input("样本序号", 0, max(max_idx, 0), 0, key="shap_sample_idx",
                                             disabled=analyzer is None)
        if plot_type == "dependence":
            with c5:
                dep_feature = st.selectbox("特征", trained['feature_names'], key="shap_dep_feature")

        with c6:
            st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
            compute = st.button("⚡ 计算 SHAP" if analyzer is None else "🔄 重新计算",
                                type="primary" if analyzer is None else "secondary",
                                use_container_width=True, key="shap_compute")

    st.caption(f"ℹ️ {SHAP_PLOTS[plot_type][1]}")

    if compute:
        with st.spinner("正在计算 SHAP 值，非树模型可能需要较长时间..."):
            try:
                analyzer = SHAPAnalyzer(
                    model=trained['models'][name]['model'],
                    X=trained['X_test'],
                    feature_names=trained['feature_names'],
                    problem_type=problem_type,
                    background=trained['X_train'],
                )
                log_message(f"开始计算 SHAP 值: {name}")
                analyzer.compute_shap_values(sample_ratio=ratio)
                st.session_state.shap_cache = {"key": cache_key, "analyzer": analyzer}
                st.session_state.fig_cache = None
                log_message(f"SHAP 值计算完成（{analyzer.explainer_type}，{len(analyzer.X_sample)} 个样本）",
                            "SUCCESS")
                st.rerun()
            except Exception as e:
                analyzer = None
                log_message(f"SHAP 分析失败: {e}", "ERROR")
                st.error(f"❌ SHAP 分析失败: {e}")

    if analyzer is None:
        empty_state("⚡", "点击「计算 SHAP」开始分析", "计算完成后切换图表类型无需重新计算")
        return

    if class_idx is not None and analyzer.n_outputs == 1 and class_names and len(class_names) == 2:
        note(f"该模型的 SHAP 输出为正类「{esc(class_names[1])}」的对数几率（log-odds）", "info")

    fig_key = (cache_key, plot_type, class_idx, int(sample_idx), dep_feature)
    fig_cache = st.session_state.fig_cache
    if not fig_cache or fig_cache["key"] != fig_key:
        try:
            with st.spinner("正在绘图..."):
                if plot_type == "summary":
                    fig = analyzer.plot_summary(class_idx=class_idx)
                elif plot_type == "bar":
                    fig = analyzer.plot_bar(class_idx=class_idx)
                elif plot_type == "waterfall":
                    fig = analyzer.plot_waterfall(sample_idx=int(sample_idx), class_idx=class_idx)
                elif plot_type == "force":
                    fig = analyzer.plot_force(sample_idx=int(sample_idx), class_idx=class_idx)
                elif plot_type == "dependence":
                    fig = analyzer.plot_dependence(dep_feature, class_idx=class_idx)
                else:
                    fig = analyzer.plot_heatmap(class_idx=class_idx)
                fig_cache = {"key": fig_key, "fig": fig, "png": fig_to_bytes(fig, "png"),
                             "svg": fig_to_bytes(fig, "svg"), "pdf": fig_to_bytes(fig, "pdf")}
                plt.close(fig)
                st.session_state.fig_cache = fig_cache
        except Exception as e:
            log_message(f"SHAP 绘图失败: {e}", "ERROR")
            st.error(f"❌ 绘图失败: {e}")
            return

    st.markdown(
        f'<span class="badge">{esc(analyzer.explainer_type)}</span>'
        f'<span class="badge gray">{len(analyzer.X_sample)} 个样本</span>'
        + (f'<span class="badge green">类别：{esc(class_names[class_idx])}</span>' if class_idx is not None else ""),
        unsafe_allow_html=True,
    )
    with card():
        st.image(fig_cache["png"], use_column_width=True)

    stem = f"shap_{plot_type}_{name}_{get_file_timestamp()}"
    d1, d2, d3, _ = st.columns([1, 1, 1, 2])
    d1.download_button("⬇️ PNG (300 DPI)", fig_cache["png"], f"{stem}.png", "image/png", use_container_width=True)
    d2.download_button("⬇️ SVG 矢量图", fig_cache["svg"], f"{stem}.svg", "image/svg+xml", use_container_width=True)
    d3.download_button("⬇️ PDF 矢量图", fig_cache["pdf"], f"{stem}.pdf", "application/pdf",
                       use_container_width=True)

    section_title("SHAP 特征重要性", "平均 |SHAP| 值")
    left, right = st.columns([3, 2])
    importance = analyzer.get_feature_importance(class_idx)
    with left:
        top = importance.head(20)
        fig = px.bar(top, x="重要性", y="特征", orientation="h", color="重要性",
                     color_continuous_scale=["#A5F3FC", "#4F46E5"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, max(300, 24 * len(top) + 80)), use_container_width=True)
    with right:
        st.dataframe(importance, use_container_width=True, hide_index=True, height=360,
                     column_config={"重要性": st.column_config.NumberColumn(format="%.5f")})
        export_df = analyzer.export_shap_values(class_idx)
        st.download_button("⬇️ 导出 SHAP 值 (CSV)", export_df.to_csv(index=False).encode("utf-8-sig"),
                           f"shap_values_{name}.csv", "text/csv", use_container_width=True)


# ==================== Tab: 预测 ====================
def render_prediction_tab():
    trained = st.session_state.trained
    if not trained:
        empty_state("🔮", "还没有可用模型", "训练完成后可上传新数据进行批量预测")
        return

    processor = trained['processor']
    problem_type = trained['problem_type']

    with card():
        c1, c2 = st.columns([3, 2])
        with c1:
            predict_file = st.file_uploader("上传待预测数据（需包含训练时使用的特征列）",
                                            type=['csv', 'xlsx', 'xls'], key="predict_file")
        with c2:
            name = model_selector(trained, "predict_model", "预测使用的模型")
            template = st.session_state.data[trained['feature_names']].head(5)
            st.download_button("📄 下载输入模板", template.to_csv(index=False).encode("utf-8-sig"),
                               "prediction_template.csv", "text/csv", use_container_width=True)
        go_predict = st.button("🔮 开始预测", type="primary", disabled=predict_file is None, key="do_predict")

    if go_predict and predict_file is not None:
        with st.spinner("正在预测..."):
            try:
                new_df, error = load_data(predict_file)
                if error:
                    raise ValueError(error)
                X_new = processor.transform(new_df)
                model = trained['models'][name]['model']
                raw_pred = np.asarray(model.predict(X_new)).ravel()

                result_df = new_df.copy()
                if problem_type == "classification":
                    result_df["预测结果"] = processor.inverse_transform_target(raw_pred)
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_new)
                        classes = getattr(model, "classes_", np.arange(proba.shape[1]))
                        for i, cls in enumerate(classes):
                            result_df[f"概率_{trained['class_names'][int(cls)]}"] = np.round(proba[:, i], 4)
                else:
                    result_df["预测结果"] = raw_pred

                evaluation = None
                target = trained['target']
                if target in new_df.columns and new_df[target].notna().any():
                    mask = new_df[target].notna().values
                    if problem_type == "classification":
                        y_true = processor._target_as_labels(new_df.loc[mask, target]).values
                        y_hat = result_df.loc[mask, "预测结果"].astype(str).values
                        evaluation = {"准确率": accuracy_score(y_true, y_hat),
                                      "F1 (加权)": f1_score(y_true, y_hat, average="weighted", zero_division=0)}
                    else:
                        y_true = pd.to_numeric(new_df.loc[mask, target], errors="coerce").values
                        ok = ~np.isnan(y_true)
                        evaluation = {"R²": r2_score(y_true[ok], raw_pred[mask][ok]),
                                      "MAE": mean_absolute_error(y_true[ok], raw_pred[mask][ok])}

                st.session_state.predictions = {
                    "df": result_df, "model": name, "file": predict_file.name,
                    "unseen": processor.get_unseen_categories(new_df), "evaluation": evaluation,
                }
                log_message(f"预测完成: {predict_file.name}, {len(result_df)} 条, 模型 {name}", "SUCCESS")
                st.rerun()  # 刷新顶部步骤条
            except Exception as e:
                st.error(f"❌ 预测失败: {e}")
                log_message(f"预测失败: {e}", "ERROR")

    pred = st.session_state.predictions
    if not pred:
        empty_state("📤", "上传数据并点击「开始预测」", "可先下载输入模板，按模板格式准备数据")
        return

    result_df = pred["df"]
    if pred["unseen"]:
        detail = "；".join(f"{esc(col)}: {esc(', '.join(vals))}" for col, vals in pred["unseen"].items())
        note(f"⚠️ 以下类别在训练数据中未出现过，预测可能不可靠：{detail}", "warning")

    c1, c2, c3 = st.columns(3)
    c1.metric("预测样本数", f"{len(result_df):,}")
    c2.metric("使用模型", pred["model"])
    c3.metric("数据文件", pred["file"])
    if pred["evaluation"]:
        note("✅ 新数据中包含目标列，已自动计算预测效果：" +
             "，".join(f"{k} <b>{v:.4f}</b>" for k, v in pred["evaluation"].items()), "success")

    section_title("预测结果")
    st.dataframe(make_arrow_safe(result_df), use_container_width=True, height=360)

    d1, d2, _ = st.columns([1, 1, 3])
    stem = f"predictions_{get_file_timestamp()}"
    d1.download_button("⬇️ 下载 CSV", result_df.to_csv(index=False).encode("utf-8-sig"), f"{stem}.csv",
                       "text/csv", use_container_width=True)
    d2.download_button("⬇️ 下载 Excel", df_to_excel_bytes(result_df, "预测结果"), f"{stem}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

    section_title("预测结果分布")
    if problem_type == "classification":
        counts = result_df["预测结果"].astype(str).value_counts()
        fig = px.pie(values=counts.values, names=counts.index, hole=0.55, color_discrete_sequence=PALETTE)
        fig.update_traces(textinfo="percent+label")
    else:
        fig = px.histogram(result_df, x="预测结果", nbins=30, color_discrete_sequence=[PALETTE[0]])
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)


# ==================== Tab: 日志与报告 ====================
def render_logs_tab(cfg):
    left, right = st.columns([3, 2])
    with left:
        section_title("运行日志")
        entries = st.session_state.logger.get_entries()
        if entries:
            rows = "".join(
                f'<div class="row"><span class="ts">{esc(ts)}</span>'
                f'<span class="lv {esc(level)}">[{esc(level)}]</span>{esc(msg)}</div>'
                for ts, level, msg in reversed(entries)
            )
            st.markdown(f'<div class="log-box">{rows}</div>', unsafe_allow_html=True)
            c1, c2, _ = st.columns([1, 1, 2])
            c1.download_button("⬇️ 下载日志", st.session_state.logger.get_logs().encode("utf-8"),
                               f"automl_log_{get_file_timestamp()}.txt", "text/plain", use_container_width=True)
            if c2.button("🗑️ 清空日志", use_container_width=True):
                st.session_state.logger.clear()
                st.rerun()
        else:
            empty_state("📝", "暂无日志", "数据加载、训练、预测等操作会记录在这里")

    with right:
        section_title("建模报告", "汇总数据画像、模型对比、SHAP 重要性和预测结果")
        trained = st.session_state.trained
        if not trained:
            note("训练完成后可生成 HTML 报告", "info")
            return
        if st.button("📄 生成 HTML 报告", type="primary", use_container_width=True):
            try:
                profiler = get_profiler(st.session_state.data, trained['target'])
                cache = st.session_state.shap_cache
                analyzer = cache["analyzer"] if cache and cache["key"][0] == trained['id'] else None
                comparison_df = trained['trainer'].get_comparison_df()
                comparison_df = comparison_df[comparison_df["模型"].isin(trained['models'])]
                pred = st.session_state.predictions
                report = build_standard_report(
                    title=f"AutoML 建模报告 · {st.session_state.data_name}",
                    problem_type="分类" if trained['problem_type'] == "classification" else "回归",
                    profiler=profiler,
                    comparison_df=comparison_df,
                    training_results={k: v for k, v in trained['trainer'].results.items() if k in trained['models']},
                    shap_analyzer=analyzer,
                    predictions_df=pred["df"] if pred else None,
                )
                report.add_summary_item("目标列", trained['target'])
                report.add_summary_item("特征数", len(trained['feature_names']))
                st.session_state.report_html = (trained['id'], report.render_html())
                log_message("建模报告已生成", "SUCCESS")
            except Exception as e:
                st.error(f"❌ 报告生成失败: {e}")
                log_message(f"报告生成失败: {e}", "ERROR")
        report_html = st.session_state.report_html
        if report_html and report_html[0] == trained['id']:
            st.download_button("⬇️ 下载报告 (.html)", report_html[1].encode("utf-8"),
                               f"automl_report_{trained['id']}.html", "text/html", use_container_width=True)
            note("✅ 报告已生成，可用浏览器打开或打印为 PDF", "success")


# ==================== 主程序 ====================
def main():
    init_session_state()
    inject_css()

    cfg, train_clicked = render_sidebar()

    # 标签页上方的内容都放在固定且非空的容器里：其中元素数量变化时，
    # 标签页在页面树中的位置不变，Streamlit 就不会把当前标签页重置为第一个。
    # 注意不要在标签页之前放空容器，Streamlit 1.31 会因此在每次重跑时重置标签页。
    header_slot = st.container()
    kpi_slot = st.container()

    if st.session_state.data is not None and cfg is not None and train_clicked:
        with kpi_slot:
            run_training(cfg)

    # 训练结束后再渲染头部，保证步骤条反映最新状态
    with header_slot:
        render_header(cfg)

    if st.session_state.data is None or cfg is None:
        with kpi_slot:
            render_welcome()
        return

    with kpi_slot:
        render_kpis(cfg)

    tabs = st.tabs(["👁️ 数据概览", "📈 模型对比", "📋 模型评估", "🧠 SHAP 解释", "🔮 批量预测", "📝 日志与报告"])
    with tabs[0]:
        render_data_tab(cfg)
    with tabs[1]:
        render_comparison_tab()
    with tabs[2]:
        render_evaluation_tab()
    with tabs[3]:
        render_shap_tab()
    with tabs[4]:
        render_prediction_tab()
    with tabs[5]:
        render_logs_tab(cfg)


if __name__ == "__main__":
    main()
