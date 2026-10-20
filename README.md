# AutoML Studio · 自动化机器学习平台

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wangshiyun72-automl.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B)
![SHAP](https://img.shields.io/badge/SHAP-0.44-6A5ACD)

基于 Streamlit 的零代码 AutoML 平台：上传数据即可一键训练多种机器学习模型，支持网格 / 随机 / 贝叶斯调参，深度集成 SHAP 可解释性分析与批量预测。

**在线体验：** https://wangshiyun72-automl.streamlit.app

---

## ✨ 功能特性

| 模块 | 说明 |
|---|---|
| 📁 **数据加载** | 支持 CSV / XLSX / XLS，自动识别 UTF-8、GBK 编码和分隔符；内置 4 个示例数据集，无需准备数据即可体验 |
| 🔍 **数据画像** | 数据预览、列类型与缺失统计、目标分布图，自动检查缺失、高基数、强相关、潜在数据泄露等质量问题 |
| 🧹 **自动预处理** | 缺失值填充（数值列用中位数、分类列用众数）、分类变量编码、日期列转数值，可选 Z-score / MinMax 标准化；分类任务自动分层抽样 |
| 🧠 **多模型训练** | 分类 13 种、回归 15 种算法，一次勾选多个模型同时训练并自动排名 |
| ⚙️ **超参数优化** | 网格搜索、随机搜索、贝叶斯搜索（scikit-optimize），交叉验证折数可调 |
| 📈 **模型评估** | 排行榜、训练集 / 测试集对比、过拟合分析、混淆矩阵、ROC 曲线、各类别指标、残差分析、特征重要性 |
| 💡 **SHAP 可解释性** | 摘要图、重要性条形图、瀑布图、力图、依赖图、热力图共 6 种；根据模型类型自动选择 Tree / Linear / Kernel 解释器，多分类可切换解释类别 |
| 🔮 **批量预测** | 上传新数据批量预测并输出各类别概率；提示训练时未出现过的类别；新数据含目标列时自动计算预测效果 |
| 💾 **一键导出** | 图表（PNG / SVG / PDF）、预测结果（CSV / Excel）、SHAP 值、模型包（.zip）、HTML 建模报告、运行日志 |

## 🤖 支持的算法

| 类别 | 分类任务 | 回归任务 |
|---|---|---|
| ⭐ 推荐算法 | 随机森林、XGBoost、LightGBM、CatBoost | 随机森林、XGBoost、LightGBM、CatBoost |
| 集成学习 | 梯度提升、AdaBoost、极端随机树 | 梯度提升、AdaBoost、极端随机树 |
| 线性模型 | 逻辑回归 | 线性回归、岭回归、Lasso、弹性网络 |
| 其他算法 | 支持向量机、K 近邻、决策树、朴素贝叶斯、神经网络 (MLP) | 支持向量回归、K 近邻、决策树、神经网络 (MLP) |

## 🚀 使用流程

1. **加载数据**：在左侧上传文件，或选择「示例数据」直接加载。
2. **选择目标列与特征**：平台会自动识别是分类还是回归问题，也可以手动切换。
3. **设置预处理与算法**：选择训练集比例、标准化方法、要训练的算法和调参方式。
4. **开始训练**：在「模型对比」和「模型评估」标签页查看结果。
5. **SHAP 解释**：选择模型后点击「计算 SHAP」；之后切换图表类型无需重新计算。
6. **批量预测**：上传新数据（可先下载输入模板），预测完成后下载结果。

## 💻 本地运行

需要 Python 3.10 或 3.11。

```bash
git clone https://github.com/wangshiyun72/automl-platform.git
cd automl-platform
pip install -r requirements.txt
streamlit run app.py
```

浏览器会自动打开 http://localhost:8501 。

## 📂 项目结构

```
automl-platform/
├── app.py                      # 主程序：页面布局、训练流程、各标签页
├── config.py                   # 路径、模型列表、超参数网格配置
├── requirements.txt            # Python 依赖（版本已锁定）
├── packages.txt                # 云端部署时安装的系统包（中文字体）
├── assets/
│   └── style.css               # 界面样式
├── .streamlit/
│   └── config.toml             # 主题配色与上传大小限制
└── modules/
    ├── data_processor.py       # 数据预处理（缺失值、编码、标准化）
    ├── data_profiler.py        # 数据画像与质量检查
    ├── feature_engineer.py     # 自动特征工程
    ├── model_trainer.py        # 模型创建、训练、调参与评估
    ├── model_diagnostics.py    # 分类 / 回归诊断指标
    ├── shap_analyzer.py        # SHAP 计算与 6 种可视化
    ├── predictor.py            # 批量预测与模型文件管理
    ├── report_generator.py     # HTML / Markdown / Excel 报告生成
    └── utils.py                # 文件读取、模型保存、日志等工具函数
```

## 🛠️ 技术栈

- **Web 框架**：Streamlit
- **数据处理**：pandas、NumPy
- **机器学习**：scikit-learn、XGBoost、LightGBM、CatBoost
- **超参数优化**：scikit-optimize
- **可解释性**：SHAP
- **可视化**：Plotly、Matplotlib

## ☁️ 部署到 Streamlit Community Cloud

1. Fork 或推送本仓库到自己的 GitHub（需为公开仓库）。
2. 登录 [share.streamlit.io](https://share.streamlit.io)，点击 **Create app**。
3. 选择仓库、`main` 分支，主文件填 `app.py`。
4. 在 **Advanced settings** 中将 Python 版本设为 **3.11**（依赖锁定的版本不支持更新的 Python）。
5. 点击 **Deploy**，首次构建约 3–5 分钟。

`packages.txt` 会自动安装文泉驿中文字体，保证 SHAP 图中的中文正常显示。

## ⚠️ 注意事项

- **SHAP 计算速度**：树模型（随机森林、XGBoost 等）使用 TreeExplainer，速度很快；SVM、KNN、神经网络等使用 KernelExplainer，较慢，因此最多采样 150 个样本。
- **云端资源有限**：免费版 Streamlit Cloud 的 CPU 和内存有限。数据量较大时建议少选几个算法，避免使用网格搜索。
- **结果请及时下载**：云端服务器重启后保存的文件会被清空，请用页面上的下载按钮保存结果。
- **数据隐私**：上传的数据只保存在服务器内存里的当前会话中，不会写入磁盘，关闭页面后即被清除。（点击「打包模型文件」时，模型文件会临时保存在服务器上。）
