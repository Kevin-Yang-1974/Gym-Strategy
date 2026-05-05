# 智能健身房积分激励策略建模

本仓库保存校内数学建模赛论文与可复现实验材料，研究主题为：

**基于用户状态增强预测与经营约束优化的智能健身房积分激励策略研究**

项目围绕智能健身房“有效线下锻炼 + 线上分享或互动”返积分机制设计问题，基于用户订单、到店、运动、线上互动和私教购买等历史数据，构建会员续费预测、私教购买预测、积分策略仿真和熵权-TOPSIS 综合评价流程。最终推荐以 S4 流失召回强化策略作为稳健主方案，并将 S5 高强度综合激励策略作为进攻型备选方案。

## 项目内容

仓库当前主要保留以下材料：

- 论文正文与排版文件；
- 论文框架、建模思路和进度记录；
- 数据结构检查与状态转移分析程序；
- 会员续费、私教购买、积分策略仿真、TOPSIS 排序和灵敏度分析程序；
- 支撑论文正文的模型输出结果、图表、CSV 和模型工件。

原始业务数据、部分中间清洗数据和本地临时文件未纳入仓库。原因是这些文件体积较大，且可能包含用户行为、订单和业务明细信息，不适合直接公开。

## 目录结构

```text
.
├── 论文正文.md
├── 论文正文Latex版.tex
├── 论文正文Latex版.pdf
├── 论文框架.md
├── 当前进度记录.md
├── 可行预测模型方案.md
├── 状态转移_续费模型_收益优化表达式.md
├── 用户分类方法建议.md
├── 文献综述_返积分促活机制.md
├── 数据整理程序输出/
│   ├── inspect_excel_structure.py
│   └── run_state_transition_analysis.py
└── 模型输出结果/
    ├── compare_state_enhanced_rf_h30.py
    ├── compare_state_enhanced_rf_h60.py
    ├── compare_pt_purchase_models_enhanced_30d.py
    ├── run_renewal_cox_h30.py
    ├── simulate_points_strategy_topsis.py
    ├── sensitivity_points_strategy.py
    ├── sensitivity_multi_delay_windows.py
    ├── sensitivity_kappa_params.py
    ├── sensitivity_kappa_multi_delay_windows.py
    ├── topsis_pt_income_robustness.py
    ├── *.csv
    ├── *.png
    ├── *.md
    └── *.joblib / *.json
```

## 建模流程

整体流程可以概括为：

```text
原始业务数据
  -> 商品识别与用户月度特征构造
  -> 用户状态转移分析
  -> 会员续费模型
  -> 私教购买模型
  -> 积分响应函数与策略仿真
  -> 经营约束筛选
  -> 熵权-TOPSIS 综合评价
  -> 灵敏度与稳健性检验
  -> 论文正文与策略推荐
```

核心方法包括：

- 商品分类与用户状态识别；
- Cox 生存模型处理会员续费右删失问题；
- 状态增强随机森林 SERF 用于会员续费概率推理；
- CaseControl-HistGBDT 用于私教购买极低频事件预测；
- 边际递减积分响应函数用于策略情景模拟；
- 平台利润非负和成本收入比约束用于筛选可行策略；
- 熵权-TOPSIS 用于多指标综合评价；
- 单因素、多延期窗口、kappa 语义更新参数和替代 TOPSIS 指标稳健性检验。

## 关键文件

### 论文与说明

- `论文正文.md`：Markdown 版论文正文。
- `论文正文Latex版.tex`：LaTeX 版论文正文。
- `论文正文Latex版.pdf`：已编译 PDF。
- `论文框架.md`：论文结构和章节安排。
- `当前进度记录.md`：建模、论文修改和提交过程记录。
- `文献综述_返积分促活机制.md`：积分促活和激励机制相关文献综述。

### 数据整理与状态分析

- `数据整理程序输出/inspect_excel_structure.py`：检查原始 Excel 文件结构。
- `数据整理程序输出/run_state_transition_analysis.py`：生成商品分类和用户状态转移相关结果。
- `模型输出结果/generate_product_and_state_results.py`：汇总商品分类与状态转移结果。

### 续费模型

- `模型输出结果/run_renewal_cox_h30.py`：H30 续费 Cox 生存模型。
- `模型输出结果/compare_state_enhanced_rf_h30.py`：H30 状态增强随机森林与对比模型。
- `模型输出结果/compare_state_enhanced_rf_h60.py`：H60 稳健性窗口下的续费模型对比。
- `模型输出结果/summarize_renewal_h30_h60_stability.py`：H30/H60 续费模型稳健性汇总。

### 私教购买模型

- `模型输出结果/compare_pt_purchase_models_enhanced_30d.py`：增强私教购买模型、类别不平衡处理和 TopK/Lift 检验。
- `模型输出结果/pt_purchase_enhanced_model.joblib`：私教购买模型工件。
- `模型输出结果/pt_purchase_enhanced_features.json`：私教购买模型特征元信息。

### 积分策略与稳健性

- `模型输出结果/simulate_points_strategy_topsis.py`：候选积分策略仿真、经营约束筛选和熵权-TOPSIS 排序主程序。
- `模型输出结果/sensitivity_points_strategy.py`：积分策略单因素灵敏度分析。
- `模型输出结果/sensitivity_multi_delay_windows.py`：多延期观察窗口分析。
- `模型输出结果/sensitivity_kappa_params.py`：kappa 语义更新参数灵敏度分析。
- `模型输出结果/sensitivity_kappa_multi_delay_windows.py`：kappa 参数与多延期窗口联合灵敏度分析。
- `模型输出结果/topsis_pt_income_robustness.py`：替代私教收入指标下的 TOPSIS 稳健性检验。

## 环境依赖

建议使用 Python 3.10 或更高版本。主要 Python 依赖包括：

```bash
pip install pandas numpy scipy scikit-learn matplotlib joblib
```

如果需要重新编译论文 PDF，需要安装支持中文的 LaTeX 环境。本项目使用 XeLaTeX 编译：

```bash
xelatex -interaction=nonstopmode -halt-on-error -file-line-error "论文正文Latex版.tex"
xelatex -interaction=nonstopmode -halt-on-error -file-line-error "论文正文Latex版.tex"
```

Windows 环境下已使用 MiKTeX + XeLaTeX 编译通过。

## 复现实验说明

由于原始业务数据未提交，干净 clone 后不能直接完整重跑全部数据清洗和建模流程。若本地已具备原始数据和清洗后的建模数据，可按以下顺序运行：

```bash
python 数据整理程序输出/inspect_excel_structure.py
python 数据整理程序输出/run_state_transition_analysis.py
python 模型输出结果/generate_product_and_state_results.py
python 模型输出结果/run_renewal_cox_h30.py
python 模型输出结果/compare_state_enhanced_rf_h30.py
python 模型输出结果/compare_state_enhanced_rf_h60.py
python 模型输出结果/compare_pt_purchase_models_enhanced_30d.py
python 模型输出结果/simulate_points_strategy_topsis.py
python 模型输出结果/sensitivity_points_strategy.py
python 模型输出结果/sensitivity_multi_delay_windows.py
python 模型输出结果/sensitivity_kappa_params.py
python 模型输出结果/sensitivity_kappa_multi_delay_windows.py
python 模型输出结果/topsis_pt_income_robustness.py
```

部分脚本依赖前序脚本生成的数据集、模型工件或输出文件。推荐先阅读 `当前进度记录.md` 和 `模型输出结果/6_模型检验与假设验证材料汇总.md`，确认当前版本的主口径和结果解释边界。

## 结果口径

论文和程序中的策略模拟结果应按以下边界理解：

- 积分策略效果是“历史预测模型 + 积分响应假设”下的情景模拟，不是随机实验因果估计。
- 续费和私教购买概率提升相关指标在最终论文中解释为用户层面概率差求和，即期望人数提升，而不是单个用户概率或百分比。
- 私教购买属于极低频事件，模型更适合作为高潜用户排序和策略模拟输入，而不是用单一阈值解释所有用户是否购买。
- S4 被推荐为稳健主策略，S5 适合作为平台愿意承担更高补贴时的进攻型备选策略。

## 当前 PDF 状态

`论文正文Latex版.pdf` 已由 `论文正文Latex版.tex` 编译生成。最近一次检查结果：

- 页面规格：A4；
- 页数：44 页；
- 编译方式：XeLaTeX 连续两遍；
- 未发现 Overfull、未定义引用、致命错误或需要继续重跑的交叉引用提示。

## 数据与隐私说明

本仓库不公开原始用户级业务数据。若需要复现全部流程，应在本地放置同结构数据，并确保符合数据授权和隐私保护要求。公开仓库中的结果文件仅用于论文复核、方法说明和流程展示。
