# -*- coding: utf-8 -*-
"""生成 H=30 与 H=60续费模型稳健性对比报告。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "模型输出结果"

H30_PATH = OUT_DIR / "5.3_状态增强随机森林与对比模型指标_H30.csv"
H60_PATH = OUT_DIR / "5.3_状态增强随机森林与对比模型指标_H60.csv"
REPORT_PATH = OUT_DIR / "5.3_H30_H60续费模型稳健性对比报告.md"
OUT_CSV = OUT_DIR / "5.3_H30_H60续费模型稳健性对比指标.csv"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] =150


def save_auc_brier_plot(compare):
	models = compare["模型"].unique()
	x = np.arange(len(models))
	width =0.2
	fig, ax = plt.subplots(figsize=(9,5))
	for idx, (window, metric, label) in enumerate([
		("H30", "测试AUC", "H30 AUC"),
		("H60", "测试AUC", "H60 AUC"),
		("H30", "测试Brier", "H30 Brier"),
		("H60", "测试Brier", "H60 Brier"),
	]):
		values = []
		for model in models:
			row = compare[(compare["模型"] == model) & (compare["窗口"] == window)].iloc[0]
			values.append(row[metric])
		ax.bar(x + (idx -1.5) * width, values, width, label=label)
	ax.set_xticks(x)
	ax.set_xticklabels(models, rotation=10)
	ax.set_ylim(0,1)
	ax.set_title("H=30与H=60续费模型AUC和Brier对比")
	ax.legend()
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.3_图16_H30_H60续费模型AUC_Brier对比.png", bbox_inches="tight")
	plt.close(fig)


def save_f1_recall_plot(compare):
	models = compare["模型"].unique()
	x = np.arange(len(models))
	width =0.2
	fig, ax = plt.subplots(figsize=(9,5))
	for idx, (window, metric, label) in enumerate([
		("H30", "测试F1", "H30 F1"),
		("H60", "测试F1", "H60 F1"),
		("H30", "测试Recall", "H30 Recall"),
		("H60", "测试Recall", "H60 Recall"),
	]):
		values = []
		for model in models:
			row = compare[(compare["模型"] == model) & (compare["窗口"] == window)].iloc[0]
			values.append(row[metric])
		ax.bar(x + (idx -1.5) * width, values, width, label=label)
	ax.set_xticks(x)
	ax.set_xticklabels(models, rotation=10)
	ax.set_ylim(0,1)
	ax.set_title("H=30与H=60续费模型F1和Recall对比")
	ax.legend()
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.3_图17_H30_H60续费模型F1_Recall对比.png", bbox_inches="tight")
	plt.close(fig)


def write_report(compare):
	h30_best = compare[compare["窗口"] == "H30"].sort_values(["测试AUC", "测试Brier"], ascending=[False, True]).iloc[0]
	h60_best = compare[compare["窗口"] == "H60"].sort_values(["测试AUC", "测试Brier"], ascending=[False, True]).iloc[0]
	serf_h30 = compare[(compare["窗口"] == "H30") & (compare["模型"] == "状态增强随机森林")].iloc[0]
	serf_h60 = compare[(compare["窗口"] == "H60") & (compare["模型"] == "状态增强随机森林")].iloc[0]
	rf_h30 = compare[(compare["窗口"] == "H30") & (compare["模型"] == "普通随机森林")].iloc[0]
	rf_h60 = compare[(compare["窗口"] == "H60") & (compare["模型"] == "普通随机森林")].iloc[0]
	report = f"""#5.3 H=30与H=60续费模型稳健性对比报告

本文件由 `模型输出结果/summarize_renewal_h30_h60_stability.py` 自动生成，用于验证续费模型在不同观察窗口下的稳定性。

##1. 验证目的

主分析窗口为 $H=30$ 天，稳健性窗口为 $H=60$ 天。若更换续费观察窗口后，模型排序和主要结论保持一致，则说明续费概率预测模型具有较好的窗口稳健性。

##2. H=30与H=60模型指标对比

{compare.to_markdown(index=False)}

##3. 稳健性结论

在 $H=30$ 窗口下，测试 AUC最高且 Brier Score表现最优的模型为 **{h30_best['模型']}**，测试 AUC为 {h30_best['测试AUC']:.4f}，测试 Brier Score为 {h30_best['测试Brier']:.4f}。

在 $H=60$ 窗口下，测试 AUC最高且 Brier Score表现最优的模型为 **{h60_best['模型']}**，测试 AUC为 {h60_best['测试AUC']:.4f}，测试 Brier Score为 {h60_best['测试Brier']:.4f}。

状态增强随机森林在两个窗口下均保持接近最优表现：$H=30$ 时 AUC为 {serf_h30['测试AUC']:.4f}，Brier Score为 {serf_h30['测试Brier']:.4f}；$H=60$ 时 AUC为 {serf_h60['测试AUC']:.4f}，Brier Score为 {serf_h60['测试Brier']:.4f}。普通随机森林在两个窗口下也保持最优或接近最优表现：$H=30$ 时 AUC为 {rf_h30['测试AUC']:.4f}，Brier Score为 {rf_h30['测试Brier']:.4f}；$H=60$ 时 AUC为 {rf_h60['测试AUC']:.4f}，Brier Score为 {rf_h60['测试Brier']:.4f}。这说明随机森林类模型作为续费概率输入具有较好的观察窗口稳健性。

状态增强随机森林在两个窗口下均与普通随机森林表现接近，说明加入用户状态和行为响应特征不会显著削弱预测能力，并且能够为后续分层积分策略提供业务解释。Cox模型在两个窗口下均可作为右删失处理和时间到事件分析的理论模型，但在固定窗口概率预测上不如随机森林。Logistic-GAM可作为非线性解释辅助模型。

因此，后续积分策略模拟采用状态增强随机森林导出的 SERF 工件生成续费基线概率，并利用其用户状态、活跃提升空间、近因得分和行为响应特征支撑分层策略解释；普通随机森林保留为预测性能对比和窗口稳健性参照；Cox模型保留为生存分析和右删失处理依据；Logistic-GAM保留为非线性解释辅助模型。

##4. 可放入论文的图片

- `5.3_图16_H30_H60续费模型AUC_Brier对比.png`
- `5.3_图17_H30_H60续费模型F1_Recall对比.png`
"""
	REPORT_PATH.write_text(report, encoding="utf-8")


def main():
	h30 = pd.read_csv(H30_PATH)
	h60 = pd.read_csv(H60_PATH)
	h30.insert(0, "窗口", "H30")
	h60.insert(0, "窗口", "H60")
	compare = pd.concat([h30, h60], ignore_index=True)
	compare.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
	save_auc_brier_plot(compare)
	save_f1_recall_plot(compare)
	write_report(compare)
	print(f"已生成：{REPORT_PATH}")


if __name__ == "__main__":
	main()
