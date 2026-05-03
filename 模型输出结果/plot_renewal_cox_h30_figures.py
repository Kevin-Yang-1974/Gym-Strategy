# -*- coding: utf-8 -*-
"""根据 H=30 Cox 主模型结果生成论文图片。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "模型输出结果"

COEF_PATH = OUT_DIR / "5.3_Cox_H30_系数风险比.csv"
PRED_PATH = OUT_DIR / "5.3_Cox_H30_预测续费概率.csv"
SUMMARY_PATH = OUT_DIR / "5.3_Cox_H30_模型摘要.csv"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] =150


def save_event_distribution(pred):
	counts = pred["event"].value_counts().sort_index()
	labels = ["未观察到续费/删失", "观察到续费"]
	values = [counts.get(0,0), counts.get(1,0)]

	fig, ax = plt.subplots(figsize=(6,4))
	bars = ax.bar(labels, values, color=["#7aa6c2", "#e6955f"])
	ax.set_ylabel("样本数")
	ax.set_title("H=30会员续费事件分布")
	ax.tick_params(axis="x", labelrotation=15)
	for bar in bars:
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width() /2, height, f"{int(height):,}", ha="center", va="bottom")
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.3_图1_H30续费事件分布.png", bbox_inches="tight")
	plt.close(fig)


def save_pred_probability_distribution(pred):
	fig, ax = plt.subplots(figsize=(7,4.5))
	ax.hist(pred["pred_renew_prob_H30"], bins=40, color="#5b8db8", alpha=0.85, edgecolor="white")
	ax.set_xlabel("预测30天内续费概率")
	ax.set_ylabel("样本数")
	ax.set_title("Cox模型预测续费概率分布（H=30）")
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.3_图2_Cox预测续费概率分布_H30.png", bbox_inches="tight")
	plt.close(fig)


def save_top_hr_bar(coef):
	coef = coef.copy()
	coef["log_HR_abs"] = np.abs(np.log(coef["HR"].clip(lower=1e-12)))
	top = coef.sort_values("log_HR_abs", ascending=False).head(15).sort_values("HR")
	colors = np.where(top["HR"] >=1, "#d9795d", "#5b8db8")

	fig, ax = plt.subplots(figsize=(8,6))
	ax.barh(top["中文含义"], top["HR"], color=colors)
	ax.axvline(1, color="black", linestyle="--", linewidth=1)
	ax.set_xlabel("风险比 HR")
	ax.set_title("Cox模型关键变量风险比 Top15（H=30）")
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.3_图3_Cox关键变量风险比Top15_H30.png", bbox_inches="tight")
	plt.close(fig)


def save_forest_plot(coef):
	plot_df = coef.copy()
	plot_df["log_HR_abs"] = np.abs(np.log(plot_df["HR"].clip(lower=1e-12)))
	plot_df = plot_df.sort_values("log_HR_abs", ascending=False).head(20).sort_values("HR")
	plot_df["ci_low"] = np.exp(plot_df["系数beta"] -1.96 * plot_df["标准误近似"])
	plot_df["ci_high"] = np.exp(plot_df["系数beta"] +1.96 * plot_df["标准误近似"])
	plot_df["ci_low"] = np.minimum(plot_df["ci_low"], plot_df["HR"])
	plot_df["ci_high"] = np.maximum(plot_df["ci_high"], plot_df["HR"])

	y = np.arange(len(plot_df))
	xerr = np.vstack([
		np.maximum(plot_df["HR"] - plot_df["ci_low"],0),
		np.maximum(plot_df["ci_high"] - plot_df["HR"],0),
	])

	fig, ax = plt.subplots(figsize=(8,7))
	ax.errorbar(plot_df["HR"], y, xerr=xerr, fmt="o", color="#333333", ecolor="#888888", capsize=3)
	ax.axvline(1, color="red", linestyle="--", linewidth=1)
	ax.set_yticks(y)
	ax.set_yticklabels(plot_df["中文含义"])
	ax.set_xlabel("风险比 HR及95%近似置信区间")
	ax.set_title("Cox模型风险比森林图（H=30，Top20变量）")
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.3_图4_Cox风险比森林图_H30.png", bbox_inches="tight")
	plt.close(fig)


def main():
	coef = pd.read_csv(COEF_PATH)
	pred = pd.read_csv(PRED_PATH)
	_ = pd.read_csv(SUMMARY_PATH)

	save_event_distribution(pred)
	save_pred_probability_distribution(pred)
	save_top_hr_bar(coef)
	save_forest_plot(coef)
	print("已生成 H=30 Cox 模型论文图片。")


if __name__ == "__main__":
	main()
