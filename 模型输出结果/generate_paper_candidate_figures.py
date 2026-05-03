# -*- coding: utf-8 -*-
"""Generate supplemental paper candidate figures from cleaned result tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent


plt.rcParams["font.sans-serif"] = [
	"Microsoft YaHei",
	"SimHei",
	"Arial Unicode MS",
	"DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


STATE_LABELS = {
	"Z1_高活跃高消费用户": "Z1\n高活跃高消费",
	"Z6_私教潜在用户": "Z6\n私教潜在",
	"Z7_流失风险用户": "Z7\n流失风险",
}

PRODUCT_LABELS = {
	"member": "会员商品",
	"pt": "私教商品",
	"entry_pass": "入场/次卡",
	"unknown": "未识别",
	"other": "其他",
}


def read_csv(name):
	return pd.read_csv(OUT_DIR / name)


def save_product_category_share():
	count_df = read_csv("5.1_商品大类分布.csv")
	amount_df = read_csv("5.1_商品大类金额汇总.csv")
	df = count_df.merge(amount_df[["商品大类", "金额总和"]], on="商品大类", how="left")
	df["订单占比"] = df["占比"].astype(float)
	df["金额占比"] = df["金额总和"] / df["金额总和"].sum()
	df["商品大类显示"] = df["商品大类"].map(PRODUCT_LABELS).fillna(df["商品大类"])
	df = df.sort_values("订单占比", ascending=False)

	x = np.arange(len(df))
	width = 0.36
	fig, ax = plt.subplots(figsize=(8.2, 4.8))
	ax.bar(x - width / 2, df["订单占比"] * 100, width=width, label="订单占比", color="#4E79A7")
	ax.bar(x + width / 2, df["金额占比"] * 100, width=width, label="金额占比", color="#F28E2B")
	ax.set_xticks(x)
	ax.set_xticklabels(df["商品大类显示"])
	ax.set_ylabel("占比（%）")
	ax.set_title("商品大类订单占比与金额占比")
	ax.legend(frameon=False)
	ax.grid(axis="y", linestyle="--", alpha=0.25)
	for i, value in enumerate(df["订单占比"] * 100):
		ax.text(i - width / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
	for i, value in enumerate(df["金额占比"] * 100):
		ax.text(i + width / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.1_图1_商品大类订单与金额占比.png", bbox_inches="tight")
	plt.close(fig)


def save_state_distribution():
	df = read_csv("5.2_当前状态分布.csv")
	df["状态显示"] = df["当前状态"].map(STATE_LABELS).fillna(df["当前状态"])
	df = df.sort_values("占比", ascending=False)

	fig, ax = plt.subplots(figsize=(7.4, 4.6))
	bars = ax.bar(df["状态显示"], df["占比"] * 100, color=["#59A14F", "#E15759", "#4E79A7"])
	ax.set_ylabel("占比（%）")
	ax.set_title("用户当前状态分布")
	ax.grid(axis="y", linestyle="--", alpha=0.25)
	for bar, count, share in zip(bars, df["样本数"], df["占比"]):
		ax.text(
			bar.get_x() + bar.get_width() / 2,
			bar.get_height(),
			f"{share * 100:.2f}%\n{int(count):,}",
			ha="center",
			va="bottom",
			fontsize=8,
		)
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.2_图1_用户当前状态分布.png", bbox_inches="tight")
	plt.close(fig)


def save_state_transition_heatmap():
	df = read_csv("5.2_状态转移概率矩阵.csv")
	index_col = df.columns[0]
	matrix = df.set_index(index_col).astype(float)
	y_labels = [STATE_LABELS.get(v, v) for v in matrix.index]
	x_labels = [STATE_LABELS.get(v, v) for v in matrix.columns]

	fig, ax = plt.subplots(figsize=(7.2, 5.6))
	im = ax.imshow(matrix.values * 100, cmap="YlGnBu", vmin=0, vmax=100)
	ax.set_xticks(range(len(x_labels)))
	ax.set_xticklabels(x_labels)
	ax.set_yticks(range(len(y_labels)))
	ax.set_yticklabels(y_labels)
	ax.set_xlabel("下一期状态")
	ax.set_ylabel("当前状态")
	ax.set_title("用户状态转移概率矩阵")
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			value = matrix.iloc[i, j] * 100
			color = "white" if value > 55 else "#1f2933"
			ax.text(j, i, f"{value:.2f}%", ha="center", va="center", color=color, fontsize=9)
	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label("转移概率（%）")
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.2_图2_用户状态转移概率热力图.png", bbox_inches="tight")
	plt.close(fig)


def best_counts(path):
	df = read_csv(path)
	return df["最优策略编号"].value_counts().rename_axis("策略").reset_index(name="次数")


def save_alt_best_count_comparison():
	series = {
		"多延期\n原口径": best_counts("5.7_多延期窗口灵敏度分析汇总.csv"),
		"多延期\n替代口径": best_counts("5.7_TOPSIS替代指标稳健性_多延期窗口汇总.csv"),
		"kappa多延期\n原口径": best_counts("5.7_kappa多延期窗口灵敏度分析汇总.csv"),
		"kappa多延期\n替代口径": best_counts("5.7_TOPSIS替代指标稳健性_kappa多延期汇总.csv"),
	}
	strategies = ["S5", "S4", "S2", "S1", "S3"]
	plot_df = pd.DataFrame(index=series.keys(), columns=strategies).fillna(0)
	for label, count_df in series.items():
		for _, row in count_df.iterrows():
			if row["策略"] in strategies:
				plot_df.loc[label, row["策略"]] = int(row["次数"])
	plot_df = plot_df.astype(int)

	fig, ax = plt.subplots(figsize=(9.2, 5.2))
	bottom = np.zeros(len(plot_df))
	colors = {"S5": "#4E79A7", "S4": "#59A14F", "S2": "#F28E2B", "S1": "#9C755F", "S3": "#E15759"}
	for strategy in strategies:
		values = plot_df[strategy].values
		ax.bar(plot_df.index, values, bottom=bottom, label=strategy, color=colors[strategy])
		for idx, value in enumerate(values):
			if value > 0:
				ax.text(idx, bottom[idx] + value / 2, str(value), ha="center", va="center", color="white", fontsize=9)
		bottom += values
	ax.set_ylabel("最优策略出现次数")
	ax.set_title("原口径与私教收入替代口径下的最优策略次数")
	ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
	ax.grid(axis="y", linestyle="--", alpha=0.25)
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.7_图11_TOPSIS替代指标最优策略次数对比.png", bbox_inches="tight")
	plt.close(fig)


def rank_count(df, strategy):
	col = f"{strategy}排序"
	count = df[col].value_counts().sort_index()
	return count


def save_alt_rank_distribution():
	multi = read_csv("5.7_TOPSIS替代指标稳健性_多延期窗口汇总.csv")
	kappa = read_csv("5.7_TOPSIS替代指标稳健性_kappa多延期汇总.csv")
	strategies = ["S5", "S4", "S3"]
	ranks = [1, 2, 3, 4, 5]
	fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=False)
	for ax, df, title in [
		(axes[0], multi, "普通多延期窗口（替代口径）"),
		(axes[1], kappa, "kappa多延期窗口（替代口径）"),
	]:
		x = np.arange(len(ranks))
		width = 0.24
		colors = {"S5": "#4E79A7", "S4": "#59A14F", "S3": "#E15759"}
		for offset, strategy in zip([-width, 0, width], strategies):
			counts = rank_count(df, strategy).reindex(ranks, fill_value=0)
			ax.bar(x + offset, counts.values, width=width, label=strategy, color=colors[strategy])
		ax.set_xticks(x)
		ax.set_xticklabels([str(v) for v in ranks])
		ax.set_xlabel("排序")
		ax.set_ylabel("出现次数")
		ax.set_title(title)
		ax.grid(axis="y", linestyle="--", alpha=0.25)
	axes[1].legend(frameon=False, loc="upper right")
	fig.suptitle("S5/S4/S3 在私教收入替代口径下的排名分布", y=1.02)
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.7_图12_TOPSIS替代指标S5_S4_S3排名分布.png", bbox_inches="tight")
	plt.close(fig)


def write_figure_catalog():
	catalog = """# 论文图片候选清单

本清单由 `模型输出结果/generate_paper_candidate_figures.py` 自动生成，用于整理当前可直接放入论文的图片与本轮补图。

## 本轮补输出图片

- `5.1_图1_商品大类订单与金额占比.png`：用于第5.1节，展示会员、私教、入场类等商品的订单占比和金额占比。
- `5.2_图1_用户当前状态分布.png`：用于第5.2节，展示 Z7、Z6、Z1 三类用户状态占比。
- `5.2_图2_用户状态转移概率热力图.png`：用于第5.2节，展示用户自然状态转移概率。
- `5.7_图11_TOPSIS替代指标最优策略次数对比.png`：用于第6.8或第7章，比较原口径与私教收入替代口径下的最优策略次数。
- `5.7_图12_TOPSIS替代指标S5_S4_S3排名分布.png`：用于第6.8或第7章，展示 S5、S4、S3 在替代口径下的排名分布。

## 已有核心图片

- 第5.3节：`5.3_图1_H30续费事件分布.png`、`5.3_图3_Cox关键变量风险比Top15_H30.png`、`5.3_图15_状态增强随机森林特征重要性Top15_H30.png`、`5.3_图16_H30_H60续费模型AUC_Brier对比.png`、`5.3_图17_H30_H60续费模型F1_Recall对比.png`。
- 第5.4节：`5.4_图4_增强私教购买模型指标对比.png`、`5.4_图5_增强私教购买模型PR曲线.png`、`5.4_图6_增强私教购买模型TopK_Lift.png`、`5.4_图7_增强私教购买模型过拟合风险.png`。
- 第5.5-5.6节：`5.5_图1_候选积分策略利润成本对比.png`、`5.5_图2_候选积分策略概率提升对比.png`、`5.6_图1_可行策略TOPSIS排序.png`、`5.6_图2_S3_S4_S5私教模型接入后对比.png`。
- 第7章：`5.7_图1_灵敏度情景最优策略变化.png`、`5.7_图3_多延期窗口S5相对S0利润差.png`、`5.7_图4_多延期窗口S5排序变化.png`、`5.7_图5_kappa参数下S5相对S0利润差热力图.png`、`5.7_图6_kappa参数下最优策略变化热力图.png`、`5.7_图7_kappa多延期窗口S5首次反超窗口热力图.png`、`5.7_图8_kappa多延期窗口S4最优次数热力图.png`、`5.7_图9_kappa参数下S4相对S0利润差热力图.png`、`5.7_图10_kappa多延期窗口S4首次反超窗口热力图.png`。

## 取舍建议

论文正文不宜把所有图都放入主文。建议主文优先使用第5.2状态转移热力图、第5.4增强私教模型PR曲线或TopK Lift、第5.6 TOPSIS排序图、第7章多延期利润差图、kappa最优策略热力图和本轮替代指标最优策略次数对比图；其余图可放附录或仅在文字中引用。
"""
	(OUT_DIR / "论文图片候选清单.md").write_text(catalog, encoding="utf-8")


def main():
	save_product_category_share()
	save_state_distribution()
	save_state_transition_heatmap()
	save_alt_best_count_comparison()
	save_alt_rank_distribution()
	write_figure_catalog()
	print("已生成论文候选补充图片和清单。")


if __name__ == "__main__":
	main()
