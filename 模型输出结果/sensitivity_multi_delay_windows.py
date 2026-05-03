# -*- coding: utf-8 -*-
"""多延期观察窗口灵敏度分析。

基于 simulate_points_strategy_topsis.py 已输出的多延期窗口方案汇总，
对每个“活跃延续系数 x 延期窗口数”组合重新计算熵权 TOPSIS 排序，
并输出可用于论文第7章的多窗口敏感性结果和图表。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import simulate_points_strategy_topsis as sim


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "模型输出结果"

MULTI_SUMMARY_PATH = OUT_DIR / "5.6_多延迟观察窗口方案汇总.csv"
OUT_SUMMARY_PATH = OUT_DIR / "5.7_多延期窗口灵敏度分析汇总.csv"
OUT_REPORT_PATH = OUT_DIR / "5.7_多延期窗口灵敏度分析报告.md"
FIG_GAP_PATH = OUT_DIR / "5.7_图3_多延期窗口S5相对S0利润差.png"
FIG_RANK_PATH = OUT_DIR / "5.7_图4_多延期窗口S5排序变化.png"


NUMERIC_COLS = [
	"delay_window_count",
	"总窗口数",
	"延续活跃系数",
	"平台利润",
	"总活跃度提升",
	"最终续费概率提升",
	"最终私教购买概率提升",
	"积分成本",
	"成本收入比",
]


def load_multi_summary():
	if not MULTI_SUMMARY_PATH.exists():
		raise FileNotFoundError(f"缺少多延期窗口方案汇总文件：{MULTI_SUMMARY_PATH}")
	df = pd.read_csv(MULTI_SUMMARY_PATH)
	for col in NUMERIC_COLS:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	for col in ["是否利润非负", "是否满足成本约束"]:
		if col in df.columns:
			df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
	return df


def pick_strategy(group, code):
	rows = group[group["策略编号"] == code]
	if rows.empty:
		return None
	return rows.iloc[0]


def summarize_group(group):
	label = group["观察期标签"].iloc[0]
	delay_count = int(group["delay_window_count"].iloc[0])
	total_windows = int(group["总窗口数"].iloc[0])
	carry_a = float(group["延续活跃系数"].iloc[0])

	topsis = sim.entropy_topsis(group.copy())
	if topsis.empty:
		best_code = "无"
		best_name = "无可行策略"
		best_score = np.nan
		rank_map = {}
		score_map = {}
	else:
		best = topsis.iloc[0]
		best_code = best["策略编号"]
		best_name = best["策略名称"]
		best_score = float(best["TOPSIS得分"])
		rank_map = topsis.set_index("策略编号")["排序"].to_dict()
		score_map = topsis.set_index("策略编号")["TOPSIS得分"].to_dict()

	s0 = pick_strategy(group, "S0")
	s1 = pick_strategy(group, "S1")
	s5 = pick_strategy(group, "S5")
	s4 = pick_strategy(group, "S4")
	s3 = pick_strategy(group, "S3")
	s2 = pick_strategy(group, "S2")

	s0_profit = float(s0["平台利润"]) if s0 is not None else np.nan
	s1_profit = float(s1["平台利润"]) if s1 is not None else np.nan
	s5_profit = float(s5["平台利润"]) if s5 is not None else np.nan

	return {
		"观察期标签": label,
		"延续活跃系数": carry_a,
		"delay_window_count": delay_count,
		"总窗口数": total_windows,
		"可行策略数量": int(len(topsis)),
		"最优策略编号": best_code,
		"最优策略名称": best_name,
		"最优策略TOPSIS得分": best_score,
		"S5排序": rank_map.get("S5", np.nan),
		"S5_TOPSIS得分": score_map.get("S5", np.nan),
		"S4排序": rank_map.get("S4", np.nan),
		"S3排序": rank_map.get("S3", np.nan),
		"S2排序": rank_map.get("S2", np.nan),
		"S5是否第一": best_code == "S5",
		"S0平台利润": s0_profit,
		"S1平台利润": s1_profit,
		"S5平台利润": s5_profit,
		"S5相对S0利润差": s5_profit - s0_profit,
		"S5相对S1利润差": s5_profit - s1_profit,
		"S5成本收入比": float(s5["成本收入比"]) if s5 is not None else np.nan,
		"S5积分成本": float(s5["积分成本"]) if s5 is not None else np.nan,
		"S5总活跃度提升": float(s5["总活跃度提升"]) if s5 is not None else np.nan,
		"S5续费概率提升": float(s5["最终续费概率提升"]) if s5 is not None else np.nan,
		"S5私教购买概率提升": float(s5["最终私教购买概率提升"]) if s5 is not None else np.nan,
		"S5私教收入提升": float(s5["私教收入提升"]) if s5 is not None else np.nan,
		"S4平台利润": float(s4["平台利润"]) if s4 is not None else np.nan,
		"S4私教购买概率提升": float(s4["最终私教购买概率提升"]) if s4 is not None else np.nan,
		"S4私教收入提升": float(s4["私教收入提升"]) if s4 is not None else np.nan,
		"S3平台利润": float(s3["平台利润"]) if s3 is not None else np.nan,
		"S3私教购买概率提升": float(s3["最终私教购买概率提升"]) if s3 is not None else np.nan,
		"S3私教收入提升": float(s3["私教收入提升"]) if s3 is not None else np.nan,
		"S2平台利润": float(s2["平台利润"]) if s2 is not None else np.nan,
	}


def build_summary(df):
	rows = []
	for _, group in df.groupby(["观察期标签", "delay_window_count"], sort=True):
		rows.append(summarize_group(group))
	out = pd.DataFrame(rows)
	return out.sort_values(["延续活跃系数", "delay_window_count"]).reset_index(drop=True)


def save_figures(summary):
	fig, ax = plt.subplots(figsize=(9,5.4))
	for label, group in summary.groupby("观察期标签"):
		group = group.sort_values("delay_window_count")
		carry = group["延续活跃系数"].iloc[0]
		ax.plot(
			group["总窗口数"],
			group["S5相对S0利润差"],
			marker="o",
			linewidth=2,
			label=f"{label}: gamma_A={carry:.2f}",
		)
	ax.axhline(0, color="#555555", linewidth=1)
	ax.set_xlabel("总观察窗口数")
	ax.set_ylabel("S5相对S0利润差")
	ax.set_title("多延期观察窗口下S5相对S0利润差")
	ax.legend()
	fig.tight_layout()
	fig.savefig(FIG_GAP_PATH, bbox_inches="tight")
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(9,5.4))
	for label, group in summary.groupby("观察期标签"):
		group = group.sort_values("delay_window_count")
		carry = group["延续活跃系数"].iloc[0]
		ax.plot(
			group["总窗口数"],
			group["S5排序"],
			marker="o",
			linewidth=2,
			label=f"{label}: gamma_A={carry:.2f}",
		)
	ax.set_xlabel("总观察窗口数")
	ax.set_ylabel("S5 TOPSIS排序")
	ax.set_title("多延期观察窗口下S5排序变化")
	ax.invert_yaxis()
	ax.set_yticks([1,2,3,4,5])
	ax.legend()
	fig.tight_layout()
	fig.savefig(FIG_RANK_PATH, bbox_inches="tight")
	plt.close(fig)


def first_cross_table(summary):
	rows = []
	for label, group in summary.groupby("观察期标签"):
		group = group.sort_values("delay_window_count")
		over = group[group["S5相对S0利润差"] > 0]
		if over.empty:
			first_total = np.nan
			first_delay = np.nan
			first_gap = np.nan
			has_cross = False
		else:
			first = over.iloc[0]
			first_total = int(first["总窗口数"])
			first_delay = int(first["delay_window_count"])
			first_gap = float(first["S5相对S0利润差"])
			has_cross = True
		last = group.iloc[-1]
		rows.append({
			"观察期标签": label,
			"延续活跃系数": float(last["延续活跃系数"]),
			"是否出现利润反超S0": has_cross,
			"首次反超延期窗口数": first_delay,
			"首次反超总窗口数": first_total,
			"首次反超时利润差": first_gap,
			"最大测试总窗口数": int(last["总窗口数"]),
			"最大测试窗口时利润差": float(last["S5相对S0利润差"]),
			"最大测试窗口时S5排序": last["S5排序"],
		})
	return pd.DataFrame(rows)


def write_report(summary):
	cross = first_cross_table(summary)
	s5_first_count = int(summary["S5是否第一"].sum())
	total_count = int(len(summary))
	best_counts = summary["最优策略编号"].value_counts().reset_index()
	best_counts.columns = ["最优策略编号", "出现次数"]

	report = f"""#5.7 多延期观察窗口灵敏度分析报告

本文件由 `模型输出结果/sensitivity_multi_delay_windows.py` 自动生成，用于补充第7章灵敏度分析中关于多延期观察窗口的结果。该分析基于 `5.6_多延迟观察窗口方案汇总.csv`，对每个“活跃延续系数 $\\gamma_A$ × 延期窗口数”组合重新计算熵权-TOPSIS排序。

##1. 分析口径

多延期观察窗口用于检验积分激励结束后，若活跃提升存在残余惯性，策略排序和累计利润差是否发生变化。窗口1计入积分成本，延期窗口不再新增积分成本，但根据残余活跃提升重新构造反事实特征：续费概率由 SERF 工件重新推理，私教购买概率由增强私教购买模型 `pt_purchase_enhanced_model.joblib` 重新推理。因此，本分析中的熵权-TOPSIS排序已使用 `最终私教购买概率提升`，该指标不再保持基线或恒为0。

##2. S5相对S0利润反超情况

{cross.to_markdown(index=False)}

##3. TOPSIS排序稳定性

在全部 {total_count} 个“延续系数 × 延期窗口数”组合中，S5 排名第一的组合数为 {s5_first_count} 个。各最优策略出现次数如下：

{best_counts.to_markdown(index=False)}

完整汇总见 `5.7_多延期窗口灵敏度分析汇总.csv`。

##4. 主要结论

多延期观察窗口结果显示，在当前响应假设下，只要延期观察窗口数达到2个，即总窗口数达到3个，S5 在所有测试的活跃延续系数情景下均出现相对 S0 的累计利润反超。随着 $\\gamma_A$ 提高，S5 在长期窗口中的利润反超幅度进一步扩大。这说明 S5 的短期利润压力主要来自窗口1的高额积分成本；若积分带来的活跃提升具有一定延期效应，则其长期累计收益存在弥补补贴成本的可能。

但该结论依赖于残余活跃提升假设、SERF 反事实续费推理路径和增强私教购买模型反事实推理路径，仍属于情景模拟，不应解释为真实因果结论。论文中应分别表述“短期利润承压”“多延期观察假设下可能反超”和“私教购买概率提升来自模型反事实预测而非真实干预估计”。

##5. 可放入论文的图片

- `5.7_图3_多延期窗口S5相对S0利润差.png`
- `5.7_图4_多延期窗口S5排序变化.png`
"""
	OUT_REPORT_PATH.write_text(report, encoding="utf-8")


def main():
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	df = load_multi_summary()
	summary = build_summary(df)
	summary.to_csv(OUT_SUMMARY_PATH, index=False, encoding="utf-8-sig")
	save_figures(summary)
	write_report(summary)
	print(f"已生成多延期窗口灵敏度分析汇总：{OUT_SUMMARY_PATH}")
	print(f"已生成多延期窗口灵敏度分析报告：{OUT_REPORT_PATH}")
	print(f"已生成论文图表：{FIG_GAP_PATH}")
	print(f"已生成论文图表：{FIG_RANK_PATH}")


if __name__ == "__main__":
	main()
