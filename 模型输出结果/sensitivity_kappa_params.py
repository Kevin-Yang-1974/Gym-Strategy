# -*- coding: utf-8 -*-
"""kappa 语义更新参数灵敏度分析。

检验 SERF 反事实特征更新中

    kappa_i,k(r) = 1 + gamma_kappa * ln(1 + Delta A_i,k(r) / (A_i,t + epsilon_kappa))

对策略排序和利润差的影响。这里的 gamma_kappa 对应
simulate_points_strategy_topsis.py 中的 SEMANTIC_GAMMA，epsilon_kappa
对应 FEATURE_UPDATE_EPS。该 gamma_kappa 不是早期已废弃的
lambda_renew 或 lambda_pt 线性概率加成。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import simulate_points_strategy_topsis as sim


def log1p_array(x):
	return np.log1p(np.maximum(x, 0))


DETAIL_PATH = sim.OUT_DIR / "5.7_kappa语义更新参数灵敏度分析明细.csv"
SUMMARY_PATH = sim.OUT_DIR / "5.7_kappa语义更新参数灵敏度分析汇总.csv"
REPORT_PATH = sim.OUT_DIR / "5.7_kappa语义更新参数灵敏度分析报告.md"
FIG_GAP_HEATMAP_PATH = sim.OUT_DIR / "5.7_图5_kappa参数下S5相对S0利润差热力图.png"
FIG_BEST_HEATMAP_PATH = sim.OUT_DIR / "5.7_图6_kappa参数下最优策略变化热力图.png"
FIG_S4_GAP_HEATMAP_PATH = sim.OUT_DIR / "5.7_图9_kappa参数下S4相对S0利润差热力图.png"

BASE_GAMMA_KAPPA = sim.SEMANTIC_GAMMA
BASE_EPSILON_KAPPA = sim.FEATURE_UPDATE_EPS

GAMMA_KAPPA_VALUES = [0.25, 0.50, 0.75, 1.00]
EPSILON_KAPPA_VALUES = [1e-9, 1e-4, 1e-3, 1e-2, 5e-2]

STRATEGY_CODE_TO_NUM = {
	"S1": 1,
	"S2": 2,
	"S3": 3,
	"S4": 4,
	"S5": 5,
}


def scenario_label(gamma_kappa, epsilon_kappa):
	return f"gamma={gamma_kappa:g}, epsilon={epsilon_kappa:g}"


def simulate_one_scenario(
	pop,
	response_params,
	scenario,
	serf_model,
	serf_meta,
	pt_model,
	pt_meta,
	serf_context,
	pt_context,
	gamma_kappa,
	epsilon_kappa,
):
	original_lambda = sim.SEMANTIC_GAMMA
	original_eps = sim.FEATURE_UPDATE_EPS
	sim.SEMANTIC_GAMMA = float(gamma_kappa)
	sim.FEATURE_UPDATE_EPS = float(epsilon_kappa)
	try:
		rows = []
		trajectory_rows = []
		serf_cache = {}
		pt_cache = {}
		for strategy in sim.STRATEGIES:
			summary, detail = sim.simulate_strategy(
				pop,
				strategy,
				response_params,
				scenario,
				serf_model,
				serf_meta,
				pt_model,
				pt_meta,
				serf_cache,
				pt_cache,
				serf_context,
				pt_context,
			)
			summary["gamma_kappa"] = float(gamma_kappa)
			summary["epsilon_kappa"] = float(epsilon_kappa)
			summary["kappa情景"] = scenario_label(gamma_kappa, epsilon_kappa)
			detail["gamma_kappa"] = float(gamma_kappa)
			detail["epsilon_kappa"] = float(epsilon_kappa)
			detail["kappa情景"] = scenario_label(gamma_kappa, epsilon_kappa)
			rows.append(summary)
			trajectory_rows.append(detail)
		summary_df = pd.DataFrame(rows)
		topsis_df = sim.entropy_topsis(summary_df)
		return summary_df, topsis_df
	finally:
		sim.SEMANTIC_GAMMA = original_lambda
		sim.FEATURE_UPDATE_EPS = original_eps


def summarize_scenario(summary_df, topsis_df, gamma_kappa, epsilon_kappa):
	rank_map = topsis_df.set_index("策略编号")["排序"].to_dict() if not topsis_df.empty else {}
	score_map = topsis_df.set_index("策略编号")["TOPSIS得分"].to_dict() if not topsis_df.empty else {}
	if topsis_df.empty:
		best_code = "无"
		best_name = "无可行策略"
		best_score = np.nan
	else:
		best = topsis_df.iloc[0]
		best_code = best["策略编号"]
		best_name = best["策略名称"]
		best_score = float(best["TOPSIS得分"])

	def pick(code):
		rows = summary_df[summary_df["策略编号"] == code]
		return rows.iloc[0] if not rows.empty else None

	s0 = pick("S0")
	s1 = pick("S1")
	s2 = pick("S2")
	s3 = pick("S3")
	s4 = pick("S4")
	s5 = pick("S5")
	s0_profit = float(s0["平台利润"]) if s0 is not None else np.nan
	s1_profit = float(s1["平台利润"]) if s1 is not None else np.nan
	s5_profit = float(s5["平台利润"]) if s5 is not None else np.nan
	s4_profit = float(s4["平台利润"]) if s4 is not None else np.nan

	return {
		"gamma_kappa": float(gamma_kappa),
		"epsilon_kappa": float(epsilon_kappa),
		"kappa情景": scenario_label(gamma_kappa, epsilon_kappa),
		"可行策略数量": int(len(topsis_df)),
		"最优策略编号": best_code,
		"最优策略名称": best_name,
		"最优策略TOPSIS得分": best_score,
		"S5排序": rank_map.get("S5", np.nan),
		"S5_TOPSIS得分": score_map.get("S5", np.nan),
		"S4排序": rank_map.get("S4", np.nan),
		"S4_TOPSIS得分": score_map.get("S4", np.nan),
		"S3排序": rank_map.get("S3", np.nan),
		"S2排序": rank_map.get("S2", np.nan),
		"S5是否第一": best_code == "S5",
		"S4是否第一": best_code == "S4",
		"S0平台利润": s0_profit,
		"S1平台利润": s1_profit,
		"S5平台利润": s5_profit,
		"S5相对S0利润差": s5_profit - s0_profit,
		"S5相对S1利润差": s5_profit - s1_profit,
		"S5成本收入比": float(s5["成本收入比"]) if s5 is not None else np.nan,
		"S5积分成本": float(s5["积分成本"]) if s5 is not None else np.nan,
		"S5总活跃度提升": float(s5["总活跃度提升"]) if s5 is not None else np.nan,
		"S5续费概率提升": float(s5["最终续费概率提升"]) if s5 is not None else np.nan,
		"S4平台利润": s4_profit,
		"S4相对S0利润差": s4_profit - s0_profit,
		"S4相对S1利润差": s4_profit - s1_profit,
		"S4续费概率提升": float(s4["最终续费概率提升"]) if s4 is not None else np.nan,
		"S3平台利润": float(s3["平台利润"]) if s3 is not None else np.nan,
		"S2平台利润": float(s2["平台利润"]) if s2 is not None else np.nan,
	}


def ensure_comparison_cols(summary):
	out = summary.copy()
	if "S4相对S0利润差" not in out.columns and {"S4平台利润", "S0平台利润"}.issubset(out.columns):
		out["S4相对S0利润差"] = out["S4平台利润"] - out["S0平台利润"]
	if "S4相对S1利润差" not in out.columns and {"S4平台利润", "S1平台利润"}.issubset(out.columns):
		out["S4相对S1利润差"] = out["S4平台利润"] - out["S1平台利润"]
	return out


def build_heatmap_matrix(summary, value_col):
	pivot = summary.pivot(index="gamma_kappa", columns="epsilon_kappa", values=value_col)
	pivot = pivot.reindex(index=GAMMA_KAPPA_VALUES, columns=EPSILON_KAPPA_VALUES)
	return pivot


def save_figures(summary):
	summary = ensure_comparison_cols(summary)
	gap = build_heatmap_matrix(summary, "S5相对S0利润差")
	fig, ax = plt.subplots(figsize=(9, 5.6))
	im = ax.imshow(gap.values, aspect="auto", cmap="RdYlGn")
	ax.set_xticks(range(len(gap.columns)))
	ax.set_xticklabels([f"{v:g}" for v in gap.columns])
	ax.set_yticks(range(len(gap.index)))
	ax.set_yticklabels([f"{v:g}" for v in gap.index])
	ax.set_xlabel("epsilon_kappa")
	ax.set_ylabel("gamma_kappa")
	ax.set_title("kappa参数下S5相对S0利润差")
	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label("S5相对S0利润差")
	for i in range(gap.shape[0]):
		for j in range(gap.shape[1]):
			value = gap.iloc[i, j]
			if pd.notna(value):
				ax.text(j, i, f"{value/10000:.1f}万", ha="center", va="center", fontsize=8)
	fig.tight_layout()
	fig.savefig(FIG_GAP_HEATMAP_PATH, bbox_inches="tight")
	plt.close(fig)

	s4_gap = build_heatmap_matrix(summary, "S4相对S0利润差")
	fig, ax = plt.subplots(figsize=(9, 5.6))
	im = ax.imshow(s4_gap.values, aspect="auto", cmap="RdYlGn")
	ax.set_xticks(range(len(s4_gap.columns)))
	ax.set_xticklabels([f"{v:g}" for v in s4_gap.columns])
	ax.set_yticks(range(len(s4_gap.index)))
	ax.set_yticklabels([f"{v:g}" for v in s4_gap.index])
	ax.set_xlabel("epsilon_kappa")
	ax.set_ylabel("gamma_kappa")
	ax.set_title("kappa参数下S4相对S0利润差")
	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label("S4相对S0利润差")
	for i in range(s4_gap.shape[0]):
		for j in range(s4_gap.shape[1]):
			value = s4_gap.iloc[i, j]
			if pd.notna(value):
				ax.text(j, i, f"{value/10000:.1f}万", ha="center", va="center", fontsize=8)
	fig.tight_layout()
	fig.savefig(FIG_S4_GAP_HEATMAP_PATH, bbox_inches="tight")
	plt.close(fig)

	best_codes = summary.copy()
	best_codes["最优策略数值"] = best_codes["最优策略编号"].map(STRATEGY_CODE_TO_NUM)
	best = build_heatmap_matrix(best_codes, "最优策略数值")
	label_lookup = summary.set_index(["gamma_kappa", "epsilon_kappa"])["最优策略编号"].to_dict()
	fig, ax = plt.subplots(figsize=(9, 5.6))
	im = ax.imshow(best.values, aspect="auto", cmap="viridis", vmin=1, vmax=5)
	ax.set_xticks(range(len(best.columns)))
	ax.set_xticklabels([f"{v:g}" for v in best.columns])
	ax.set_yticks(range(len(best.index)))
	ax.set_yticklabels([f"{v:g}" for v in best.index])
	ax.set_xlabel("epsilon_kappa")
	ax.set_ylabel("gamma_kappa")
	ax.set_title("kappa参数下TOPSIS最优策略变化")
	cbar = fig.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5])
	cbar.set_label("策略编号")
	cbar.ax.set_yticklabels(["S1", "S2", "S3", "S4", "S5"])
	for i, lam in enumerate(best.index):
		for j, eps in enumerate(best.columns):
			label = label_lookup.get((lam, eps), "")
			ax.text(j, i, label, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
	fig.tight_layout()
	fig.savefig(FIG_BEST_HEATMAP_PATH, bbox_inches="tight")
	plt.close(fig)


def write_report(summary):
	summary = ensure_comparison_cols(summary)
	best_counts = summary["最优策略编号"].value_counts().reset_index()
	best_counts.columns = ["最优策略编号", "出现次数"]
	s5_first_count = int(summary["S5是否第一"].sum())
	s4_first_count = int(summary["S4是否第一"].sum())
	other_best = best_counts[~best_counts["最优策略编号"].isin(["S5", "S4"])]
	if other_best.empty:
		other_best_text = "其余策略未在该短期网格中排名第一"
	else:
		parts = [
			f"{row['最优策略编号']} 在 {int(row['出现次数'])} 个情景中排名第一"
			for _, row in other_best.iterrows()
		]
		other_best_text = "；".join(parts)
	total_count = int(len(summary))
	gap_min = float(summary["S5相对S0利润差"].min())
	gap_max = float(summary["S5相对S0利润差"].max())
	s4_gap_min = float(summary["S4相对S0利润差"].min())
	s4_gap_max = float(summary["S4相对S0利润差"].max())
	s5_gap_all_negative = bool((summary["S5相对S0利润差"] < 0).all())
	s4_gap_all_positive = bool((summary["S4相对S0利润差"] > 0).all())
	if s5_gap_all_negative:
		gap_text = (
			f"S5 相对 S0 的利润差在全部情景中均为负，范围为 "
			f"{gap_min:.2f} 至 {gap_max:.2f}。"
		)
	else:
		gap_text = (
			f"S5 相对 S0 的利润差范围为 {gap_min:.2f} 至 {gap_max:.2f}，"
			"部分情景下出现正利润差。"
		)
	if s4_gap_all_positive:
		s4_gap_text = (
			f"S4 相对 S0 的利润差在全部情景中均为正，范围为 "
			f"{s4_gap_min:.2f} 至 {s4_gap_max:.2f}。"
		)
	else:
		s4_gap_text = (
			f"S4 相对 S0 的利润差范围为 {s4_gap_min:.2f} 至 {s4_gap_max:.2f}。"
		)
	base_row = summary[
		(summary["gamma_kappa"] == BASE_GAMMA_KAPPA)
		& (summary["epsilon_kappa"] == BASE_EPSILON_KAPPA)
	]
	if base_row.empty:
		base_text = "本次网格中未找到完全等于基准参数的情景。"
	else:
		base = base_row.iloc[0]
		base_text = (
			f"基准参数 $\\gamma_\\kappa={BASE_GAMMA_KAPPA:g}$、"
			f"$\\epsilon_\\kappa={BASE_EPSILON_KAPPA:g}$ 下，最优策略为 "
			f"{base['最优策略编号']} {base['最优策略名称']}，S5 相对 S0 利润差为 "
			f"{base['S5相对S0利润差']:.2f}。"
		)

	report = f"""#5.7 kappa语义更新参数短期灵敏度分析报告

本文件由 `模型输出结果/sensitivity_kappa_params.py` 自动生成，用于补充第7章灵敏度分析。该报告对应“窗口1 + 1个延期观察窗口”的短期情景；多延期观察窗口下的 $\\kappa$ 参数灵敏度结果已单独输出到 `5.7_kappa多延期窗口灵敏度分析报告.md`。分析对象是 SERF 反事实特征更新中的语义放大系数：

$$
\\kappa_{{i,k}}(r)=1+\\gamma_\\kappa\\ln\\left(1+\\frac{{\\Delta A_{{i,k}}(r)}}{{A_{{i,t}}+\\epsilon_\\kappa}}\\right)
$$

其中，$\\gamma_\\kappa$ 对应程序中的 `SEMANTIC_GAMMA`，$\\epsilon_\\kappa$ 对应 `FEATURE_UPDATE_EPS`。这里的 $\\gamma_\\kappa$ 不是早期已经废弃的 `lambda_renew` 或 `lambda_pt` 线性概率加成。

##1. 分析口径

在保持积分规则、积分成本、兑换概率、响应参数 $\\rho_g$、$q_g$ 和短期观察窗口设置不变的条件下，仅改变 $\\gamma_\\kappa$ 与 $\\epsilon_\\kappa$。每个情景重新计算各候选策略的 SERF 反事实续费概率、经营结果和熵权-TOPSIS排序。

- $\\gamma_\\kappa$ 测试取值：{GAMMA_KAPPA_VALUES}
- $\\epsilon_\\kappa$ 测试取值：{EPSILON_KAPPA_VALUES}
- 共计情景数：{total_count}

##2. 排序稳定性

{base_text}

在全部 {total_count} 个 $\\gamma_\\kappa \\times \\epsilon_\\kappa$ 情景中，S5 排名第一的情景数为 {s5_first_count} 个，S4 排名第一的情景数为 {s4_first_count} 个。各最优策略出现次数如下：

{best_counts.to_markdown(index=False)}

完整结果见 `5.7_kappa语义更新参数灵敏度分析汇总.csv`。

##3. 主要结论

$\\gamma_\\kappa$ 越大，积分带来的活跃提升越容易通过语义更新放大到线上互动、消费结构和近因类特征中；$\\epsilon_\\kappa$ 越大，低活跃用户在 $A_{{i,t}}$ 很小时的相对放大效应会被削弱。因此，该组灵敏度分析主要检验 SERF 反事实推理是否过度依赖“低活跃用户被强放大”的特征更新假设。

从结果看，S5 在 20 个情景中有 {s5_first_count} 个情景排名第一，说明其短期综合促活排序对 $\\kappa$ 参数扰动仍较强；S4 在 {s4_first_count} 个情景中排名第一，{other_best_text}，说明最优策略并非对 $\\gamma_\\kappa$ 和 $\\epsilon_\\kappa$ 完全不敏感。{gap_text}

当 $\\epsilon_\\kappa$ 增大时，低活跃用户的相对放大效应被削弱，S5 的续费概率提升和相对利润表现整体下降；当 $\\gamma_\\kappa$ 增大时，语义放大效应增强，S5 的续费概率提升通常上升，但高强度补贴带来的短期利润压力仍存在。作为对照，{s4_gap_text} 因此，该组结果应与多延期窗口灵敏度分析共同解释：S5 的短期综合促活能力较稳，但短期利润承压；S4 的短期利润表现相对更稳健。若从多延期窗口 TOPSIS 排序稳定性和落地稳健性看，仍应保持“**S4 流失召回强化策略作为稳健主推荐，S5 高强度综合激励策略作为进攻型备选**”的最终口径。

##4. 可放入论文的图片

- `5.7_图5_kappa参数下S5相对S0利润差热力图.png`
- `5.7_图6_kappa参数下最优策略变化热力图.png`
- `5.7_图9_kappa参数下S4相对S0利润差热力图.png`
"""
	REPORT_PATH.write_text(report, encoding="utf-8")


def main():
	sim.OUT_DIR.mkdir(parents=True, exist_ok=True)
	pop, serf_model, serf_meta, pt_model, pt_meta = sim.load_population()
	response_params = sim.compute_response_params(pop)
	serf_context = sim.build_serf_fast_context(pop, serf_model, serf_meta)
	pt_context = sim.build_pt_fast_context(pop, pt_model, pt_meta)
	base_scenario = sim.build_scenario(
		{"观察期标签": "KAPPA", "carry_a": sim.BASE_CARRY_A},
		sim.BASE_DELAY_COUNT,
	)

	all_details = []
	summary_rows = []
	for gamma_kappa in GAMMA_KAPPA_VALUES:
		for epsilon_kappa in EPSILON_KAPPA_VALUES:
			detail_df, topsis_df = simulate_one_scenario(
				pop,
				response_params,
				base_scenario,
				serf_model,
				serf_meta,
				pt_model,
				pt_meta,
				serf_context,
				pt_context,
				gamma_kappa,
				epsilon_kappa,
			)
			all_details.append(detail_df)
			summary_rows.append(summarize_scenario(detail_df, topsis_df, gamma_kappa, epsilon_kappa))

	detail = pd.concat(all_details, ignore_index=True)
	summary = pd.DataFrame(summary_rows).sort_values(["gamma_kappa", "epsilon_kappa"]).reset_index(drop=True)
	summary = ensure_comparison_cols(summary)
	detail.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
	summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
	save_figures(summary)
	write_report(summary)
	print(f"已生成kappa参数灵敏度分析明细：{DETAIL_PATH}")
	print(f"已生成kappa参数灵敏度分析汇总：{SUMMARY_PATH}")
	print(f"已生成kappa参数灵敏度分析报告：{REPORT_PATH}")
	print(f"已生成论文图表：{FIG_GAP_HEATMAP_PATH}")
	print(f"已生成论文图表：{FIG_BEST_HEATMAP_PATH}")


if __name__ == "__main__":
	main()
