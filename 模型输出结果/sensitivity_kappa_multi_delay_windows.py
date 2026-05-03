# -*- coding: utf-8 -*-
"""kappa 参数的多延期观察窗口灵敏度分析。

该脚本补充 sensitivity_kappa_params.py 的短期结果，固定基准
活跃延续系数 gamma_A=0.35，在多个 delay_window_count 下检验
gamma_kappa 与 epsilon_kappa 对累计利润差和 TOPSIS 排序的影响。

这里的 gamma_kappa 对应 simulate_points_strategy_topsis.py 中的
SEMANTIC_GAMMA，不是早期废弃的 lambda_renew 或 lambda_pt。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import simulate_points_strategy_topsis as sim


def log1p_array(x):
	return np.log1p(np.maximum(x, 0))


SUMMARY_PATH = sim.OUT_DIR / "5.7_kappa多延期窗口灵敏度分析汇总.csv"
PARTIAL_PATH = sim.OUT_DIR / "5.7_kappa多延期窗口灵敏度分析汇总_partial.csv"
REPORT_PATH = sim.OUT_DIR / "5.7_kappa多延期窗口灵敏度分析报告.md"
FIG_CROSS_PATH = sim.OUT_DIR / "5.7_图7_kappa多延期窗口S5首次反超窗口热力图.png"
FIG_S4_COUNT_PATH = sim.OUT_DIR / "5.7_图8_kappa多延期窗口S4最优次数热力图.png"
FIG_S4_CROSS_PATH = sim.OUT_DIR / "5.7_图10_kappa多延期窗口S4首次反超窗口热力图.png"

BASE_CARRY_A = sim.BASE_CARRY_A
DELAY_WINDOW_COUNTS = sim.DELAY_WINDOW_COUNTS
GAMMA_KAPPA_VALUES = [0.25, 0.50, 0.75, 1.00]
EPSILON_KAPPA_VALUES = [1e-9, 1e-4, 1e-3, 1e-2, 5e-2]


def scenario_label(gamma_kappa, epsilon_kappa):
	return f"gamma={gamma_kappa:g}, epsilon={epsilon_kappa:g}"


def set_kappa_params(gamma_kappa, epsilon_kappa):
	sim.SEMANTIC_GAMMA = float(gamma_kappa)
	sim.FEATURE_UPDATE_EPS = float(epsilon_kappa)


def pick_strategy(group, code):
	rows = group[group["策略编号"] == code]
	if rows.empty:
		return None
	return rows.iloc[0]


def summarize_group(summary_df, gamma_kappa, epsilon_kappa, delay_count):
	topsis = sim.entropy_topsis(summary_df.copy())
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

	s0 = pick_strategy(summary_df, "S0")
	s1 = pick_strategy(summary_df, "S1")
	s2 = pick_strategy(summary_df, "S2")
	s3 = pick_strategy(summary_df, "S3")
	s4 = pick_strategy(summary_df, "S4")
	s5 = pick_strategy(summary_df, "S5")

	s0_profit = float(s0["平台利润"]) if s0 is not None else np.nan
	s1_profit = float(s1["平台利润"]) if s1 is not None else np.nan
	s5_profit = float(s5["平台利润"]) if s5 is not None else np.nan
	s4_profit = float(s4["平台利润"]) if s4 is not None else np.nan

	return {
		"gamma_kappa": float(gamma_kappa),
		"epsilon_kappa": float(epsilon_kappa),
		"kappa情景": scenario_label(gamma_kappa, epsilon_kappa),
		"延续活跃系数": BASE_CARRY_A,
		"delay_window_count": int(delay_count),
		"总窗口数": int(1 + delay_count),
		"可行策略数量": int(len(topsis)),
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
		"S5私教购买概率提升": float(s5["最终私教购买概率提升"]) if s5 is not None else np.nan,
		"S5私教收入提升": float(s5["私教收入提升"]) if s5 is not None else np.nan,
		"S4平台利润": s4_profit,
		"S4相对S0利润差": s4_profit - s0_profit,
		"S4相对S1利润差": s4_profit - s1_profit,
		"S4续费概率提升": float(s4["最终续费概率提升"]) if s4 is not None else np.nan,
		"S4私教购买概率提升": float(s4["最终私教购买概率提升"]) if s4 is not None else np.nan,
		"S4私教收入提升": float(s4["私教收入提升"]) if s4 is not None else np.nan,
		"S3平台利润": float(s3["平台利润"]) if s3 is not None else np.nan,
		"S3私教购买概率提升": float(s3["最终私教购买概率提升"]) if s3 is not None else np.nan,
		"S3私教收入提升": float(s3["私教收入提升"]) if s3 is not None else np.nan,
		"S2平台利润": float(s2["平台利润"]) if s2 is not None else np.nan,
	}


def ensure_comparison_cols(summary):
	out = summary.copy()
	if "S4相对S0利润差" not in out.columns and {"S4平台利润", "S0平台利润"}.issubset(out.columns):
		out["S4相对S0利润差"] = out["S4平台利润"] - out["S0平台利润"]
	if "S4相对S1利润差" not in out.columns and {"S4平台利润", "S1平台利润"}.issubset(out.columns):
		out["S4相对S1利润差"] = out["S4平台利润"] - out["S1平台利润"]
	return out


def run_one_kappa_grid(pop, response_params, serf_model, serf_meta, pt_model, pt_meta, serf_context, pt_context, gamma_kappa, epsilon_kappa):
	original_lambda = sim.SEMANTIC_GAMMA
	original_eps = sim.FEATURE_UPDATE_EPS
	set_kappa_params(gamma_kappa, epsilon_kappa)
	try:
		rows = []
		serf_cache = {}
		pt_cache = {}
		for delay_count in DELAY_WINDOW_COUNTS:
			print(
				f"  delay_window_count={delay_count}",
				flush=True,
			)
			scenario = sim.build_scenario(
				{"观察期标签": "KAPPA_MULTI", "carry_a": BASE_CARRY_A},
				delay_count,
			)
			summaries = []
			for strategy in sim.STRATEGIES:
				summary, _ = sim.simulate_strategy(
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
				summaries.append(summary)
			rows.append(summarize_group(pd.DataFrame(summaries), gamma_kappa, epsilon_kappa, delay_count))
		return rows
	finally:
		sim.SEMANTIC_GAMMA = original_lambda
		sim.FEATURE_UPDATE_EPS = original_eps


def first_cross_table(summary):
	summary = ensure_comparison_cols(summary)
	rows = []
	for (gamma_kappa, epsilon_kappa), group in summary.groupby(["gamma_kappa", "epsilon_kappa"], sort=True):
		group = group.sort_values("delay_window_count")
		over = group[group["S5相对S0利润差"] > 0]
		if over.empty:
			first_delay = np.nan
			first_total = np.nan
			first_gap = np.nan
			has_cross = False
		else:
			first = over.iloc[0]
			first_delay = int(first["delay_window_count"])
			first_total = int(first["总窗口数"])
			first_gap = float(first["S5相对S0利润差"])
			has_cross = True
		s4_over = group[group["S4相对S0利润差"] > 0]
		if s4_over.empty:
			s4_first_delay = np.nan
			s4_first_total = np.nan
			s4_first_gap = np.nan
			s4_has_cross = False
		else:
			s4_first = s4_over.iloc[0]
			s4_first_delay = int(s4_first["delay_window_count"])
			s4_first_total = int(s4_first["总窗口数"])
			s4_first_gap = float(s4_first["S4相对S0利润差"])
			s4_has_cross = True
		rows.append({
			"gamma_kappa": float(gamma_kappa),
			"epsilon_kappa": float(epsilon_kappa),
			"是否出现利润反超S0": has_cross,
			"首次反超延期窗口数": first_delay,
			"首次反超总窗口数": first_total,
			"首次反超时利润差": first_gap,
			"S5第一次数": int(group["S5是否第一"].sum()),
			"S4第一次数": int(group["S4是否第一"].sum()),
			"最大测试窗口S5相对S0利润差": float(group.iloc[-1]["S5相对S0利润差"]),
			"最大测试窗口S5排序": group.iloc[-1]["S5排序"],
			"S4是否出现利润反超S0": s4_has_cross,
			"S4首次反超延期窗口数": s4_first_delay,
			"S4首次反超总窗口数": s4_first_total,
			"S4首次反超时利润差": s4_first_gap,
			"最大测试窗口S4相对S0利润差": float(group.iloc[-1]["S4相对S0利润差"]),
			"最大测试窗口S4排序": group.iloc[-1]["S4排序"],
		})
	return pd.DataFrame(rows)


def save_figures(summary):
	summary = ensure_comparison_cols(summary)
	cross = first_cross_table(summary)
	cross_matrix = cross.pivot(index="gamma_kappa", columns="epsilon_kappa", values="首次反超总窗口数")
	cross_matrix = cross_matrix.reindex(index=GAMMA_KAPPA_VALUES, columns=EPSILON_KAPPA_VALUES)
	plot_matrix = cross_matrix.fillna(0)

	fig, ax = plt.subplots(figsize=(9, 5.6))
	im = ax.imshow(plot_matrix.values, aspect="auto", cmap="YlGnBu")
	ax.set_xticks(range(len(plot_matrix.columns)))
	ax.set_xticklabels([f"{v:g}" for v in plot_matrix.columns])
	ax.set_yticks(range(len(plot_matrix.index)))
	ax.set_yticklabels([f"{v:g}" for v in plot_matrix.index])
	ax.set_xlabel("epsilon_kappa")
	ax.set_ylabel("gamma_kappa")
	ax.set_title("kappa多延期窗口下S5首次反超S0的总窗口数")
	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label("首次反超总窗口数，0表示未反超")
	for i in range(plot_matrix.shape[0]):
		for j in range(plot_matrix.shape[1]):
			raw = cross_matrix.iloc[i, j]
			label = "未" if pd.isna(raw) else str(int(raw))
			ax.text(j, i, label, ha="center", va="center", fontsize=9)
	fig.tight_layout()
	fig.savefig(FIG_CROSS_PATH, bbox_inches="tight")
	plt.close(fig)

	s4_cross_matrix = cross.pivot(index="gamma_kappa", columns="epsilon_kappa", values="S4首次反超总窗口数")
	s4_cross_matrix = s4_cross_matrix.reindex(index=GAMMA_KAPPA_VALUES, columns=EPSILON_KAPPA_VALUES)
	s4_plot_matrix = s4_cross_matrix.fillna(0)
	fig, ax = plt.subplots(figsize=(9, 5.6))
	im = ax.imshow(s4_plot_matrix.values, aspect="auto", cmap="YlOrBr")
	ax.set_xticks(range(len(s4_plot_matrix.columns)))
	ax.set_xticklabels([f"{v:g}" for v in s4_plot_matrix.columns])
	ax.set_yticks(range(len(s4_plot_matrix.index)))
	ax.set_yticklabels([f"{v:g}" for v in s4_plot_matrix.index])
	ax.set_xlabel("epsilon_kappa")
	ax.set_ylabel("gamma_kappa")
	ax.set_title("kappa多延期窗口下S4首次反超S0的总窗口数")
	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label("首次反超总窗口数，0表示未反超")
	for i in range(s4_plot_matrix.shape[0]):
		for j in range(s4_plot_matrix.shape[1]):
			raw = s4_cross_matrix.iloc[i, j]
			label = "未" if pd.isna(raw) else str(int(raw))
			ax.text(j, i, label, ha="center", va="center", fontsize=9)
	fig.tight_layout()
	fig.savefig(FIG_S4_CROSS_PATH, bbox_inches="tight")
	plt.close(fig)

	s4_matrix = cross.pivot(index="gamma_kappa", columns="epsilon_kappa", values="S4第一次数")
	s4_matrix = s4_matrix.reindex(index=GAMMA_KAPPA_VALUES, columns=EPSILON_KAPPA_VALUES)
	fig, ax = plt.subplots(figsize=(9, 5.6))
	im = ax.imshow(s4_matrix.values, aspect="auto", cmap="PuBuGn", vmin=0, vmax=len(DELAY_WINDOW_COUNTS))
	ax.set_xticks(range(len(s4_matrix.columns)))
	ax.set_xticklabels([f"{v:g}" for v in s4_matrix.columns])
	ax.set_yticks(range(len(s4_matrix.index)))
	ax.set_yticklabels([f"{v:g}" for v in s4_matrix.index])
	ax.set_xlabel("epsilon_kappa")
	ax.set_ylabel("gamma_kappa")
	ax.set_title("kappa多延期窗口下S4排名第一次数")
	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label("S4排名第一次数")
	for i in range(s4_matrix.shape[0]):
		for j in range(s4_matrix.shape[1]):
			value = s4_matrix.iloc[i, j]
			if pd.notna(value):
				ax.text(j, i, str(int(value)), ha="center", va="center", fontsize=9)
	fig.tight_layout()
	fig.savefig(FIG_S4_COUNT_PATH, bbox_inches="tight")
	plt.close(fig)


def write_report(summary):
	summary = ensure_comparison_cols(summary)
	cross = first_cross_table(summary)
	total = int(len(summary))
	kappa_count = int(len(cross))
	best_counts = summary["最优策略编号"].value_counts().reset_index()
	best_counts.columns = ["最优策略编号", "出现次数"]
	cross_count = int(cross["是否出现利润反超S0"].sum())
	s4_cross_count = int(cross["S4是否出现利润反超S0"].sum())
	s5_first_total = int(summary["S5是否第一"].sum())
	s4_first_total = int(summary["S4是否第一"].sum())
	s4_gap_min = float(summary["S4相对S0利润差"].min())
	s4_gap_max = float(summary["S4相对S0利润差"].max())

	report = f"""#5.7 kappa多延期观察窗口灵敏度分析报告

本文件由 `模型输出结果/sensitivity_kappa_multi_delay_windows.py` 自动生成，用于补充 $\\kappa$ 语义更新参数在多个延期观察窗口下的稳定性检验。

##1. 分析口径

本分析固定活跃延续系数 $\\gamma_A={BASE_CARRY_A}$，测试 $\\gamma_\\kappa \\times \\epsilon_\\kappa$ 参数组合在多个延期观察窗口下的累计利润差和 TOPSIS 排序变化。延期窗口数沿用主模型设置：

- `delay_window_count` = {DELAY_WINDOW_COUNTS}
- $\\gamma_\\kappa$ = {GAMMA_KAPPA_VALUES}
- $\\epsilon_\\kappa$ = {EPSILON_KAPPA_VALUES}
- 共计 $\\kappa$ 参数组合数 = {kappa_count}
- 共计 TOPSIS 组合数 = {total}

这里的 $\\gamma_\\kappa$ 对应程序中的 `SEMANTIC_GAMMA`，不是旧模型中的 `lambda_renew` 或 `lambda_pt` 线性概率加成。

##2. 利润反超窗口

在 {kappa_count} 个 $\\kappa$ 参数组合中，有 {cross_count} 个组合在测试的多延期窗口内出现 S5 相对 S0 的累计利润反超。各参数组合的首次反超窗口如下：

{cross.to_markdown(index=False)}

作为对照，S4 相对 S0 的累计利润反超在 {s4_cross_count} 个 $\\kappa$ 参数组合中出现。S4 相对 S0 的利润差在全部“$\\kappa$ 参数 × 延期窗口数”组合中的范围为 {s4_gap_min:.2f} 至 {s4_gap_max:.2f}。

##3. TOPSIS排序稳定性

在全部 {total} 个“$\\kappa$ 参数 × 延期窗口数”组合中，S5 排名第一的组合数为 {s5_first_total} 个，S4 排名第一的组合数为 {s4_first_total} 个。各最优策略出现次数如下：

{best_counts.to_markdown(index=False)}

完整结果见 `5.7_kappa多延期窗口灵敏度分析汇总.csv`。

##4. 主要结论

多延期窗口结果用于回答：即使改变 $\\gamma_\\kappa$ 与 $\\epsilon_\\kappa$，S5 是否仍会在较长观察窗口中弥补短期补贴成本，S4 相对无积分基准 S0 是否也存在利润改善，以及 S4 是否仍具有更稳定的综合排序。当前脚本调用主策略模拟中的 SERF 续费反事实推理和增强私教购买模型反事实推理，因此 TOPSIS 中已包含 `最终私教购买概率提升`，该指标不再保持基线或恒为0。

若 $\\epsilon_\\kappa$ 较小或 $\\gamma_\\kappa$ 较大，低活跃用户的相对活跃提升更容易被放大到 SERF 反事实特征中，S5 的长期累计利润更容易改善；若 $\\epsilon_\\kappa$ 较大，低活跃用户的相对放大效应被削弱，S5 的累计利润反超可能推迟或消失。

该分析仍属于情景模拟。它不能证明积分机制的真实长期因果效果，只能说明在不同 SERF 与增强私教模型反事实特征更新假设下，策略排序和累计利润差是否稳健。最终论文口径仍应与主灵敏度分析保持一致：**S4 流失召回强化策略作为稳健主推荐，S5 高强度综合激励策略作为进攻型备选**。

##5. 可放入论文的图片

- `5.7_图7_kappa多延期窗口S5首次反超窗口热力图.png`
- `5.7_图8_kappa多延期窗口S4最优次数热力图.png`
- `5.7_图10_kappa多延期窗口S4首次反超窗口热力图.png`
"""
	REPORT_PATH.write_text(report, encoding="utf-8")


def main():
	sim.OUT_DIR.mkdir(parents=True, exist_ok=True)
	if PARTIAL_PATH.exists():
		PARTIAL_PATH.unlink()
	pop, serf_model, serf_meta, pt_model, pt_meta = sim.load_population()
	response_params = sim.compute_response_params(pop)
	serf_context = sim.build_serf_fast_context(pop, serf_model, serf_meta)
	pt_context = sim.build_pt_fast_context(pop, pt_model, pt_meta)

	all_rows = []
	for gamma_kappa in GAMMA_KAPPA_VALUES:
		for epsilon_kappa in EPSILON_KAPPA_VALUES:
			print(
				f"运行 kappa 多延期情景: gamma={gamma_kappa:g}, epsilon={epsilon_kappa:g}",
				flush=True,
			)
			all_rows.extend(
				run_one_kappa_grid(
					pop,
					response_params,
					serf_model,
					serf_meta,
					pt_model,
					pt_meta,
					serf_context,
					pt_context,
					gamma_kappa,
					epsilon_kappa,
				)
			)
			partial = pd.DataFrame(all_rows).sort_values(
				["gamma_kappa", "epsilon_kappa", "delay_window_count"]
			).reset_index(drop=True)
			partial = ensure_comparison_cols(partial)
			partial.to_csv(PARTIAL_PATH, index=False, encoding="utf-8-sig")
			print(f"已保存中间结果：{PARTIAL_PATH}，行数={len(partial)}", flush=True)
	summary = pd.DataFrame(all_rows).sort_values(
		["gamma_kappa", "epsilon_kappa", "delay_window_count"]
	).reset_index(drop=True)
	summary = ensure_comparison_cols(summary)
	summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
	save_figures(summary)
	write_report(summary)
	print(f"已生成kappa多延期窗口灵敏度分析汇总：{SUMMARY_PATH}")
	print(f"已生成kappa多延期窗口灵敏度分析报告：{REPORT_PATH}")
	print(f"已生成论文图表：{FIG_CROSS_PATH}")
	print(f"已生成论文图表：{FIG_S4_COUNT_PATH}")


if __name__ == "__main__":
	main()
