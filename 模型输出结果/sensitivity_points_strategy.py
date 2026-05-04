# -*- coding: utf-8 -*-
"""积分策略关键参数灵敏度分析。

本脚本复用 simulate_points_strategy_topsis.py 中的 SERF 反事实推理、
策略模拟和熵权 TOPSIS 逻辑，只改变单一情景参数，检验最优策略排序
是否对积分成本、兑换概率、响应假设和经营约束过度敏感。
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import simulate_points_strategy_topsis as sim


def log1p_array(x):
	return np.log1p(np.maximum(x, 0))


DETAIL_PATH = sim.OUT_DIR / "5.7_积分策略灵敏度分析明细.csv"
SUMMARY_PATH = sim.OUT_DIR / "5.7_积分策略灵敏度分析汇总.csv"
REPORT_PATH = sim.OUT_DIR / "5.7_积分策略灵敏度分析报告.md"
FIG_BEST_PATH = sim.OUT_DIR / "5.7_图1_灵敏度情景最优策略变化.png"
FIG_S5_PATH = sim.OUT_DIR / "5.7_图2_S5排名与利润差敏感性.png"

BASE_C_POINT = sim.C_POINT
BASE_ETA = sim.ETA
BASE_REDEEM_SCALE = 1.0
BASE_RHO_SCALE = 1.0
BASE_Q_SCALE = 1.0
BASE_CARRY_A = sim.BASE_CARRY_A


SENSITIVITY_SCENARIOS = [
	{
		"参数类别": "基准",
		"情景": "基准",
		"参数取值": "基准口径",
		"c_point": BASE_C_POINT,
		"redeem_scale": BASE_REDEEM_SCALE,
		"eta": BASE_ETA,
		"rho_scale": BASE_RHO_SCALE,
		"q_scale": BASE_Q_SCALE,
		"carry_a": BASE_CARRY_A,
	},
	{
		"参数类别": "单位积分抵扣金额c",
		"情景": "偏低",
		"参数取值": "c=0.005",
		"c_point": 0.005,
		"redeem_scale": BASE_REDEEM_SCALE,
		"eta": BASE_ETA,
		"rho_scale": BASE_RHO_SCALE,
		"q_scale": BASE_Q_SCALE,
		"carry_a": BASE_CARRY_A,
	},
	{
		"参数类别": "单位积分抵扣金额c",
		"情景": "偏高",
		"参数取值": "c=0.015",
		"c_point": 0.015,
		"redeem_scale": BASE_REDEEM_SCALE,
		"eta": BASE_ETA,
		"rho_scale": BASE_RHO_SCALE,
		"q_scale": BASE_Q_SCALE,
		"carry_a": BASE_CARRY_A,
	},
	{
		"参数类别": "兑换概率",
		"情景": "偏低",
		"参数取值": "redeem_prob x 0.8",
		"c_point": BASE_C_POINT,
		"redeem_scale": 0.8,
		"eta": BASE_ETA,
		"rho_scale": BASE_RHO_SCALE,
		"q_scale": BASE_Q_SCALE,
		"carry_a": BASE_CARRY_A,
	},
	{
		"参数类别": "兑换概率",
		"情景": "偏高",
		"参数取值": "redeem_prob x 1.2",
		"c_point": BASE_C_POINT,
		"redeem_scale": 1.2,
		"eta": BASE_ETA,
		"rho_scale": BASE_RHO_SCALE,
		"q_scale": BASE_Q_SCALE,
		"carry_a": BASE_CARRY_A,
	},
	{
		"参数类别": "成本收入比上限eta",
		"情景": "偏严",
		"参数取值": "eta=0.12",
		"c_point": BASE_C_POINT,
		"redeem_scale": BASE_REDEEM_SCALE,
		"eta": 0.12,
		"rho_scale": BASE_RHO_SCALE,
		"q_scale": BASE_Q_SCALE,
		"carry_a": BASE_CARRY_A,
	},
	{
		"参数类别": "成本收入比上限eta",
		"情景": "偏松",
		"参数取值": "eta=0.24",
		"c_point": BASE_C_POINT,
		"redeem_scale": BASE_REDEEM_SCALE,
		"eta": 0.24,
		"rho_scale": BASE_RHO_SCALE,
		"q_scale": BASE_Q_SCALE,
		"carry_a": BASE_CARRY_A,
	},
	{
		"参数类别": "响应强度rho_g",
		"情景": "保守",
		"参数取值": "rho_g x 0.7",
		"c_point": BASE_C_POINT,
		"redeem_scale": BASE_REDEEM_SCALE,
		"eta": BASE_ETA,
		"rho_scale": 0.7,
		"q_scale": BASE_Q_SCALE,
		"carry_a": BASE_CARRY_A,
	},
	{
		"参数类别": "响应强度rho_g",
		"情景": "乐观",
		"参数取值": "rho_g x 1.3",
		"c_point": BASE_C_POINT,
		"redeem_scale": BASE_REDEEM_SCALE,
		"eta": BASE_ETA,
		"rho_scale": 1.3,
		"q_scale": BASE_Q_SCALE,
		"carry_a": BASE_CARRY_A,
	},
	{
		"参数类别": "参考响应比例q_g",
		"情景": "保守",
		"参数取值": "q_g x 0.7",
		"c_point": BASE_C_POINT,
		"redeem_scale": BASE_REDEEM_SCALE,
		"eta": BASE_ETA,
		"rho_scale": BASE_RHO_SCALE,
		"q_scale": 0.7,
		"carry_a": BASE_CARRY_A,
	},
	{
		"参数类别": "参考响应比例q_g",
		"情景": "乐观",
		"参数取值": "q_g x 1.3",
		"c_point": BASE_C_POINT,
		"redeem_scale": BASE_REDEEM_SCALE,
		"eta": BASE_ETA,
		"rho_scale": BASE_RHO_SCALE,
		"q_scale": 1.3,
		"carry_a": BASE_CARRY_A,
	},
	{
		"参数类别": "活跃延续系数gamma_A",
		"情景": "偏低",
		"参数取值": "gamma_A=0.20",
		"c_point": BASE_C_POINT,
		"redeem_scale": BASE_REDEEM_SCALE,
		"eta": BASE_ETA,
		"rho_scale": BASE_RHO_SCALE,
		"q_scale": BASE_Q_SCALE,
		"carry_a": 0.20,
	},
	{
		"参数类别": "活跃延续系数gamma_A",
		"情景": "偏高",
		"参数取值": "gamma_A=0.50",
		"c_point": BASE_C_POINT,
		"redeem_scale": BASE_REDEEM_SCALE,
		"eta": BASE_ETA,
		"rho_scale": BASE_RHO_SCALE,
		"q_scale": BASE_Q_SCALE,
		"carry_a": 0.50,
	},
]


def scenario_id(row):
	if row["参数类别"] == "基准":
		return "基准"
	return f"{row['参数类别']}:{row['情景']}"


def make_strategies(redeem_scale):
	strategies = deepcopy(sim.STRATEGIES)
	for strategy in strategies:
		strategy["redeem_prob"] = float(np.clip(strategy["redeem_prob"] * redeem_scale, 0, 1))
	return strategies


def compute_response_params(pop, rho_scale=1.0, q_scale=1.0):
	q75 = pop["A_score"].quantile(0.75)
	if q75 <= 0:
		q75 = pop["A_score"].mean() + 1e-5
	target_a = pop.loc[pop["A_score"] >= q75, "A_score"].mean()
	q75_s = pop["S_score"].quantile(0.75)
	if q75_s <= 0:
		q75_s = pop["S_score"].mean() + 1e-5
	target_s = pop.loc[pop["S_score"] >= q75_s, "S_score"].mean()

	params = {}
	for state, base in sim.GROUP_PARAMS.items():
		mask = pop["user_state"] == state
		mean_a = pop.loc[mask, "A_score"].mean() if mask.sum() else pop["A_score"].mean()
		mean_s = pop.loc[mask, "S_score"].mean() if mask.sum() else pop["S_score"].mean()
		rho = float(base["rho"] * rho_scale)
		q = float(np.clip(base["q"] * q_scale, 1e-6, 0.95))
		alpha_base = max(rho * (target_a - mean_a), rho * 2.0)
		alpha_s_base = max(rho * (target_s - mean_s), rho * 2.0)
		beta_p0_base = -np.log(1 - q) / base["pref_p0"]
		beta_p1_base = -np.log(1 - q) / base["pref_p1"]
		params[state] = {
			"alpha_a_base": alpha_base,
			"alpha_s_base": alpha_s_base,
			"beta_p0_base": beta_p0_base,
			"beta_p1_base": beta_p1_base,
			"rho": rho,
			"q": q,
			"pref_p0": base["pref_p0"],
			"pref_p1": base["pref_p1"],
		}
	return params


def summarize_scenario(scenario, detail_df, topsis_df):
	sid = scenario_id(scenario)
	s0 = detail_df[detail_df["策略编号"] == "S0"].iloc[0]
	s5 = detail_df[detail_df["策略编号"] == "S5"].iloc[0]
	s4 = detail_df[detail_df["策略编号"] == "S4"].iloc[0]
	s2 = detail_df[detail_df["策略编号"] == "S2"].iloc[0]

	if topsis_df.empty:
		best_code = "无"
		best_name = "无可行策略"
		best_score = np.nan
		s5_rank = np.nan
		s4_rank = np.nan
		s2_rank = np.nan
		s5_score = np.nan
	else:
		best = topsis_df.iloc[0]
		best_code = best["策略编号"]
		best_name = best["策略名称"]
		best_score = float(best["TOPSIS得分"])
		rank_map = topsis_df.set_index("策略编号")["排序"].to_dict()
		score_map = topsis_df.set_index("策略编号")["TOPSIS得分"].to_dict()
		s5_rank = rank_map.get("S5", np.nan)
		s4_rank = rank_map.get("S4", np.nan)
		s2_rank = rank_map.get("S2", np.nan)
		s5_score = score_map.get("S5", np.nan)

	return {
		"情景ID": sid,
		"参数类别": scenario["参数类别"],
		"情景": scenario["情景"],
		"参数取值": scenario["参数取值"],
		"c": scenario["c_point"],
		"redeem_prob_scale": scenario["redeem_scale"],
		"eta": scenario["eta"],
		"rho_scale": scenario["rho_scale"],
		"q_scale": scenario["q_scale"],
		"gamma_A": scenario["carry_a"],
		"可行策略数量": int(len(topsis_df)),
		"最优策略编号": best_code,
		"最优策略名称": best_name,
		"最优策略TOPSIS得分": best_score,
		"S5排序": s5_rank,
		"S4排序": s4_rank,
		"S2排序": s2_rank,
		"S5_TOPSIS得分": s5_score,
		"S5是否第一": bool(best_code == "S5"),
		"S0平台利润": float(s0["平台利润"]),
		"S5平台利润": float(s5["平台利润"]),
		"S5相对S0利润差": float(s5["平台利润"] - s0["平台利润"]),
		"S5成本收入比": float(s5["成本收入比"]),
		"S5积分成本": float(s5["积分成本"]),
		"S5总活跃度提升": float(s5["总活跃度提升"]),
		"S5续费概率提升": float(s5["最终续费概率提升"]),
		"S4平台利润": float(s4["平台利润"]),
		"S2平台利润": float(s2["平台利润"]),
	}


def run_one_scenario(pop, serf_model, serf_meta, pt_model, pt_meta, serf_context, pt_context, scenario):
	old_c_point = sim.C_POINT
	old_eta = sim.ETA
	try:
		sim.C_POINT = scenario["c_point"]
		sim.ETA = scenario["eta"]

		strategies = make_strategies(scenario["redeem_scale"])
		response_params = compute_response_params(pop, scenario["rho_scale"], scenario["q_scale"])
		model_scenario = sim.build_scenario(
			{"观察期标签": scenario_id(scenario), "carry_a": scenario["carry_a"]},
			sim.BASE_DELAY_COUNT,
		)
		serf_cache = {}
		pt_cache = {}
		rows = []
		for strategy in strategies:
			summary, _ = sim.simulate_strategy(
				pop,
				strategy,
				response_params,
				model_scenario,
				serf_model,
				serf_meta,
				pt_model,
				pt_meta,
				serf_cache,
				pt_cache,
				serf_context,
				pt_context,
			)
			rows.append(summary)
		detail_df = pd.DataFrame(rows)
		for key in ["参数类别", "情景", "参数取值"]:
			detail_df[key] = scenario[key]
		detail_df["情景ID"] = scenario_id(scenario)
		detail_df["c"] = scenario["c_point"]
		detail_df["redeem_prob_scale"] = scenario["redeem_scale"]
		detail_df["eta"] = scenario["eta"]
		detail_df["rho_scale"] = scenario["rho_scale"]
		detail_df["q_scale"] = scenario["q_scale"]
		detail_df["gamma_A"] = scenario["carry_a"]

		topsis_df = sim.entropy_topsis(detail_df)
		if not topsis_df.empty:
			topsis_cols = topsis_df[["策略编号", "TOPSIS得分", "排序"]]
			detail_df = detail_df.merge(topsis_cols, on="策略编号", how="left")
		else:
			detail_df["TOPSIS得分"] = np.nan
			detail_df["排序"] = np.nan

		summary = summarize_scenario(scenario, detail_df, topsis_df)
		return detail_df, summary
	finally:
		sim.C_POINT = old_c_point
		sim.ETA = old_eta


def save_figures(summary_df):
	plot_df = summary_df.copy()
	plot_df["图表标签"] = plot_df["情景ID"]
	colors = {
		"S1": "#7aa6c2",
		"S2": "#8ab17d",
		"S3": "#b07aa1",
		"S4": "#e6a15c",
		"S5": "#5b8db8",
		"无": "#b8b8b8",
	}
	fig, ax = plt.subplots(figsize=(11,5.2))
	x = np.arange(len(plot_df))
	bar_colors = [colors.get(code, "#999999") for code in plot_df["最优策略编号"]]
	ax.bar(x, plot_df["最优策略TOPSIS得分"].fillna(0), color=bar_colors)
	for idx, row in plot_df.iterrows():
		ax.text(idx, max(row["最优策略TOPSIS得分"] if pd.notna(row["最优策略TOPSIS得分"]) else 0, 0) + 0.015,
				row["最优策略编号"], ha="center", va="bottom", fontsize=9)
	ax.set_xticks(x)
	ax.set_xticklabels(plot_df["图表标签"], rotation=35, ha="right")
	ax.set_ylim(0, max(0.8, plot_df["最优策略TOPSIS得分"].max() * 1.15))
	ax.set_ylabel("最优策略TOPSIS得分")
	ax.set_title("灵敏度情景下的最优策略变化")
	fig.tight_layout()
	fig.savefig(FIG_BEST_PATH, bbox_inches="tight")
	plt.close(fig)

	fig, ax1 = plt.subplots(figsize=(11,5.2))
	ax1.bar(x, plot_df["S5相对S0利润差"], color="#d08c60", alpha=0.82, label="S5相对S0利润差")
	ax1.axhline(0, color="#555555", linewidth=1)
	ax1.set_ylabel("S5相对S0利润差")
	ax2 = ax1.twinx()
	rank_y = plot_df["S5排序"].fillna(len(sim.STRATEGIES))
	ax2.plot(x, rank_y, color="#2f5f8f", marker="o", linewidth=2, label="S5排序")
	ax2.set_ylabel("S5排序")
	ax2.invert_yaxis()
	ax1.set_xticks(x)
	ax1.set_xticklabels(plot_df["图表标签"], rotation=35, ha="right")
	ax1.set_title("S5排名与利润差灵敏度")
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
	fig.tight_layout()
	fig.savefig(FIG_S5_PATH, bbox_inches="tight")
	plt.close(fig)


def write_report(summary_df):
	base = summary_df[summary_df["参数类别"] == "基准"].iloc[0]
	s5_first_count = int(summary_df["S5是否第一"].sum())
	total_count = int(len(summary_df))
	best_counts = summary_df["最优策略编号"].value_counts().reset_index()
	best_counts.columns = ["最优策略编号", "出现次数"]
	unstable = summary_df[summary_df["最优策略编号"] != base["最优策略编号"]]
	if unstable.empty:
		change_text = "所有测试情景下最优策略均与基准情景一致。"
	else:
		change_text = "以下情景的最优策略与基准情景不同：\n\n" + unstable[
			["情景ID", "参数取值", "最优策略编号", "最优策略名称", "最优策略TOPSIS得分", "S5排序", "S5相对S0利润差"]
		].to_markdown(index=False)

	report = f"""#5.7 积分策略灵敏度分析报告

本文件由 `模型输出结果/sensitivity_points_strategy.py` 自动生成，用于检验积分策略模拟结果对关键情景参数的敏感性。分析口径与主策略模拟保持一致：用户先根据线下积分系数 $p_0$ 和线上积分系数 $p_1$ 产生行为响应，再按响应后的线上线下活跃特征计算积分、调用状态增强随机森林 SERF 和增强私教购买模型进行反事实推理，不使用早期的 `lambda_renew` 或 `lambda_pt` 线性概率加成。

##1. 分析设置

基准口径为：单位积分抵扣金额 $c={BASE_C_POINT}$，成本收入比上限 $\\eta={BASE_ETA}$，策略层兑换概率不缩放，响应参数 $\\rho_g$ 与 $q_g$ 不缩放，活跃延续系数 $\\gamma_A={BASE_CARRY_A}$。每个灵敏度情景只改变一个参数，并重新计算各候选策略的经营结果和熵权-TOPSIS排序。

需要说明的是，本报告只对应“窗口1 + 1个延期观察窗口”的短期灵敏度分析。多延期观察窗口下的灵敏度结果已单独输出到 `5.7_多延期窗口灵敏度分析报告.md`，用于检验总观察窗口数增加时 S5 相对 S0 的累计利润差和 TOPSIS 排序变化。

##2. 情景汇总

{summary_df[["情景ID", "参数取值", "可行策略数量", "最优策略编号", "最优策略名称", "最优策略TOPSIS得分", "S5排序", "S5相对S0利润差", "S5成本收入比"]].to_markdown(index=False)}

##3. 稳定性结论

基准情景下，最优策略为 **{base['最优策略编号']}：{base['最优策略名称']}**，TOPSIS 得分为 {base['最优策略TOPSIS得分']:.4f}。在全部 {total_count} 个情景中，S5 排名第一的情景数为 {s5_first_count} 个。

{change_text}

从灵敏度结果看，若单位积分抵扣金额或兑换概率升高，积分成本和成本收入比会随之上升，S5 的短期利润压力更明显；若响应强度 $\\rho_g$、参考响应比例 $q_g$ 或活跃延续系数 $\\gamma_A$ 提高，S5 的促活和续费概率改善更明显。若成本约束收紧，则可行策略集合可能缩小，S4 或 S2 可作为更保守的备选方案。

##4. 最优策略出现次数

{best_counts.to_markdown(index=False)}

##5. 可放入论文的图片

- `5.7_图1_灵敏度情景最优策略变化.png`
- `5.7_图2_S5排名与利润差敏感性.png`

需要注意，本文灵敏度分析仍属于基于历史预测模型和响应参数假设的情景模拟，不代表真实积分干预的因果效应。后续若平台积累真实积分发放、兑换和用户响应记录，可进一步用实验或准实验方法识别积分机制的实际因果效果。
"""
	REPORT_PATH.write_text(report, encoding="utf-8")


def main():
	sim.OUT_DIR.mkdir(parents=True, exist_ok=True)
	pop, serf_model, serf_meta, pt_model, pt_meta = sim.load_population()
	serf_context = sim.build_serf_fast_context(pop, serf_model, serf_meta)
	pt_context = sim.build_pt_fast_context(pop, pt_model, pt_meta)

	detail_frames = []
	summary_rows = []
	for scenario in SENSITIVITY_SCENARIOS:
		detail_df, summary = run_one_scenario(
			pop,
			serf_model,
			serf_meta,
			pt_model,
			pt_meta,
			serf_context,
			pt_context,
			scenario,
		)
		detail_frames.append(detail_df)
		summary_rows.append(summary)

	detail_all = pd.concat(detail_frames, ignore_index=True)
	summary_df = pd.DataFrame(summary_rows)
	detail_all.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
	summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
	save_figures(summary_df)
	write_report(summary_df)

	print(f"已生成积分策略灵敏度分析明细：{DETAIL_PATH}")
	print(f"已生成积分策略灵敏度分析汇总：{SUMMARY_PATH}")
	print(f"已生成积分策略灵敏度分析报告：{REPORT_PATH}")
	print(f"已生成论文图表：{FIG_BEST_PATH}")
	print(f"已生成论文图表：{FIG_S5_PATH}")


if __name__ == "__main__":
	main()
