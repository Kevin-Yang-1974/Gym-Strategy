# -*- coding: utf-8 -*-
"""候选积分策略模拟、经营约束筛选、熵权-TOPSIS排序与多延期观察窗口分析。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "数据整理程序输出" / "整理后的数据"
OUT_DIR = BASE_DIR / "模型输出结果"

RENEW_PRED_PATH = OUT_DIR / "5.3_状态增强随机森林与对比模型预测_H30.csv"
RENEW_DATA_PATH = DATA_DIR / "renewal_survival_dataset_H30.csv"
PT_PRED_PATH = OUT_DIR / "5.4_私教购买模型测试集预测.csv"
PT_DATA_PATH = DATA_DIR / "pt_purchase_dataset_30d.csv"

STRATEGY_DETAIL_PATH = OUT_DIR / "5.5_候选积分策略模拟明细.csv"
STRATEGY_SUMMARY_PATH = OUT_DIR / "5.5_候选积分策略汇总.csv"
TOPSIS_PATH = OUT_DIR / "5.6_熵权TOPSIS策略排序.csv"
REPORT_PATH = OUT_DIR / "5.5_5.6_积分策略模拟与TOPSIS结果.md"

MULTI_WINDOW_SUMMARY_PATH = OUT_DIR / "5.6_多延迟观察窗口方案汇总.csv"
MULTI_WINDOW_TRAJECTORY_PATH = OUT_DIR / "5.6_多延迟观察窗口轨迹.csv"
S5_VS_S0_PATH = OUT_DIR / "5.6_S5_vs_S0利润差.csv"
S5_VS_S0_CROSSOVER_PATH = OUT_DIR / "5.6_S5_vs_S0交叉点表.csv"
MULTI_WINDOW_REPORT_PATH = OUT_DIR / "5.6_多延迟观察窗口分析.md"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] =150

C_POINT =0.01
TAU =0.30
ETA =0.18
EPS =1e-9
WINDOW1_WEIGHT =1.0
DELAYED_WINDOW_WEIGHT =1.0

BASE_CARRY_A =0.35
BASE_CARRY_RENEW =0.35
BASE_CARRY_PT =0.25

LONG_HORIZON_CARRY_SCENARIOS = [
	{"观察期标签": "L1", "carry_a":0.20, "carry_renew":0.20, "carry_pt":0.15},
	{"观察期标签": "L2", "carry_a":0.35, "carry_renew":0.35, "carry_pt":0.25},
	{"观察期标签": "L3", "carry_a":0.50, "carry_renew":0.50, "carry_pt":0.35},
	{"观察期标签": "L4", "carry_a":0.70, "carry_renew":0.60, "carry_pt":0.45},
]
DELAY_WINDOW_COUNTS = [1,2,3,4,6,8,12]
BASE_DELAY_COUNT =1

STRATEGIES = [
	{
		"策略编号": "S0",
		"策略名称": "无积分基准策略",
		"p0":0,
		"p1":0,
		"bonus_z1":0,
		"bonus_z6":0,
		"bonus_z7":0,
		"pmax":0,
		"redeem_prob":0,
		"rho_scale":0,
		"q_scale":0,
	},
	{
		"策略编号": "S1",
		"策略名称": "低补贴保留策略",
		"p0":6,
		"p1":2,
		"bonus_z1":20,
		"bonus_z6":10,
		"bonus_z7":10,
		"pmax":120,
		"redeem_prob":0.30,
		"rho_scale":0.80,
		"q_scale":0.80,
	},
	{
		"策略编号": "S2",
		"策略名称": "均衡促活策略",
		"p0":10,
		"p1":4,
		"bonus_z1":20,
		"bonus_z6":40,
		"bonus_z7":50,
		"pmax":220,
		"redeem_prob":0.50,
		"rho_scale":1.00,
		"q_scale":1.00,
	},
	{
		"策略编号": "S3",
		"策略名称": "私教转化强化策略",
		"p0":9,
		"p1":4,
		"bonus_z1":10,
		"bonus_z6":90,
		"bonus_z7":35,
		"pmax":260,
		"redeem_prob":0.55,
		"rho_scale":1.10,
		"q_scale":1.10,
	},
	{
		"策略编号": "S4",
		"策略名称": "流失召回强化策略",
		"p0":12,
		"p1":5,
		"bonus_z1":10,
		"bonus_z6":40,
		"bonus_z7":110,
		"pmax":300,
		"redeem_prob":0.60,
		"rho_scale":1.20,
		"q_scale":1.20,
	},
	{
		"策略编号": "S5",
		"策略名称": "高强度综合激励策略",
		"p0":16,
		"p1":7,
		"bonus_z1":40,
		"bonus_z6":100,
		"bonus_z7":130,
		"pmax":420,
		"redeem_prob":0.70,
		"rho_scale":1.50,
		"q_scale":1.50,
	},
]

GROUP_PARAMS = {
	"Z1_高活跃高消费用户": {"rho":0.25, "q":0.18, "pref":160, "lambda_renew":0.50, "lambda_pt":0.25},
	"Z6_私教潜在用户": {"rho":0.45, "q":0.28, "pref":220, "lambda_renew":0.85, "lambda_pt":1.20},
	"Z7_流失风险用户": {"rho":0.35, "q":0.22, "pref":240, "lambda_renew":1.10, "lambda_pt":0.45},
	"其他": {"rho":0.30, "q":0.20, "pref":200, "lambda_renew":0.70, "lambda_pt":0.50},
}


def load_population():
	renew_pred = pd.read_csv(RENEW_PRED_PATH)
	renew_pred["end_date"] = pd.to_datetime(renew_pred["end_date"], errors="coerce")
	renew_data_cols = [
		"user_id",
		"cycle_id",
		"end_date",
		"current_member_price",
		"visit_count",
		"post_count",
		"like_given_count",
		"comment_given_count",
		"A_score",
		"S_score",
		"user_state",
	]
	renew_data = pd.read_csv(RENEW_DATA_PATH, usecols=renew_data_cols, low_memory=False)
	renew_data["end_date"] = pd.to_datetime(renew_data["end_date"], errors="coerce")
	pop = renew_pred.merge(renew_data, on=["user_id", "cycle_id", "end_date"], how="left")
	pop = pop.rename(columns={"普通随机森林_prob": "p_renew_base"})

	pt_pred = pd.read_csv(PT_PRED_PATH, usecols=["user_id", "随机森林_prob"])
	pt_prob = pt_pred.groupby("user_id", as_index=False)["随机森林_prob"].mean().rename(columns={"随机森林_prob": "p_pt_base"})
	pop = pop.merge(pt_prob, on="user_id", how="left")

	pt_data_cols = ["user_id", "pt_unit_price_est", "pt_purchase_amount_next", "y_pt"]
	pt_data = pd.read_csv(PT_DATA_PATH, usecols=pt_data_cols, low_memory=False)
	positive_amount_mean = pt_data.loc[pt_data["y_pt"] ==1, "pt_purchase_amount_next"].replace(0, np.nan).dropna().mean()
	if pd.isna(positive_amount_mean):
		positive_amount_mean =300.0
	pt_price_by_user = pt_data.groupby("user_id", as_index=False)["pt_unit_price_est"].max()
	pop = pop.merge(pt_price_by_user, on="user_id", how="left")

	pop["p_renew_base"] = pop["p_renew_base"].fillna(pop["p_renew_base"].mean()).clip(0,1)
	pop["p_pt_base"] = pop["p_pt_base"].fillna(pt_pred["随机森林_prob"].mean()).clip(0,1)
	pop["Price_m"] = pd.to_numeric(pop["current_member_price"], errors="coerce").fillna(pop["current_member_price"].median()).clip(lower=1)
	pop["Price_p"] = pd.to_numeric(pop["pt_unit_price_est"], errors="coerce").fillna(0)
	pop.loc[pop["Price_p"] <=0, "Price_p"] = positive_amount_mean
	pop["Price_p"] = pop["Price_p"].clip(lower=1)
	pop["visit_count"] = pd.to_numeric(pop["visit_count"], errors="coerce").fillna(0).clip(lower=0)
	for col in ["post_count", "like_given_count", "comment_given_count", "A_score", "S_score"]:
		pop[col] = pd.to_numeric(pop[col], errors="coerce").fillna(0).clip(lower=0)
	pop["user_state"] = pop["user_state"].fillna("其他")
	pop.loc[~pop["user_state"].isin(GROUP_PARAMS.keys()), "user_state"] = "其他"
	pop["N_eff"] = pop["visit_count"].clip(upper=12)
	pop["I_eff"] = (pop["post_count"] + pop["like_given_count"] + pop["comment_given_count"]).clip(upper=20)
	return pop


def group_bonus(strategy, state):
	if state == "Z1_高活跃高消费用户":
		return strategy["bonus_z1"]
	if state == "Z6_私教潜在用户":
		return strategy["bonus_z6"]
	if state == "Z7_流失风险用户":
		return strategy["bonus_z7"]
	return min(strategy["bonus_z6"], strategy["bonus_z7"])


def compute_response_params(pop):
	target = pop["A_score"].quantile(0.75)
	params = {}
	for state in GROUP_PARAMS:
		mask = pop["user_state"] == state
		if mask.sum() ==0:
			mean_a = pop["A_score"].mean()
		else:
			mean_a = pop.loc[mask, "A_score"].mean()
		base = GROUP_PARAMS[state]
		alpha_base = max(base["rho"] * (target - mean_a),0)
		beta_base = -np.log(1 - base["q"]) / base["pref"]
		params[state] = {"alpha_base": alpha_base, "beta_base": beta_base, **base}
	return params


def build_scenario(carry_scenario, delay_count):
	return {
		"观察期标签": carry_scenario["观察期标签"],
		"delay_window_count": delay_count,
		"总窗口数":1 + delay_count,
		"carry_a": carry_scenario["carry_a"],
		"carry_renew": carry_scenario["carry_renew"],
		"carry_pt": carry_scenario["carry_pt"],
	}


def apply_window_transition(p_base, delta_a, user_state, response_params, active_strategy, carry_renew=BASE_CARRY_RENEW, carry_pt=BASE_CARRY_PT):
	lambda_renew = user_state.map(lambda s: response_params[s]["lambda_renew"])
	lambda_pt = user_state.map(lambda s: response_params[s]["lambda_pt"])
	if active_strategy:
		p_renew_new = (p_base["renew"] + lambda_renew * delta_a * (1 - p_base["renew"])).clip(0,1)
		p_pt_new = (p_base["pt"] + lambda_pt * delta_a * (1 - p_base["pt"])).clip(0,1)
	else:
		p_renew_new = (p_base["renew"] + carry_renew * lambda_renew * delta_a * (1 - p_base["renew"])).clip(0,1)
		p_pt_new = (p_base["pt"] + carry_pt * lambda_pt * delta_a * (1 - p_base["pt"])).clip(0,1)
	return p_renew_new, p_pt_new


def simulate_strategy(pop, strategy, response_params, scenario):
	carry_a = scenario["carry_a"]
	carry_renew = scenario["carry_renew"]
	carry_pt = scenario["carry_pt"]
	delay_window_count = scenario["delay_window_count"]

	df = pop[["user_id", "user_state", "p_renew_base", "p_pt_base", "Price_m", "Price_p", "N_eff", "I_eff", "A_score"]].copy()
	df["bonus"] = df["user_state"].map(lambda s: group_bonus(strategy, s))
	df["points_w1"] = np.minimum(strategy["p0"] * df["N_eff"] + strategy["p1"] * df["I_eff"] + df["bonus"], strategy["pmax"])
	alpha = df["user_state"].map(lambda s: response_params[s]["alpha_base"] * strategy["rho_scale"])
	beta = df["user_state"].map(lambda s: response_params[s]["beta_base"] * strategy["q_scale"])
	df["delta_A_w1"] = alpha * (1 - np.exp(-beta * df["points_w1"]))

	p_base = {"renew": df["p_renew_base"], "pt": df["p_pt_base"]}
	p_renew_curr, p_pt_curr = apply_window_transition(p_base, df["delta_A_w1"], df["user_state"], response_params, active_strategy=True)

	rm_w1 = float((WINDOW1_WEIGHT * p_renew_curr * df["Price_m"]).sum())
	rp_w1 = float((WINDOW1_WEIGHT * p_pt_curr * df["Price_p"]).sum())
	raw_cost_w1 = df["points_w1"] * C_POINT * strategy["redeem_prob"]
	cost_cap_w1 = TAU * WINDOW1_WEIGHT * (p_renew_curr * df["Price_m"] + p_pt_curr * df["Price_p"])
	point_cost_w1 = np.minimum(raw_cost_w1, cost_cap_w1)
	cost_w1 = float(point_cost_w1.sum())

	trajectory_rows = [{
		"观察期标签": scenario["观察期标签"],
		"delay_window_count": delay_window_count,
		"总窗口数": scenario["总窗口数"],
		"策略编号": strategy["策略编号"],
		"策略名称": strategy["策略名称"],
		"window_index":1,
		"window_type": "active",
		"window_weight": WINDOW1_WEIGHT,
		"会员收入": rm_w1,
		"私教收入": rp_w1,
		"积分成本": cost_w1,
		"平台利润贡献":0.35 * rm_w1 +0.30 * rp_w1 - cost_w1,
		"累计平台利润":0.35 * rm_w1 +0.30 * rp_w1 - cost_w1,
		"窗口末续费概率提升": float((p_renew_curr - df["p_renew_base"]).sum()),
		"窗口末私教购买概率提升": float((p_pt_curr - df["p_pt_base"]).sum()),
		"窗口活跃提升": float(df["delta_A_w1"].sum()),
	}]

	delayed_rm_total =0.0
	delayed_rp_total =0.0
	delta_a_total = float(df["delta_A_w1"].sum())
	delta_a_prev = df["delta_A_w1"]
	cum_profit = trajectory_rows[0]["累计平台利润"]

	for window_idx in range(2, delay_window_count +2):
		delta_a_curr = carry_a * delta_a_prev
		p_prev = {"renew": p_renew_curr, "pt": p_pt_curr}
		p_renew_curr, p_pt_curr = apply_window_transition(
			p_prev,
			delta_a_curr,
			df["user_state"],
			response_params,
			active_strategy=False,
			carry_renew=carry_renew,
			carry_pt=carry_pt,
		)
		rm_curr = float((DELAYED_WINDOW_WEIGHT * p_renew_curr * df["Price_m"]).sum())
		rp_curr = float((DELAYED_WINDOW_WEIGHT * p_pt_curr * df["Price_p"]).sum())
		profit_curr =0.35 * rm_curr +0.30 * rp_curr
		cum_profit += profit_curr
		delayed_rm_total += rm_curr
		delayed_rp_total += rp_curr
		delta_a_total += float(delta_a_curr.sum())
		trajectory_rows.append({
			"观察期标签": scenario["观察期标签"],
			"delay_window_count": delay_window_count,
			"总窗口数": scenario["总窗口数"],
			"策略编号": strategy["策略编号"],
			"策略名称": strategy["策略名称"],
			"window_index": window_idx,
			"window_type": "delayed",
			"window_weight": DELAYED_WINDOW_WEIGHT,
			"会员收入": rm_curr,
			"私教收入": rp_curr,
			"积分成本":0.0,
			"平台利润贡献": profit_curr,
			"累计平台利润": cum_profit,
			"窗口末续费概率提升": float((p_renew_curr - df["p_renew_base"]).sum()),
			"窗口末私教购买概率提升": float((p_pt_curr - df["p_pt_base"]).sum()),
			"窗口活跃提升": float(delta_a_curr.sum()),
		})
		delta_a_prev = delta_a_curr

	rm_total = rm_w1 + delayed_rm_total
	rp_total = rp_w1 + delayed_rp_total
	cost_total = cost_w1
	profit_total =0.35 * rm_total +0.30 * rp_total - cost_total
	cost_rate = cost_total / max(rm_total + rp_total, EPS)
	summary = {
		"观察期标签": scenario["观察期标签"],
		"delay_window_count": delay_window_count,
		"总窗口数": scenario["总窗口数"],
		"窗口1权重": WINDOW1_WEIGHT,
		"延期窗口权重": DELAYED_WINDOW_WEIGHT,
		"延续活跃系数": carry_a,
		"延续续费系数": carry_renew,
		"延续私教系数": carry_pt,
		"策略编号": strategy["策略编号"],
		"策略名称": strategy["策略名称"],
		"p0": strategy["p0"],
		"p1": strategy["p1"],
		"Pmax": strategy["pmax"],
		"兑换概率": strategy["redeem_prob"],
		"窗口1会员收入": rm_w1,
		"窗口1私教收入": rp_w1,
		"延期窗口会员收入合计": delayed_rm_total,
		"延期窗口私教收入合计": delayed_rp_total,
		"会员收入": rm_total,
		"私教收入": rp_total,
		"积分成本": cost_total,
		"平台利润": profit_total,
		"总活跃度提升": delta_a_total,
		"最终续费概率提升": float((p_renew_curr - df["p_renew_base"]).sum()),
		"最终私教购买概率提升": float((p_pt_curr - df["p_pt_base"]).sum()),
		"成本收入比": cost_rate,
		"是否利润非负": profit_total >=0,
		"是否满足成本约束": cost_rate <= ETA,
	}
	trajectory_df = pd.DataFrame(trajectory_rows)
	return summary, trajectory_df


def compute_s0_gap_table(summary_df):
	base = summary_df[summary_df["策略编号"] == "S0"][["观察期标签", "delay_window_count", "平台利润"]].rename(columns={"平台利润": "S0平台利润"})
	s5 = summary_df[summary_df["策略编号"] == "S5"][["观察期标签", "delay_window_count", "总窗口数", "延续活跃系数", "延续续费系数", "延续私教系数", "平台利润"]].rename(columns={"平台利润": "S5平台利润"})
	gap = s5.merge(base, on=["观察期标签", "delay_window_count"], how="left")
	gap["利润差_S5减S0"] = gap["S5平台利润"] - gap["S0平台利润"]
	gap["利润差占S0比例"] = gap["利润差_S5减S0"] / gap["S0平台利润"].replace(0, np.nan)
	gap["是否已超过S0"] = gap["利润差_S5减S0"] >0
	return gap.sort_values(["观察期标签", "delay_window_count"])


def compute_crossover_table(gap_df):
	rows = []
	for label, group in gap_df.groupby("观察期标签"):
		group = group.sort_values("delay_window_count")
		over = group[group["是否已超过S0"]]
		if over.empty:
			first_delay = np.nan
			first_total = np.nan
			first_gap = np.nan
			has_cross = False
		else:
			first = over.iloc[0]
			first_delay = int(first["delay_window_count"])
			first_total = int(first["总窗口数"])
			first_gap = float(first["利润差_S5减S0"])
			has_cross = True
		last = group.iloc[-1]
		rows.append({
			"观察期标签": label,
			"延续活跃系数": float(last["延续活跃系数"]),
			"延续续费系数": float(last["延续续费系数"]),
			"延续私教系数": float(last["延续私教系数"]),
			"最大测试延期窗口数": int(last["delay_window_count"]),
			"最大测试总窗口数": int(last["总窗口数"]),
			"最大测试窗口内是否超过S0": has_cross,
			"首次超过S0的delay_window_count": first_delay,
			"首次超过S0的总窗口数": first_total,
			"首次超过S0时利润差": first_gap,
			"最大测试窗口时利润差": float(last["利润差_S5减S0"]),
		})
	return pd.DataFrame(rows)


def entropy_topsis(summary_df):
	criteria = ["平台利润", "总活跃度提升", "最终续费概率提升", "最终私教购买概率提升", "积分成本", "成本收入比"]
	positive = ["平台利润", "总活跃度提升", "最终续费概率提升", "最终私教购买概率提升"]
	feasible = summary_df[(summary_df["是否利润非负"]) & (summary_df["是否满足成本约束"]) & (summary_df["策略编号"] != "S0")].copy()
	if feasible.empty:
		return feasible
	matrix = feasible[criteria].astype(float).copy()
	for col in criteria:
		min_v = matrix[col].min()
		max_v = matrix[col].max()
		if np.isclose(max_v, min_v):
			matrix[col] =1.0
		elif col in positive:
			matrix[col] = (matrix[col] - min_v) / (max_v - min_v)
		else:
			matrix[col] = (max_v - matrix[col]) / (max_v - min_v)
	matrix = matrix + EPS
	p = matrix / matrix.sum(axis=0)
	k =1 / np.log(len(matrix)) if len(matrix) >1 else 1
	entropy = -k * (p * np.log(p)).sum(axis=0)
	d =1 - entropy
	weights = d / d.sum() if d.sum() >0 else pd.Series(1 / len(criteria), index=criteria)
	weighted = matrix * weights
	ideal_best = weighted.max(axis=0)
	ideal_worst = weighted.min(axis=0)
	d_best = np.sqrt(((weighted - ideal_best) **2).sum(axis=1))
	d_worst = np.sqrt(((weighted - ideal_worst) **2).sum(axis=1))
	feasible["TOPSIS得分"] = d_worst / (d_best + d_worst + EPS)
	feasible["排序"] = feasible["TOPSIS得分"].rank(ascending=False, method="min").astype(int)
	return feasible.sort_values("排序")


def save_figures(summary_df, topsis_df):
	plot_df = summary_df[summary_df["策略编号"] != "S0"].copy()
	fig, ax1 = plt.subplots(figsize=(8,5))
	x = np.arange(len(plot_df))
	ax1.bar(x -0.2, plot_df["平台利润"], width=0.4, label="平台利润", color="#5b8db8")
	ax1.set_ylabel("平台利润")
	ax2 = ax1.twinx()
	ax2.bar(x +0.2, plot_df["积分成本"], width=0.4, label="积分成本", color="#e6955f")
	ax2.set_ylabel("积分成本")
	ax1.set_xticks(x)
	ax1.set_xticklabels(plot_df["策略编号"])
	ax1.set_title("候选积分策略利润与成本对比")
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.5_图1_候选积分策略利润成本对比.png", bbox_inches="tight")
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(8,5))
	ax.bar(plot_df["策略编号"], plot_df["最终续费概率提升"], label="最终续费概率提升", color="#8ab17d")
	ax.bar(plot_df["策略编号"], plot_df["最终私教购买概率提升"], bottom=plot_df["最终续费概率提升"], label="最终私教购买概率提升", color="#b07aa1")
	ax.set_title("候选积分策略概率提升对比")
	ax.set_ylabel("概率提升总量")
	ax.legend()
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.5_图2_候选积分策略概率提升对比.png", bbox_inches="tight")
	plt.close(fig)

	if not topsis_df.empty:
		fig, ax = plt.subplots(figsize=(8,5))
		ordered = topsis_df.sort_values("TOPSIS得分", ascending=True)
		ax.barh(ordered["策略编号"] + "_" + ordered["策略名称"], ordered["TOPSIS得分"], color="#5b8db8")
		ax.set_xlabel("TOPSIS得分")
		ax.set_title("可行积分策略TOPSIS排序")
		fig.tight_layout()
		fig.savefig(OUT_DIR / "5.6_图1_可行策略TOPSIS排序.png", bbox_inches="tight")
		plt.close(fig)


def write_report(summary_df, topsis_df):
	best = topsis_df.iloc[0] if not topsis_df.empty else None
	if best is not None:
		best_text = f"综合窗口1与1个延期观察窗口的经营约束和熵权-TOPSIS排序，最优策略为 **{best['策略编号']}：{best['策略名称']}**，TOPSIS得分为 {best['TOPSIS得分']:.4f}。"
	else:
		best_text = "当前候选策略中没有同时满足利润非负和成本收入比约束的可行策略，需要降低积分强度或放宽成本约束。"
	report = f"""#5.5 候选积分策略模拟与5.6 熵权-TOPSIS综合评价结果

本文件由 `模型输出结果/simulate_points_strategy_topsis.py` 自动生成，用于连接会员续费概率、私教购买概率、积分响应函数和平台利润函数。

##1. 概率输入与模拟边界

会员续费概率采用普通随机森林输出的 $\\hat p_i^{{renew,30}}$，私教购买概率采用随机森林输出的 $\\hat p_i^{{pt}}$。由于历史数据中没有真实积分干预记录，积分带来的活跃提升和概率提升属于基于响应假设的情景模拟结果，不解释为真实因果效应。

##2. 两阶段窗口设定

本次基准模拟将策略效果划分为两个时间窗口：窗口1为返利机制实施期，窗口2为返利结束后的延续观察期。窗口1计入积分发放与兑换成本，窗口2不再新增积分成本，但保留窗口1形成的部分活跃提升惯性，并据此计算第二窗口末的最终续费概率和最终私教购买概率。

##3. 策略模拟汇总

{summary_df.to_markdown(index=False)}

##4.经营约束与TOPSIS排序

$$\\Pi(r_j)\\geq0,\\quad C_{{point}}(r_j)\\leq\\eta(R_m(r_j)+R_p(r_j))$$

其中，本次模拟设定成本收入比上限 $\\eta={ETA}$，收入和利润均按两个窗口合计值计算。

{topsis_df.to_markdown(index=False) if not topsis_df.empty else '无可行策略。'}

##5. 最优策略结论

{best_text}

##6.论文表述边界

积分策略结果应表述为“基于历史行为预测模型和积分响应假设的情景模拟结果”。由于缺少真实积分干预实验数据，不能将续费概率提升、私教购买概率提升和利润变化解释为真实因果效应。后续若平台实施该策略，应通过 A/B 实验进一步验证真实运营效果。
"""
	REPORT_PATH.write_text(report, encoding="utf-8")


def write_multi_window_report(summary_df, gap_df, crossover_df):
	best_rows = gap_df.groupby("观察期标签", as_index=False).apply(lambda x: x.loc[x["利润差_S5减S0"].idxmax()]).reset_index(drop=True)
	report = f"""#5.6 多延迟观察窗口分析

本文件由 `模型输出结果/simulate_points_strategy_topsis.py` 自动生成，用于比较在**窗口1权重=1.0、每个延期观察窗口权重=1.0**时，S5 相对无积分基准策略 S0 的长期累计利润变化。

##1. 分析方案

本次长期效应分析固定：

- 窗口1权重 =1.0
- 每个延期观察窗口权重 =1.0
- 延期窗口数测试为：{DELAY_WINDOW_COUNTS}

长期效应参数组如下：

{pd.DataFrame(LONG_HORIZON_CARRY_SCENARIOS).to_markdown(index=False)}

##2. S5 与 S0 的利润差表

{gap_df.to_markdown(index=False)}

##3. S5 是否出现利润反超的交叉点表

{crossover_df.to_markdown(index=False)}

##4. 各长期效应场景下“最接近反超”的结果

{best_rows[["观察期标签", "delay_window_count", "总窗口数", "S5平台利润", "S0平台利润", "利润差_S5减S0", "是否已超过S0"]].to_markdown(index=False)}

##5. 多窗口策略总表（含S0与S5以外策略）

{summary_df.to_markdown(index=False)}

##6.结论解释

若某一场景的 `利润差_S5减S0` 始终为负，则表示在当前长期延续效应假设下，即使把延期观察窗口扩展到更长周期，S5 的累计利润仍未超过 S0。若某一场景出现正值，则 `首次超过S0的delay_window_count` 即为反超所需的最小延期窗口数。

##7.结果边界

该分析仍属于基于历史行为预测与积分响应假设的情景模拟，不构成真实运营因果结论。若未来实施积分策略，应通过 A/B 实验验证返利结束后的真实长期延续效应。
"""
	MULTI_WINDOW_REPORT_PATH.write_text(report, encoding="utf-8")


def main():
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	pop = load_population()
	response_params = compute_response_params(pop)

	base_scenario = build_scenario(LONG_HORIZON_CARRY_SCENARIOS[1], BASE_DELAY_COUNT)
	base_summaries = []
	base_details = []
	for strategy in STRATEGIES:
		summary, detail = simulate_strategy(pop, strategy, response_params, base_scenario)
		base_summaries.append(summary)
		base_details.append(detail)
	base_summary_df = pd.DataFrame(base_summaries)
	base_detail_df = pd.concat(base_details, ignore_index=True)
	base_topsis_df = entropy_topsis(base_summary_df)

	base_summary_df.to_csv(STRATEGY_SUMMARY_PATH, index=False, encoding="utf-8-sig")
	base_detail_df.to_csv(STRATEGY_DETAIL_PATH, index=False, encoding="utf-8-sig")
	base_topsis_df.to_csv(TOPSIS_PATH, index=False, encoding="utf-8-sig")
	save_figures(base_summary_df, base_topsis_df)
	write_report(base_summary_df, base_topsis_df)

	all_summaries = []
	all_trajectories = []
	for carry_scenario in LONG_HORIZON_CARRY_SCENARIOS:
		for delay_count in DELAY_WINDOW_COUNTS:
			scenario = build_scenario(carry_scenario, delay_count)
			for strategy in STRATEGIES:
				summary, detail = simulate_strategy(pop, strategy, response_params, scenario)
				all_summaries.append(summary)
				all_trajectories.append(detail)

	multi_summary_df = pd.DataFrame(all_summaries)
	multi_trajectory_df = pd.concat(all_trajectories, ignore_index=True)
	gap_df = compute_s0_gap_table(multi_summary_df)
	crossover_df = compute_crossover_table(gap_df)

	multi_summary_df.to_csv(MULTI_WINDOW_SUMMARY_PATH, index=False, encoding="utf-8-sig")
	multi_trajectory_df.to_csv(MULTI_WINDOW_TRAJECTORY_PATH, index=False, encoding="utf-8-sig")
	gap_df.to_csv(S5_VS_S0_PATH, index=False, encoding="utf-8-sig")
	crossover_df.to_csv(S5_VS_S0_CROSSOVER_PATH, index=False, encoding="utf-8-sig")
	write_multi_window_report(multi_summary_df, gap_df, crossover_df)

	print(f"已生成积分策略模拟与TOPSIS结果：{REPORT_PATH}")
	print(f"已生成多延迟观察窗口分析：{MULTI_WINDOW_REPORT_PATH}")


if __name__ == "__main__":
	main()
