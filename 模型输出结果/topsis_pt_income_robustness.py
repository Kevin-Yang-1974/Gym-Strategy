# -*- coding: utf-8 -*-
"""TOPSIS 替代指标稳健性检验。

本脚本不覆盖原 TOPSIS 和灵敏度分析结果。它只把 TOPSIS 中的
`最终私教购买概率提升` 替换为 `私教收入提升`，用于检查 S4/S5/S3
排序是否依赖私教概率提升这一指标口径。
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd

import simulate_points_strategy_topsis as sim


OUT_DIR = Path(__file__).resolve().parent

BASE_TOPSIS_PATH = OUT_DIR / "5.6_熵权TOPSIS策略排序.csv"
MULTI_DELAY_PATH = OUT_DIR / "5.6_多延迟观察窗口方案汇总.csv"
ORIG_MULTI_SUMMARY_PATH = OUT_DIR / "5.7_多延期窗口灵敏度分析汇总.csv"
ORIG_KAPPA_MULTI_SUMMARY_PATH = OUT_DIR / "5.7_kappa多延期窗口灵敏度分析汇总.csv"

BASE_OUT_PATH = OUT_DIR / "5.7_TOPSIS替代指标稳健性_基准排序.csv"
MULTI_OUT_PATH = OUT_DIR / "5.7_TOPSIS替代指标稳健性_多延期窗口汇总.csv"
KAPPA_OUT_PATH = OUT_DIR / "5.7_TOPSIS替代指标稳健性_kappa多延期汇总.csv"
REPORT_PATH = OUT_DIR / "5.7_TOPSIS替代指标稳健性报告.md"

CRITERIA_ORIGINAL = [
	"平台利润",
	"总活跃度提升",
	"最终续费概率提升",
	"最终私教购买概率提升",
	"积分成本",
	"成本收入比",
]
CRITERIA_ALTERNATIVE = [
	"平台利润",
	"总活跃度提升",
	"最终续费概率提升",
	"私教收入提升",
	"积分成本",
	"成本收入比",
]
POSITIVE_ALTERNATIVE = [
	"平台利润",
	"总活跃度提升",
	"最终续费概率提升",
	"私教收入提升",
]
FOCUS_CODES = ["S5", "S4", "S3", "S2", "S1"]
GAMMA_KAPPA_VALUES = [0.25, 0.50, 0.75, 1.00]
EPSILON_KAPPA_VALUES = [1e-9, 1e-4, 1e-3, 1e-2, 5e-2]


def log1p_array(x):
	return np.log1p(np.maximum(x, 0))


def normalize_bool(series):
	if series.dtype == bool:
		return series
	return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def prepare_bool_columns(df):
	out = df.copy()
	for col in ["是否利润非负", "是否满足成本约束"]:
		if col in out.columns:
			out[col] = normalize_bool(out[col])
	return out


def entropy_topsis_pt_income(summary_df):
	missing = [col for col in CRITERIA_ALTERNATIVE if col not in summary_df.columns]
	if missing:
		raise KeyError(f"缺少替代 TOPSIS 所需字段: {missing}")

	df = prepare_bool_columns(summary_df)
	feasible = df[
		(df["是否利润非负"])
		& (df["是否满足成本约束"])
		& (df["策略编号"] != "S0")
	].copy()
	if feasible.empty:
		return feasible

	matrix = feasible[CRITERIA_ALTERNATIVE].astype(float).copy()
	for col in CRITERIA_ALTERNATIVE:
		min_v = matrix[col].min()
		max_v = matrix[col].max()
		if np.isclose(max_v, min_v):
			matrix[col] = 1.0
		elif col in POSITIVE_ALTERNATIVE:
			matrix[col] = (matrix[col] - min_v) / (max_v - min_v)
		else:
			matrix[col] = (max_v - matrix[col]) / (max_v - min_v)

	matrix = matrix + sim.EPS
	p = matrix / matrix.sum(axis=0)
	k = 1 / np.log(len(matrix)) if len(matrix) > 1 else 1
	entropy = -k * (p * np.log(p)).sum(axis=0)
	d = 1 - entropy
	weights = d / d.sum() if d.sum() > 0 else pd.Series(1 / len(CRITERIA_ALTERNATIVE), index=CRITERIA_ALTERNATIVE)

	weighted = matrix * weights
	ideal_best = weighted.max(axis=0)
	ideal_worst = weighted.min(axis=0)
	d_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
	d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

	feasible["替代TOPSIS得分"] = d_worst / (d_best + d_worst + sim.EPS)
	feasible["替代排序"] = feasible["替代TOPSIS得分"].rank(ascending=False, method="min").astype(int)
	for col, value in weights.items():
		feasible[f"替代权重_{col}"] = float(value)
	return feasible.sort_values("替代排序")


def pick_strategy(group, code):
	rows = group[group["策略编号"] == code]
	if rows.empty:
		return None
	return rows.iloc[0]


def rank_map(topsis_df, rank_col="替代排序", score_col="替代TOPSIS得分"):
	if topsis_df.empty:
		return {}, {}
	return (
		topsis_df.set_index("策略编号")[rank_col].to_dict(),
		topsis_df.set_index("策略编号")[score_col].to_dict(),
	)


def summarize_topsis_group(group, extra=None):
	topsis = entropy_topsis_pt_income(group.copy())
	if topsis.empty:
		best_code = "无"
		best_name = "无可行策略"
		best_score = np.nan
		ranks = {}
		scores = {}
	else:
		best = topsis.iloc[0]
		best_code = best["策略编号"]
		best_name = best["策略名称"]
		best_score = float(best["替代TOPSIS得分"])
		ranks, scores = rank_map(topsis)

	s0 = pick_strategy(group, "S0")
	s1 = pick_strategy(group, "S1")
	s2 = pick_strategy(group, "S2")
	s3 = pick_strategy(group, "S3")
	s4 = pick_strategy(group, "S4")
	s5 = pick_strategy(group, "S5")

	s0_profit = float(s0["平台利润"]) if s0 is not None else np.nan
	s1_profit = float(s1["平台利润"]) if s1 is not None else np.nan
	s4_profit = float(s4["平台利润"]) if s4 is not None else np.nan
	s5_profit = float(s5["平台利润"]) if s5 is not None else np.nan

	row = {
		"可行策略数量": int(len(topsis)),
		"最优策略编号": best_code,
		"最优策略名称": best_name,
		"最优策略替代TOPSIS得分": best_score,
		"S5排序": ranks.get("S5", np.nan),
		"S5_替代TOPSIS得分": scores.get("S5", np.nan),
		"S4排序": ranks.get("S4", np.nan),
		"S4_替代TOPSIS得分": scores.get("S4", np.nan),
		"S3排序": ranks.get("S3", np.nan),
		"S3_替代TOPSIS得分": scores.get("S3", np.nan),
		"S2排序": ranks.get("S2", np.nan),
		"S1排序": ranks.get("S1", np.nan),
		"S5是否第一": best_code == "S5",
		"S4是否第一": best_code == "S4",
		"S3是否第一": best_code == "S3",
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
	if extra:
		row = {**extra, **row}
	return row


def build_baseline():
	df = pd.read_csv(BASE_TOPSIS_PATH)
	df = prepare_bool_columns(df)
	topsis = entropy_topsis_pt_income(df)
	out = topsis.copy()
	if "TOPSIS得分" in out.columns:
		out = out.rename(columns={"TOPSIS得分": "原TOPSIS得分", "排序": "原排序"})
	cols = [
		"观察期标签",
		"delay_window_count",
		"总窗口数",
		"策略编号",
		"策略名称",
		"平台利润",
		"总活跃度提升",
		"最终续费概率提升",
		"最终私教购买概率提升",
		"私教收入提升",
		"积分成本",
		"成本收入比",
		"原TOPSIS得分",
		"原排序",
		"替代TOPSIS得分",
		"替代排序",
	]
	cols = [col for col in cols if col in out.columns]
	out[cols].to_csv(BASE_OUT_PATH, index=False, encoding="utf-8-sig")
	return out


def build_multi_delay():
	df = pd.read_csv(MULTI_DELAY_PATH)
	df = prepare_bool_columns(df)
	rows = []
	group_cols = ["观察期标签", "延续活跃系数", "delay_window_count", "总窗口数"]
	for keys, group in df.groupby(group_cols, sort=True):
		extra = dict(zip(group_cols, keys))
		rows.append(summarize_topsis_group(group, extra))
	out = pd.DataFrame(rows).sort_values(["延续活跃系数", "delay_window_count"]).reset_index(drop=True)
	out.to_csv(MULTI_OUT_PATH, index=False, encoding="utf-8-sig")
	return out


def scenario_label(gamma_kappa, epsilon_kappa):
	return f"gamma={gamma_kappa:g}, epsilon={epsilon_kappa:g}"


def set_kappa_params(gamma_kappa, epsilon_kappa):
	sim.SEMANTIC_GAMMA = float(gamma_kappa)
	sim.FEATURE_UPDATE_EPS = float(epsilon_kappa)


def run_one_kappa_grid(pop, response_params, serf_model, serf_meta, pt_model, pt_meta, serf_context, pt_context, gamma_kappa, epsilon_kappa):
	original_gamma = sim.SEMANTIC_GAMMA
	original_eps = sim.FEATURE_UPDATE_EPS
	set_kappa_params(gamma_kappa, epsilon_kappa)
	try:
		rows = []
		serf_cache = {}
		pt_cache = {}
		for delay_count in sim.DELAY_WINDOW_COUNTS:
			scenario = sim.build_scenario(
				{"观察期标签": "KAPPA_MULTI_ALT_PT_INCOME", "carry_a": sim.BASE_CARRY_A},
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
			group = pd.DataFrame(summaries)
			rows.append(
				summarize_topsis_group(
					group,
					{
						"gamma_kappa": float(gamma_kappa),
						"epsilon_kappa": float(epsilon_kappa),
						"kappa情景": scenario_label(gamma_kappa, epsilon_kappa),
						"延续活跃系数": sim.BASE_CARRY_A,
						"delay_window_count": int(delay_count),
						"总窗口数": int(1 + delay_count),
					},
				)
			)
		return rows
	finally:
		sim.SEMANTIC_GAMMA = original_gamma
		sim.FEATURE_UPDATE_EPS = original_eps


def build_kappa_multi_delay():
	pop, serf_model, serf_meta, pt_model, pt_meta = sim.load_population()
	response_params = sim.compute_response_params(pop)
	serf_context = sim.build_serf_fast_context(pop, serf_model, serf_meta)
	pt_context = sim.build_pt_fast_context(pop, pt_model, pt_meta)

	if KAPPA_OUT_PATH.exists():
		existing = pd.read_csv(KAPPA_OUT_PATH)
	else:
		existing = pd.DataFrame()
	all_rows = existing.to_dict("records") if not existing.empty else []
	completed = set()
	if not existing.empty:
		for (gamma_kappa, epsilon_kappa), group in existing.groupby(["gamma_kappa", "epsilon_kappa"]):
			if len(group) >= len(sim.DELAY_WINDOW_COUNTS):
				completed.add((float(gamma_kappa), float(epsilon_kappa)))

	for gamma_kappa in GAMMA_KAPPA_VALUES:
		for epsilon_kappa in EPSILON_KAPPA_VALUES:
			key = (float(gamma_kappa), float(epsilon_kappa))
			if key in completed:
				print(f"跳过已完成替代指标 kappa 多延期情景: gamma={gamma_kappa:g}, epsilon={epsilon_kappa:g}", flush=True)
				continue
			start = time.time()
			print(f"运行替代指标 kappa 多延期情景: gamma={gamma_kappa:g}, epsilon={epsilon_kappa:g}", flush=True)
			new_rows = run_one_kappa_grid(
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
			all_rows.extend(new_rows)
			out = pd.DataFrame(all_rows).sort_values(["gamma_kappa", "epsilon_kappa", "delay_window_count"]).reset_index(drop=True)
			out.to_csv(KAPPA_OUT_PATH, index=False, encoding="utf-8-sig")
			elapsed = time.time() - start
			print(f"已写入该情景结果，用时 {elapsed / 60:.1f} 分钟: {KAPPA_OUT_PATH}", flush=True)
	out = pd.DataFrame(all_rows).sort_values(["gamma_kappa", "epsilon_kappa", "delay_window_count"]).reset_index(drop=True)
	out.to_csv(KAPPA_OUT_PATH, index=False, encoding="utf-8-sig")
	return out


def count_best(summary):
	if summary.empty:
		return pd.DataFrame(columns=["最优策略编号", "出现次数"])
	out = summary["最优策略编号"].value_counts().rename_axis("最优策略编号").reset_index(name="出现次数")
	return out.sort_values(["出现次数", "最优策略编号"], ascending=[False, True]).reset_index(drop=True)


def rank_distribution(summary, code):
	col = f"{code}排序"
	if col not in summary.columns:
		return pd.DataFrame(columns=["排序", "次数"])
	out = summary[col].value_counts(dropna=False).rename_axis("排序").reset_index(name="次数")
	return out.sort_values("排序").reset_index(drop=True)


def compare_best_counts(alt_summary, orig_path):
	if not orig_path.exists():
		return count_best(alt_summary).rename(columns={"出现次数": "替代口径次数"})
	orig = pd.read_csv(orig_path)
	orig_counts = count_best(orig).rename(columns={"出现次数": "原口径次数"})
	alt_counts = count_best(alt_summary).rename(columns={"出现次数": "替代口径次数"})
	out = pd.merge(orig_counts, alt_counts, on="最优策略编号", how="outer").fillna(0)
	out["原口径次数"] = out["原口径次数"].astype(int)
	out["替代口径次数"] = out["替代口径次数"].astype(int)
	return out.sort_values(["替代口径次数", "原口径次数", "最优策略编号"], ascending=[False, False, True])


def format_table(df):
	if df.empty:
		return "无"
	return df.to_markdown(index=False)


def baseline_focus_table(base_alt):
	cols = [
		"策略编号",
		"策略名称",
		"平台利润",
		"最终私教购买概率提升",
		"私教收入提升",
		"原排序",
		"替代排序",
		"替代TOPSIS得分",
	]
	cols = [col for col in cols if col in base_alt.columns]
	return base_alt[base_alt["策略编号"].isin(["S5", "S4", "S3", "S2", "S1"])][cols].sort_values("替代排序")


def write_report(base_alt, multi_alt, kappa_alt):
	base_focus = baseline_focus_table(base_alt)
	multi_counts = compare_best_counts(multi_alt, ORIG_MULTI_SUMMARY_PATH)
	kappa_counts = compare_best_counts(kappa_alt, ORIG_KAPPA_MULTI_SUMMARY_PATH)
	multi_s3_first = int(multi_alt["S3是否第一"].sum())
	kappa_s3_first = int(kappa_alt["S3是否第一"].sum())

	report = f"""# 5.7 TOPSIS 替代指标稳健性报告

本报告由 `模型输出结果/topsis_pt_income_robustness.py` 自动生成。检验目的，是把熵权-TOPSIS 中的 `最终私教购买概率提升` 替换为 `私教收入提升`，观察 S4、S5、S3 的相对排序是否稳定。

## 1. 指标口径

原 TOPSIS 正向指标为：

- `平台利润`
- `总活跃度提升`
- `最终续费概率提升`
- `最终私教购买概率提升`

替代 TOPSIS 正向指标为：

- `平台利润`
- `总活跃度提升`
- `最终续费概率提升`
- `私教收入提升`

负向指标保持不变：

- `积分成本`
- `成本收入比`

因此，本检验属于评价指标口径稳健性分析，不重新训练 SERF 续费模型，不重新训练增强私教购买模型，也不改变积分响应参数。

## 2. 基准窗口结果

基准 L2 两窗口下，替代指标 TOPSIS 排序如下：

{format_table(base_focus)}

完整结果见 `5.7_TOPSIS替代指标稳健性_基准排序.csv`。

## 3. 多延期窗口结果

多延期窗口共 {len(multi_alt)} 个“延续活跃系数 × 延期窗口数”组合。原口径和替代口径下的最优策略次数对比如下：

{format_table(multi_counts)}

替代口径下，S5 排名第一 {int(multi_alt["S5是否第一"].sum())} 次，S4 排名第一 {int(multi_alt["S4是否第一"].sum())} 次，S3 排名第一 {multi_s3_first} 次。完整结果见 `5.7_TOPSIS替代指标稳健性_多延期窗口汇总.csv`。

S5 排名分布：

{format_table(rank_distribution(multi_alt, "S5"))}

S4 排名分布：

{format_table(rank_distribution(multi_alt, "S4"))}

S3 排名分布：

{format_table(rank_distribution(multi_alt, "S3"))}

## 4. kappa 多延期窗口结果

kappa 多延期窗口共 {len(kappa_alt)} 个“$\\gamma_\\kappa$ 参数 × $\\epsilon_\\kappa$ 参数 × 延期窗口数”组合。原口径和替代口径下的最优策略次数对比如下：

{format_table(kappa_counts)}

替代口径下，S5 排名第一 {int(kappa_alt["S5是否第一"].sum())} 次，S4 排名第一 {int(kappa_alt["S4是否第一"].sum())} 次，S3 排名第一 {kappa_s3_first} 次。完整结果见 `5.7_TOPSIS替代指标稳健性_kappa多延期汇总.csv`。

S5 排名分布：

{format_table(rank_distribution(kappa_alt, "S5"))}

S4 排名分布：

{format_table(rank_distribution(kappa_alt, "S4"))}

S3 排名分布：

{format_table(rank_distribution(kappa_alt, "S3"))}

## 5. 稳健性结论

若替代口径下 S5 仍在基准或多数多延期组合中排名靠前，说明其综合促活、续费和私教收入提升优势不完全依赖 `最终私教购买概率提升` 这一概率人数指标。若 S4 仍在大量情景中排名靠前，同时利润差和成本压力更优，则可以继续把 S4 作为稳健落地主推荐。若 S3 只在少数情景中排名第一，说明私教收入指标替代后仍不足以支持其成为综合最优，只适合作为私教转化专项备选。

该检验仍属于基于历史预测模型和积分响应假设的情景模拟，不能解释为真实积分干预的因果效果。
"""
	REPORT_PATH.write_text(report, encoding="utf-8")


def main():
	print("重算基准 TOPSIS 替代指标口径...")
	base_alt = build_baseline()
	print(f"已生成: {BASE_OUT_PATH}")

	print("重算多延期窗口 TOPSIS 替代指标口径...")
	multi_alt = build_multi_delay()
	print(f"已生成: {MULTI_OUT_PATH}")

	print("重算 kappa 多延期窗口 TOPSIS 替代指标口径...")
	kappa_alt = build_kappa_multi_delay()
	print(f"已生成: {KAPPA_OUT_PATH}")

	write_report(base_alt, multi_alt, kappa_alt)
	print(f"已生成: {REPORT_PATH}")


if __name__ == "__main__":
	main()
