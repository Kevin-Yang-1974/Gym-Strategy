# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "整理后的数据"
OUTPUT_DIR = BASE_DIR / "模型输出结果"
OUTPUT_DIR.mkdir(exist_ok=True)

STATE_DATA_PATH = DATA_DIR / "state_transition_dataset.csv"
USER_MONTH_PATH = DATA_DIR / "user_month_features.csv"
REPORT_PATH = OUTPUT_DIR / "状态转移分析结果.md"
STATE_DIST_PATH = OUTPUT_DIR / "状态分布表.csv"
TRANSITION_COUNT_PATH = OUTPUT_DIR / "状态转移频数矩阵.csv"
TRANSITION_PROB_PATH = OUTPUT_DIR / "状态转移概率矩阵.csv"
STATE_SUMMARY_PATH = OUTPUT_DIR / "状态转移关键指标.csv"

STATE_ORDER = [
	"Z1_高活跃高消费用户",
	"Z2_高活跃低消费用户",
	"Z3_低活跃高消费用户",
	"Z4_线上活跃线下弱用户",
	"Z5_低活跃低互动低消费用户",
	"Z6_私教潜在用户",
	"Z7_流失风险用户",
]


def md_table(df, max_rows=50):
	if df.empty:
		return "无数据。\n"
	return df.head(max_rows).to_markdown(index=False) + "\n"


def pct(x):
	return f"{x:.2%}"


def read_data():
	state_df = pd.read_csv(STATE_DATA_PATH, encoding="utf-8-sig")
	user_month_df = pd.read_csv(USER_MONTH_PATH, encoding="utf-8-sig")
	return state_df, user_month_df


def state_distribution(user_month_df):
	dist = user_month_df["user_state"].value_counts(dropna=False).reset_index()
	dist.columns = ["state", "count"]
	dist["rate"] = dist["count"] / dist["count"].sum()
	dist["rate_percent"] = dist["rate"].map(pct)
	return dist


def transition_matrices(state_df):
	count = pd.crosstab(state_df["Z_t"], state_df["Z_t_next"])
	prob = pd.crosstab(state_df["Z_t"], state_df["Z_t_next"], normalize="index")

	available_states = [state for state in STATE_ORDER if state in count.index or state in count.columns]
	count = count.reindex(index=available_states, columns=available_states, fill_value=0)
	prob = prob.reindex(index=available_states, columns=available_states, fill_value=0).round(4)
	return count, prob


def key_state_summary(count, prob):
	rows = []
	for state in prob.index:
		total = int(count.loc[state].sum())
		stay_prob = float(prob.loc[state, state]) if state in prob.columns else0
		to_risk_prob = float(prob.loc[state, "Z7_流失风险用户"]) if "Z7_流失风险用户" in prob.columns else0
		to_pt_prob = float(prob.loc[state, "Z6_私教潜在用户"]) if "Z6_私教潜在用户" in prob.columns else0
		to_high_value_prob = float(prob.loc[state, "Z1_高活跃高消费用户"]) if "Z1_高活跃高消费用户" in prob.columns else0
		rows.append({
			"state": state,
			"transition_count": total,
			"stay_prob": stay_prob,
			"stay_percent": pct(stay_prob),
			"to_risk_prob": to_risk_prob,
			"to_risk_percent": pct(to_risk_prob),
			"to_pt_potential_prob": to_pt_prob,
			"to_pt_potential_percent": pct(to_pt_prob),
			"to_high_value_prob": to_high_value_prob,
			"to_high_value_percent": pct(to_high_value_prob),
		})
	return pd.DataFrame(rows)


def write_report(state_dist, count, prob, summary):
	lines = []
	lines.append("# 状态转移分析结果\n\n")
	lines.append("本结果基于 `user_month_features.csv` 和 `state_transition_dataset.csv`生成，用于论文中用户状态划分与自然状态转移分析部分。\n\n")

	lines.append("##1. 用户状态分布\n\n")
	lines.append(md_table(state_dist))
	lines.append("从状态分布看，当前真实数据中用户状态主要集中在少数几类。这说明理论上设定的多类用户状态在实证样本中并非均衡出现，后续应重点围绕样本量充足、业务含义明确的核心状态进行解释。\n\n")

	lines.append("##2. 状态转移频数矩阵\n\n")
	lines.append(md_table(count.reset_index().rename(columns={"Z_t": "当前状态"}), max_rows=100))

	lines.append("\n##3. 状态转移概率矩阵\n\n")
	prob_display = prob.reset_index().rename(columns={"Z_t": "当前状态"})
	lines.append(md_table(prob_display, max_rows=100))

	lines.append("\n##4.关键转移指标\n\n")
	lines.append(md_table(summary, max_rows=100))

	lines.append("\n##5.结果解释\n\n")
	if "Z1_高活跃高消费用户" in prob.index:
		stay = prob.loc["Z1_高活跃高消费用户", "Z1_高活跃高消费用户"]
		risk = prob.loc["Z1_高活跃高消费用户", "Z7_流失风险用户"] if "Z7_流失风险用户" in prob.columns else0
		lines.append(f"- 高活跃高消费用户的状态保持概率为 {pct(stay)}，转为流失风险用户的概率为 {pct(risk)}。这说明高价值用户具有一定稳定性，但仍存在流失风险，积分策略上不宜过度补贴，应以低成本保留和权益感知强化为主。\n")
	if "Z6_私教潜在用户" in prob.index:
		stay = prob.loc["Z6_私教潜在用户", "Z6_私教潜在用户"]
		risk = prob.loc["Z6_私教潜在用户", "Z7_流失风险用户"] if "Z7_流失风险用户" in prob.columns else0
		lines.append(f"- 私教潜在用户的状态保持概率为 {pct(stay)}，转为流失风险用户的概率为 {pct(risk)}。这说明该类用户既具有私教转化价值，也存在较明显流失风险，适合设计与私教体验、力量训练打卡或课程抵扣相关的定向积分激励。\n")
	if "Z7_流失风险用户" in prob.index:
		stay = prob.loc["Z7_流失风险用户", "Z7_流失风险用户"]
		to_pt = prob.loc["Z7_流失风险用户", "Z6_私教潜在用户"] if "Z6_私教潜在用户" in prob.columns else0
		to_high = prob.loc["Z7_流失风险用户", "Z1_高活跃高消费用户"] if "Z1_高活跃高消费用户" in prob.columns else0
		lines.append(f"- 流失风险用户保持为流失风险用户的概率为 {pct(stay)}，转为私教潜在用户的概率为 {pct(to_pt)}，转为高活跃高消费用户的概率为 {pct(to_high)}。这说明流失风险用户自然恢复概率较低，需要通过召回型积分或低门槛线下运动任务进行干预。\n")

	lines.append("\n##6. 对后续积分策略的启示\n\n")
	lines.append("1. 对高活跃高消费用户，应采用低补贴、权益提醒和持续签到奖励，避免对本就稳定的用户过度让利。\n")
	lines.append("2. 对私教潜在用户，应重点设计私教体验课、力量训练、私教抵扣相关积分，提高私教转化概率。\n")
	lines.append("3. 对流失风险用户，应采用召回型积分任务，例如返场打卡、连续到店奖励和低门槛互动任务，以提高重新活跃概率。\n")
	lines.append("4.由于部分理论状态样本量较少，后续策略模拟应优先基于实证中样本量充足的核心状态进行分层。\n")

	REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def main():
	state_df, user_month_df = read_data()
	state_dist = state_distribution(user_month_df)
	count, prob = transition_matrices(state_df)
	summary = key_state_summary(count, prob)

	state_dist.to_csv(STATE_DIST_PATH, index=False, encoding="utf-8-sig")
	count.to_csv(TRANSITION_COUNT_PATH, encoding="utf-8-sig")
	prob.to_csv(TRANSITION_PROB_PATH, encoding="utf-8-sig")
	summary.to_csv(STATE_SUMMARY_PATH, index=False, encoding="utf-8-sig")
	write_report(state_dist, count, prob, summary)

	print(f"已生成：{REPORT_PATH}")
	print(f"已生成：{STATE_DIST_PATH}")
	print(f"已生成：{TRANSITION_COUNT_PATH}")
	print(f"已生成：{TRANSITION_PROB_PATH}")
	print(f"已生成：{STATE_SUMMARY_PATH}")


if __name__ == "__main__":
	main()
