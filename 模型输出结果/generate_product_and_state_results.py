# -*- coding: utf-8 -*-
"""生成论文第5.1、5.2节所需的商品分类与状态转移结果。"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "数据整理程序输出" / "整理后的数据"
OUT_DIR = BASE_DIR / "模型输出结果"

CLASSIFIED_ORDERS = DATA_DIR / "classified_orders.csv"
STATE_TRANSITIONS = DATA_DIR / "state_transition_dataset.csv"


def md_table(df):
	cols = list(df.columns)
	lines = []
	lines.append("| " + " | ".join(str(c) for c in cols) + " |")
	lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
	for _, row in df.iterrows():
		lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
	return "\n".join(lines)


def summarize_product_classification(orders):
	total_orders = len(orders)

	product_type_dist = (
		orders["product_type"]
		.value_counts(dropna=False)
		.rename_axis("商品大类")
		.reset_index(name="订单数")
	)
	product_type_dist["占比"] = product_type_dist["订单数"] / total_orders

	product_subtype_dist = (
		orders["product_subtype"]
		.value_counts(dropna=False)
		.rename_axis("商品子类")
		.reset_index(name="订单数")
	)
	product_subtype_dist["占比"] = product_subtype_dist["订单数"] / total_orders

	type_amount = (
		orders.groupby("product_type", dropna=False)
		.agg(
			订单数=("product_type", "size"),
			金额总和=("amount", "sum"),
			平均金额=("amount", "mean"),
		)
		.reset_index()
		.rename(columns={"product_type": "商品大类"})
		.sort_values("订单数", ascending=False)
	)

	subtype_amount = (
		orders.groupby("product_subtype", dropna=False)
		.agg(
			订单数=("product_subtype", "size"),
			金额总和=("amount", "sum"),
			平均金额=("amount", "mean"),
		)
		.reset_index()
		.rename(columns={"product_subtype": "商品子类"})
		.sort_values("订单数", ascending=False)
	)

	flags = []
	flag_columns = [
		("is_member_product", "会员商品"),
		("is_pt_product", "私教商品"),
		("is_discount_product", "优惠类商品"),
		("is_gift_product", "赠送权益/0元权益"),
		("is_valid_revenue", "正常收入订单"),
		("need_manual_check", "需要人工复核"),
	]
	for col, name in flag_columns:
		count = int(orders[col].sum())
		flags.append({"指标": name, "订单数": count, "占比": count / total_orders})

	flag_summary = pd.DataFrame(flags)
	return product_type_dist, product_subtype_dist, type_amount, subtype_amount, flag_summary


def summarize_state_transition(transitions):
	state_order = ["Z1_高活跃高消费用户", "Z6_私教潜在用户", "Z7_流失风险用户"]

	z_dist = transitions["Z_t"].value_counts().rename_axis("当前状态").reset_index(name="样本数")
	z_dist["占比"] = z_dist["样本数"] / len(transitions)

	z_next_dist = transitions["Z_t_next"].value_counts().rename_axis("下一期状态").reset_index(name="样本数")
	z_next_dist["占比"] = z_next_dist["样本数"] / len(transitions)

	freq = pd.crosstab(transitions["Z_t"], transitions["Z_t_next"])
	prob = pd.crosstab(transitions["Z_t"], transitions["Z_t_next"], normalize="index")

	existing = [s for s in state_order if s in freq.index or s in freq.columns]
	freq = freq.reindex(index=existing, columns=existing, fill_value=0)
	prob = prob.reindex(index=existing, columns=existing, fill_value=0)

	return z_dist, z_next_dist, freq, prob


def format_numeric_tables(product_type_dist, product_subtype_dist, type_amount, subtype_amount, flag_summary, z_dist, z_next_dist, freq, prob):
	for df in [product_type_dist, product_subtype_dist, flag_summary, z_dist, z_next_dist]:
		df["占比"] = df["占比"].map(lambda x: f"{x:.4%}")
		for col in ["订单数", "样本数"]:
			if col in df.columns:
				df[col] = df[col].map(lambda x: f"{int(x):,}")

	for df in [type_amount, subtype_amount]:
		df["订单数"] = df["订单数"].map(lambda x: f"{int(x):,}")
		df["金额总和"] = df["金额总和"].map(lambda x: f"{float(x):,.2f}")
		df["平均金额"] = df["平均金额"].map(lambda x: f"{float(x):,.2f}")

	freq_fmt = freq.reset_index().rename(columns={"Z_t": "当前状态"})
	for col in freq_fmt.columns[1:]:
		freq_fmt[col] = freq_fmt[col].map(lambda x: f"{int(x):,}")

	prob_fmt = prob.reset_index().rename(columns={"Z_t": "当前状态"})
	for col in prob_fmt.columns[1:]:
		prob_fmt[col] = prob_fmt[col].map(lambda x: f"{float(x):.4f}")

	return freq_fmt, prob_fmt


def write_outputs():
	OUT_DIR.mkdir(parents=True, exist_ok=True)

	orders = pd.read_csv(CLASSIFIED_ORDERS)
	transitions = pd.read_csv(STATE_TRANSITIONS)

	product_type_dist, product_subtype_dist, type_amount, subtype_amount, flag_summary = summarize_product_classification(orders)
	z_dist, z_next_dist, freq, prob = summarize_state_transition(transitions)

	product_type_dist.to_csv(OUT_DIR / "5.1_商品大类分布.csv", index=False, encoding="utf-8-sig")
	product_subtype_dist.to_csv(OUT_DIR / "5.1_商品子类分布.csv", index=False, encoding="utf-8-sig")
	type_amount.to_csv(OUT_DIR / "5.1_商品大类金额汇总.csv", index=False, encoding="utf-8-sig")
	subtype_amount.to_csv(OUT_DIR / "5.1_商品子类金额汇总.csv", index=False, encoding="utf-8-sig")
	flag_summary.to_csv(OUT_DIR / "5.1_商品标记汇总.csv", index=False, encoding="utf-8-sig")
	z_dist.to_csv(OUT_DIR / "5.2_当前状态分布.csv", index=False, encoding="utf-8-sig")
	z_next_dist.to_csv(OUT_DIR / "5.2_下一期状态分布.csv", index=False, encoding="utf-8-sig")
	freq.to_csv(OUT_DIR / "5.2_状态转移频数矩阵.csv", encoding="utf-8-sig")
	prob.to_csv(OUT_DIR / "5.2_状态转移概率矩阵.csv", encoding="utf-8-sig")

	freq_fmt, prob_fmt = format_numeric_tables(
		product_type_dist,
		product_subtype_dist,
		type_amount,
		subtype_amount,
		flag_summary,
		z_dist,
		z_next_dist,
		freq,
		prob,
	)

	report = f"""#5.1 商品分类与5.2 状态转移结果

本文件由 `模型输出结果/generate_product_and_state_results.py` 自动生成，可直接作为论文第5.1节和第5.2节结果整理的基础。

##5.1 商品分类与权益周期识别结果

### 商品大类分布

{md_table(product_type_dist)}

### 商品子类分布

{md_table(product_subtype_dist)}

### 各商品大类订单量与金额

{md_table(type_amount)}

### 各商品子类订单量与金额

{md_table(subtype_amount)}

### 商品标记汇总

{md_table(flag_summary)}

###论文文字分析建议

由商品分类结果可见，会员类商品是平台订单的主体，订单量占比最高；私教商品订单量相对较少，但平均订单金额明显高于会员商品，说明私教产品具有更高单笔收入贡献。赠送权益或0元权益订单占比较低，建模时不作为正常收入订单计入，但可用于辅助判断用户权益状态。需要人工复核的订单占比较低，说明基于商品名称、价格和规则的自动分类结果总体可用于后续会员续费、私教购买和收入模拟分析。

在收入统计中，需要注意私教订单若已包含在总订单数据中，不能在总收入中重复计入。本文后续收益模型将会员收入和私教收入分别建模，并在利润函数中按照会员卡35%、私教卡30%的毛利率计算平台经营收益。

##5.2 用户状态划分与状态转移结果

### 当前状态分布

{md_table(z_dist)}

### 下一期状态分布

{md_table(z_next_dist)}

### 状态转移频数矩阵

{md_table(freq_fmt)}

### 状态转移概率矩阵

{md_table(prob_fmt)}

###论文文字分析建议

从用户状态分布看，真实数据中并非7类理论状态均有充足样本，最终实证结果主要集中在高活跃高消费用户、私教潜在用户和流失风险用户三类。这说明线上互动、无氧训练和高消费行为在自然月尺度上具有明显稀疏性，因此有必要将样本量过少的理论状态进行实证简化，重点分析具有运营解释价值的核心状态。

状态转移矩阵表明，流失风险用户具有较强的状态惯性，其下一期仍保持流失风险状态的概率最高。这意味着仅依赖用户自然恢复较难显著提升整体活跃度，需要通过召回型积分或低门槛激励引导其重新到店。私教潜在用户下一期保持私教潜在状态和转入流失风险状态的概率均较高，说明该类用户具有一定训练基础或私教转化可能，但若缺少持续激励和服务承接，也容易转入低活跃状态。高活跃高消费用户保持原状态的概率相对较高，但仍存在转入流失风险状态的可能，因此对该类用户更适合采用低补贴、高体验感的保留型积分策略。

上述结果说明，用户状态自然演化存在明显差异，统一积分规则难以同时兼顾用户召回、私教转化和高价值用户保留。后续积分策略模拟应围绕核心用户状态设计差异化候选策略，并结合续费概率、私教购买概率、积分成本和平台利润进行综合评价。
"""

	report_path = OUT_DIR / "5.1_商品分类与5.2状态转移结果.md"
	report_path.write_text(report, encoding="utf-8")

	print(f"已生成结果文件：{report_path}")


if __name__ == "__main__":
	write_outputs()
