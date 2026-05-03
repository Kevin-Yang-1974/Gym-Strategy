# -*- coding: utf-8 -*-
"""运行 H=30会员续费 Cox 生存模型主模型。"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "数据整理程序输出" / "整理后的数据"
OUT_DIR = BASE_DIR / "模型输出结果"

DATA_PATH = DATA_DIR / "renewal_survival_dataset_H30.csv"
REPORT_PATH = OUT_DIR / "5.3_会员续费Cox生存模型_H30结果.md"
COEF_PATH = OUT_DIR / "5.3_Cox_H30_系数风险比.csv"
PRED_PATH = OUT_DIR / "5.3_Cox_H30_预测续费概率.csv"
SUMMARY_PATH = OUT_DIR / "5.3_Cox_H30_模型摘要.csv"

FEATURES = [
	"current_member_duration_days",
	"current_member_price",
	"visit_count",
	"stay_hours_total",
	"run_distance_total",
	"run_minutes_total",
	"calorie_total",
	"anaerobic_volume",
	"anaerobic_count",
	"anaerobic_calorie_total",
	"post_count",
	"like_given_count",
	"comment_given_count",
	"like_received_count",
	"comment_received_count",
	"order_amount_total",
	"order_count",
	"order_amount_avg",
	"valid_revenue_amount",
	"member_amount",
	"member_order_count",
	"pt_amount",
	"pt_order_count",
	"discount_order_ratio",
	"pt_private_amount",
	"pt_sessions_bought",
	"days_since_last_visit",
	"days_since_last_buy",
	"days_since_last_active",
	"days_since_last_interact",
	"active_member",
	"dte",
	"pt_bought_before",
	"pt_amount_total_before",
	"pt_sessions_bought_before",
	"days_since_last_pt_buy",
	"pt_unit_price_est",
	"A_score",
	"S_score",
]

LOG_FEATURES = [
	"current_member_price",
	"visit_count",
	"stay_hours_total",
	"run_distance_total",
	"run_minutes_total",
	"calorie_total",
	"anaerobic_volume",
	"anaerobic_count",
	"anaerobic_calorie_total",
	"post_count",
	"like_given_count",
	"comment_given_count",
	"like_received_count",
	"comment_received_count",
	"order_amount_total",
	"order_count",
	"order_amount_avg",
	"valid_revenue_amount",
	"member_amount",
	"member_order_count",
	"pt_amount",
	"pt_order_count",
	"discount_order_ratio",
	"pt_private_amount",
	"pt_sessions_bought",
	"days_since_last_visit",
	"days_since_last_buy",
	"days_since_last_active",
	"days_since_last_interact",
	"pt_amount_total_before",
	"pt_sessions_bought_before",
	"days_since_last_pt_buy",
	"pt_unit_price_est",
	"A_score",
	"S_score",
]

RENAME = {
	"current_member_duration_days": "当前会员权益时长",
	"current_member_price": "当前会员价格",
	"visit_count": "到店次数",
	"stay_hours_total": "在馆时长",
	"run_distance_total": "跑步距离",
	"run_minutes_total": "跑步时长",
	"calorie_total": "运动消耗",
	"anaerobic_volume": "无氧训练量",
	"anaerobic_count": "无氧训练次数",
	"anaerobic_calorie_total": "无氧消耗",
	"post_count": "发帖数",
	"like_given_count": "主动点赞数",
	"comment_given_count": "主动评论数",
	"like_received_count": "被点赞数",
	"comment_received_count": "被评论数",
	"order_amount_total": "订单总额",
	"order_count": "订单次数",
	"order_amount_avg": "平均客单价",
	"valid_revenue_amount": "有效收入金额",
	"member_amount": "会员消费金额",
	"member_order_count": "会员订单次数",
	"pt_amount": "私教消费金额",
	"pt_order_count": "私教订单次数",
	"discount_order_ratio": "优惠订单占比",
	"pt_private_amount": "私教课包金额",
	"pt_sessions_bought": "本期私教课时",
	"days_since_last_visit": "距上次到店天数",
	"days_since_last_buy": "距上次购买天数",
	"days_since_last_active": "距上次活跃天数",
	"days_since_last_interact": "距上次互动天数",
	"active_member": "是否有效会员",
	"dte": "剩余会员天数",
	"pt_bought_before": "历史是否买过私教",
	"pt_amount_total_before": "历史私教消费金额",
	"pt_sessions_bought_before": "历史私教课时",
	"days_since_last_pt_buy": "距上次私教购买天数",
	"pt_unit_price_est": "私教单价估计",
	"A_score": "线下活跃度得分",
	"S_score": "线上互动强度得分",
}


class CoxBreslowModel:
	def __init__(self, ridge=1e-4):
		self.ridge = ridge
		self.beta = None
		self.result = None

	def _prepare_sorted(self, x, duration, event):
		order = np.argsort(duration, kind="mergesort")
		return x[order], duration[order], event[order]

	def _loss_grad(self, beta, x, duration, event):
		x_sorted, duration_sorted, event_sorted = self._prepare_sorted(x, duration, event)
		xb = x_sorted @ beta
		xb = np.clip(xb, -50,50)
		exp_xb = np.exp(xb)

		risk_sum = np.cumsum(exp_xb[::-1])[::-1]
		risk_x_sum = np.cumsum((exp_xb[:, None] * x_sorted)[::-1], axis=0)[::-1]

		unique_event_times = np.unique(duration_sorted[event_sorted ==1])
		loglik =0.0
		grad = np.zeros_like(beta)

		for t in unique_event_times:
			start = np.searchsorted(duration_sorted, t, side="left")
			end = np.searchsorted(duration_sorted, t, side="right")
			event_mask = event_sorted[start:end] ==1
			d = int(event_mask.sum())
			if d ==0:
				continue
			event_xb_sum = xb[start:end][event_mask].sum()
			event_x_sum = x_sorted[start:end][event_mask].sum(axis=0)
			denom = risk_sum[start]
			denom_x = risk_x_sum[start]
			loglik += event_xb_sum - d * np.log(denom)
			grad += event_x_sum - d * denom_x / denom

		penalty =0.5 * self.ridge * np.sum(beta * beta)
		loss = -loglik + penalty
		grad = -grad + self.ridge * beta
		return loss, grad

	def fit(self, x, duration, event):
		beta0 = np.zeros(x.shape[1])
		result = minimize(
			fun=lambda b: self._loss_grad(b, x, duration, event)[0],
			x0=beta0,
			jac=lambda b: self._loss_grad(b, x, duration, event)[1],
			method="L-BFGS-B",
			options={"maxiter":300, "ftol":1e-7, "gtol":1e-5},
		)
		self.beta = result.x
		self.result = result
		return self

	def predict_risk_score(self, x):
		return x @ self.beta

	def baseline_survival_at(self, x, duration, event, horizon):
		x_sorted, duration_sorted, event_sorted = self._prepare_sorted(x, duration, event)
		xb = np.clip(x_sorted @ self.beta, -50,50)
		exp_xb = np.exp(xb)
		risk_sum = np.cumsum(exp_xb[::-1])[::-1]

		unique_event_times = np.unique(duration_sorted[(event_sorted ==1) & (duration_sorted <= horizon)])
		cum_hazard =0.0
		for t in unique_event_times:
			start = np.searchsorted(duration_sorted, t, side="left")
			end = np.searchsorted(duration_sorted, t, side="right")
			d = int((event_sorted[start:end] ==1).sum())
			if d >0:
				cum_hazard += d / risk_sum[start]
		return float(np.exp(-cum_hazard))


def preprocess(df):
	model_df = df[["duration", "event", "censored", "renew_in_H"] + FEATURES].copy()
	model_df["duration"] = pd.to_numeric(model_df["duration"], errors="coerce").fillna(0.0)
	model_df["duration"] = model_df["duration"].clip(lower=1e-6)
	model_df["event"] = pd.to_numeric(model_df["event"], errors="coerce").fillna(0).astype(int)

	for col in FEATURES:
		model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
		median = model_df[col].median()
		if pd.isna(median):
			median =0.0
		model_df[col] = model_df[col].fillna(median)
		if col in LOG_FEATURES:
			min_value = model_df[col].min()
			if min_value <0:
				model_df[col] = model_df[col] - min_value
			model_df[col] = np.log1p(model_df[col])

	x_raw = model_df[FEATURES].to_numpy(dtype=float)
	means = x_raw.mean(axis=0)
	stds = x_raw.std(axis=0)
	stds[stds ==0] =1.0
	x = (x_raw - means) / stds
	duration = model_df["duration"].to_numpy(dtype=float)
	event = model_df["event"].to_numpy(dtype=int)
	return model_df, x, duration, event, means, stds


def approximate_c_index(duration, event, risk_score, max_pairs=300000, seed=2026):
	rng = np.random.default_rng(seed)
	event_idx = np.where(event ==1)[0]
	if len(event_idx) ==0:
		return np.nan,0

	concordant =0.0
	comparable =0
	attempts =0
	max_attempts = max_pairs *20
	n = len(duration)

	while comparable < max_pairs and attempts < max_attempts:
		i = int(rng.choice(event_idx))
		j = int(rng.integers(0, n))
		attempts +=1
		if duration[i] >= duration[j]:
			continue
		comparable +=1
		if risk_score[i] > risk_score[j]:
			concordant +=1
		elif risk_score[i] == risk_score[j]:
			concordant +=0.5

	if comparable ==0:
		return np.nan,0
	return concordant / comparable, comparable


def get_standard_errors(result, p):
	try:
		hess_inv = result.hess_inv.todense()
		se = np.sqrt(np.maximum(np.diag(hess_inv),0))
		if len(se) == p:
			return se
	except Exception:
		pass
	return np.full(p, np.nan)


def build_report(coef_df, summary, pred_summary):
	top_pos = coef_df.sort_values("HR", ascending=False).head(8)
	top_neg = coef_df.sort_values("HR", ascending=True).head(8)

	report = f"""#5.3会员续费 Cox 生存模型 H=30 主模型结果

本文件由 `模型输出结果/run_renewal_cox_h30.py` 自动生成，用于整理论文第5.3节中会员续费 Cox 生存模型主结果。

##1. 模型设定

本文以会员到期后的续费等待时间作为生存时间变量，建立 Cox 比例风险模型：

$$
h_i(t)=h_0(t)\\exp(\\beta^TX_i)
$$

其中，`duration` 表示用户从会员到期到续费或删失的观察时间，`event=1` 表示观察到续费事件，`event=0` 表示未观察到续费或右删失。Cox 模型能够自然纳入右删失样本，因此本节主模型使用 `renewal_survival_dataset_H30.csv` 中全部可用样本。

为降低极端值影响，金额、次数、时长和间隔类变量进入模型前使用 `log1p`变换，并对所有特征做标准化处理。因此，系数和风险比反映的是变量标准化后一单位变化对续费风险率的相对影响。

##2. 样本与模型摘要

| 指标 | 数值 |
|---|---:|
| 样本量 | {summary['样本量']:,} |
|续费事件数 | {summary['续费事件数']:,} |
|续费事件率 | {summary['续费事件率']:.4%} |
|右删失样本数 | {summary['右删失样本数']:,} |
|右删失比例 | {summary['右删失比例']:.4%} |
| H |30 |
| 优化是否收敛 | {summary['优化是否收敛']} |
|负部分对数似然 | {summary['负部分对数似然']:.4f} |
|近似 Harrell C-index | {summary['近似C_index']:.4f} |
| C-index抽样可比对数 | {summary['C_index抽样可比对数']:,} |
| H=30窗口预测AUC | {summary['H30预测AUC']:.4f} |
| H=30窗口Brier Score | {summary['H30_Brier_Score']:.4f} |

说明：C-index 使用随机抽样的可比样本对近似计算，用于衡量模型对续费早晚顺序的区分能力；AUC 和 Brier Score 是将 Cox 模型在 $H=30$ 下的预测续费概率与窗口内续费标签进行比较得到，主要用于辅助展示预测能力。

##3. Cox模型系数与风险比

{coef_df.to_markdown(index=False)}

##4. 风险比方向解释

风险比 $HR=\\exp(\\beta)$。当 $HR>1$ 时，说明该变量增大与更高的续费风险率相关，即用户更倾向于更早续费；当 $HR<1$ 时，说明该变量增大与较低的续费风险率相关，即续费倾向较弱或续费时间更晚。

### 风险比最高的变量

{top_pos[['变量', '中文含义', 'HR', 'p值']].to_markdown(index=False)}

### 风险比最低的变量

{top_neg[['变量', '中文含义', 'HR', 'p值']].to_markdown(index=False)}

##5.预测续费概率分布

| 指标 | 数值 |
|---|---:|
| 平均预测续费概率 | {pred_summary['mean']:.4f} |
| 标准差 | {pred_summary['std']:.4f} |
| 最小值 | {pred_summary['min']:.4f} |
|25%分位数 | {pred_summary['25%']:.4f} |
| 中位数 | {pred_summary['50%']:.4f} |
|75%分位数 | {pred_summary['75%']:.4f} |
| 最大值 | {pred_summary['max']:.4f} |

##6.论文表述建议

从 Cox 生存模型结果看，模型将续费行为刻画为会员到期后的时间到事件问题，并保留了未观察到续费事件的右删失样本。相比简单二分类模型，Cox 模型不仅可以估计用户是否会在观察窗口内续费，还能利用续费等待时间信息解释不同用户特征与续费快慢之间的关系。

若某些变量的风险比大于1且具有统计显著性，可解释为该变量升高会提高用户在到期后较短时间内续费的相对风险率；若风险比小于1，则说明该变量升高与较低的续费风险率相关。由于本文使用的是历史观测数据，相关结果应解释为行为特征与续费倾向之间的统计关联，而不能直接解释为因果效应。

后续应使用 `renewal_survival_dataset_H60.csv` 建立 $H=60$ Cox稳健性模型，并比较主要变量方向、风险比和预测能力是否稳定。
"""
	return report


def main():
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	df = pd.read_csv(DATA_PATH)
	model_df, x, duration, event, means, stds = preprocess(df)

	model = CoxBreslowModel(ridge=1e-4).fit(x, duration, event)
	risk_score = model.predict_risk_score(x)
	baseline_s30 = model.baseline_survival_at(x, duration, event,30.0)
	pred_prob =1.0 - np.power(baseline_s30, np.exp(np.clip(risk_score, -50,50)))
	pred_prob = np.clip(pred_prob,0,1)

	se = get_standard_errors(model.result, len(FEATURES))
	z_value = model.beta / se
	p_value =2 * (1 - norm.cdf(np.abs(z_value)))
	coef_df = pd.DataFrame({
		"变量": FEATURES,
		"中文含义": [RENAME.get(c, c) for c in FEATURES],
		"系数beta": model.beta,
		"标准误近似": se,
		"z值近似": z_value,
		"p值": p_value,
		"HR": np.exp(model.beta),
	})
	coef_df["显著性"] = pd.cut(
		coef_df["p值"],
		bins=[-np.inf,0.001,0.01,0.05,0.1, np.inf],
		labels=["***", "**", "*", ".", ""],
	)
	coef_df = coef_df.sort_values("HR", ascending=False)

	c_index, comparable_pairs = approximate_c_index(duration, event, risk_score)
	auc = roc_auc_score(model_df["renew_in_H"].astype(int), pred_prob)
	brier = brier_score_loss(model_df["renew_in_H"].astype(int), pred_prob)

	summary = {
		"样本量": len(model_df),
		"续费事件数": int(model_df["event"].sum()),
		"续费事件率": float(model_df["event"].mean()),
		"右删失样本数": int(model_df["censored"].sum()),
		"右删失比例": float(model_df["censored"].mean()),
		"优化是否收敛": bool(model.result.success),
		"负部分对数似然": float(model.result.fun),
		"近似C_index": float(c_index),
		"C_index抽样可比对数": int(comparable_pairs),
		"H30预测AUC": float(auc),
		"H30_Brier_Score": float(brier),
	}

	pred_out = pd.DataFrame({
		"duration": model_df["duration"],
		"event": model_df["event"],
		"censored": model_df["censored"],
		"renew_in_H": model_df["renew_in_H"],
		"risk_score": risk_score,
		"pred_renew_prob_H30": pred_prob,
	})

	pred_summary = pred_out["pred_renew_prob_H30"].describe().to_dict()
	coef_df.to_csv(COEF_PATH, index=False, encoding="utf-8-sig")
	pred_out.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")
	pd.DataFrame([summary]).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

	report = build_report(coef_df, summary, pred_summary)
	REPORT_PATH.write_text(report, encoding="utf-8")
	print(f"已生成：{REPORT_PATH}")


if __name__ == "__main__":
	main()
