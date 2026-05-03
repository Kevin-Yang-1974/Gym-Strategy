# -*- coding: utf-8 -*-
"""比较状态增强随机森林、普通随机森林、Logistic-GAM近似和 Cox 的 H=60续费概率预测效果。"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "数据整理程序输出" / "整理后的数据"
OUT_DIR = BASE_DIR / "模型输出结果"
DATA_PATH = DATA_DIR / "renewal_survival_dataset_H60.csv"

REPORT_PATH = OUT_DIR / "5.3_状态增强随机森林与对比模型结果_H60.md"
METRIC_PATH = OUT_DIR / "5.3_状态增强随机森林与对比模型指标_H60.csv"
IMPORTANCE_PATH = OUT_DIR / "5.3_状态增强随机森林特征重要性_H60.csv"
PRED_PATH = OUT_DIR / "5.3_状态增强随机森林与对比模型预测_H60.csv"
SERF_MODEL_PATH = OUT_DIR / "serf_renewal_h60_model.joblib"
SERF_META_PATH = OUT_DIR / "serf_renewal_h60_features.json"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] =150

BASE_FEATURES = [
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
	"pt_amount",
	"pt_order_count",
	"discount_order_ratio",
	"pt_private_amount",
	"pt_sessions_bought",
	"pt_bought_before",
	"pt_amount_total_before",
	"pt_sessions_bought_before",
	"pt_unit_price_est",
	"A_score",
	"S_score",
]
CATEGORICAL_FEATURES = ["current_member_type", "user_state"]
RESPONSE_FEATURES = ["active_gap", "o2o_strength", "pt_potential", "buy_stickiness", "visit_recency_score", "active_recency_score", "interact_recency_score"]
RESPONSE_SOURCE_FEATURES = [
	"days_since_last_visit",
	"days_since_last_buy",
	"days_since_last_active",
	"days_since_last_interact",
]
STATE_RF_FEATURES = BASE_FEATURES + RESPONSE_FEATURES + CATEGORICAL_FEATURES
PLAIN_RF_FEATURES = BASE_FEATURES + CATEGORICAL_FEATURES
GAM_SPLINE_FEATURES = ["visit_count", "current_member_price", "A_score", "S_score"]
GAM_LINEAR_FEATURES = [f for f in BASE_FEATURES if f not in GAM_SPLINE_FEATURES]

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
	"pt_amount": "私教消费金额",
	"pt_order_count": "私教订单次数",
	"discount_order_ratio": "优惠订单占比",
	"pt_private_amount": "私教课包金额",
	"pt_sessions_bought": "本期私教课时",
	"days_since_last_visit": "距上次到店天数",
	"days_since_last_buy": "距上次购买天数",
	"days_since_last_active": "距上次活跃天数",
	"days_since_last_interact": "距上次互动天数",
	"pt_bought_before": "历史是否买过私教",
	"pt_amount_total_before": "历史私教消费金额",
	"pt_sessions_bought_before": "历史私教课时",
	"days_since_last_pt_buy": "距上次私教购买天数",
	"pt_unit_price_est": "私教单价估计",
	"A_score": "线下活跃度得分",
	"S_score": "线上互动强度得分",
	"current_member_type": "当前会员类型",
	"user_state": "用户状态",
	"active_gap": "活跃提升空间",
	"o2o_strength": "线上线下联动强度",
	"pt_potential": "私教转化潜力",
	"buy_stickiness": "购买粘性",
	"visit_recency_score": "到店近因得分",
	"active_recency_score": "活跃近因得分",
	"interact_recency_score": "互动近因得分",
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
		xb = np.clip(x_sorted @ beta, -50,50)
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
			denom = risk_sum[start]
			loglik += xb[start:end][event_mask].sum() - d * np.log(denom)
			grad += x_sorted[start:end][event_mask].sum(axis=0) - d * risk_x_sum[start] / denom
		loss = -loglik +0.5 * self.ridge * np.sum(beta * beta)
		grad = -grad + self.ridge * beta
		return loss, grad

	def fit(self, x, duration, event):
		result = minimize(
			fun=lambda b: self._loss_grad(b, x, duration, event)[0],
			x0=np.zeros(x.shape[1]),
			jac=lambda b: self._loss_grad(b, x, duration, event)[1],
			method="L-BFGS-B",
			options={"maxiter":300, "ftol":1e-7, "gtol":1e-5},
		)
		self.beta = result.x
		self.result = result
		return self

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

	def predict_prob(self, x_train, train_duration, train_event, x_eval):
		baseline_s60 = self.baseline_survival_at(x_train, train_duration, train_event,60.0)
		risk_score = x_eval @ self.beta
		pred =1.0 - np.power(baseline_s60, np.exp(np.clip(risk_score, -50,50)))
		return np.clip(pred,0,1)


def log1p_array(x):
	return np.log1p(np.maximum(x,0))


def add_response_features(df, train_mask=None, active_target=None):
	out = df.copy()
	if active_target is None:
		if train_mask is None:
			train_mask = np.ones(len(out), dtype=bool)
		active_target = float(out.loc[train_mask, "A_score"].quantile(0.75))
	out["active_gap"] = np.maximum(active_target - out["A_score"].fillna(0), 0)
	out["o2o_strength"] = out["A_score"].fillna(0) * np.log1p(np.maximum(out["S_score"].fillna(0), 0))
	out["pt_potential"] = np.log1p(np.maximum(out["anaerobic_count"].fillna(0), 0)) * (
				1 - out["pt_bought_before"].fillna(0).clip(0, 1))
	out["buy_stickiness"] = np.log1p(np.maximum(out["pt_amount_total_before"].fillna(0), 0)) / (
				out["days_since_last_buy"].fillna(999) + 1)
	out["visit_recency_score"] = 1 / (out["days_since_last_visit"].fillna(999) + 1)
	out["active_recency_score"] = 1 / (out["days_since_last_active"].fillna(999) + 1)
	out["interact_recency_score"] = 1 / (out["days_since_last_interact"].fillna(999) + 1)
	return out, active_target


def load_data():
	df = pd.read_csv(DATA_PATH, low_memory=False)
	df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
	df = df.dropna(subset=["end_date"])
	df["renew_in_H"] = pd.to_numeric(df["renew_in_H"], errors="coerce").fillna(0).astype(int)
	df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(int)
	df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).clip(lower=1e-6)
	df["censored"] = pd.to_numeric(df["censored"], errors="coerce").fillna(0).astype(int)
	for col in list(dict.fromkeys(BASE_FEATURES + RESPONSE_SOURCE_FEATURES)):
		df[col] = pd.to_numeric(df[col], errors="coerce")
	for col in CATEGORICAL_FEATURES:
		df[col] = df[col].astype("object").where(df[col].notna(), "缺失")
	return df


def split_binary(df):
	binary_df = df[df["censored"] ==0].copy()
	cutoff = binary_df["end_date"].quantile(0.8)
	train_mask = (binary_df["end_date"] <= cutoff).to_numpy()
	binary_df, active_target = add_response_features(binary_df, train_mask)
	train_df = binary_df[binary_df["end_date"] <= cutoff].copy()
	test_df = binary_df[binary_df["end_date"] > cutoff].copy()
	return train_df, test_df, cutoff, active_target


def split_cox(df, cutoff, active_target):
	train_df = df[df["end_date"] <= cutoff].copy()
	test_df = df[(df["end_date"] > cutoff) & (df["censored"] ==0)].copy()
	train_df, _ = add_response_features(train_df, active_target=active_target)
	test_df, _ = add_response_features(test_df, active_target=active_target)
	return train_df, test_df


def numeric_pipe(scale=True):
	steps = [("imputer", SimpleImputer(strategy="median")), ("log", FunctionTransformer(log1p_array, feature_names_out="one-to-one"))]
	if scale:
		steps.append(("scaler", StandardScaler()))
	return Pipeline(steps)


def categorical_pipe():
	return Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50))])


def make_rf(features):
	num_features = [f for f in features if f not in CATEGORICAL_FEATURES]
	cat_features = [f for f in features if f in CATEGORICAL_FEATURES]
	preprocess = ColumnTransformer([("num", numeric_pipe(False), num_features), ("cat", categorical_pipe(), cat_features)])
	model = RandomForestClassifier(n_estimators=220, max_depth=12, min_samples_leaf=80, class_weight="balanced_subsample", n_jobs=-1, random_state=2026)
	return Pipeline([("preprocess", preprocess), ("model", model)])


def make_gam():
	preprocess = ColumnTransformer([
		("spline", Pipeline([("imputer", SimpleImputer(strategy="median")), ("log", FunctionTransformer(log1p_array, feature_names_out="one-to-one")), ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)), ("scaler", StandardScaler())]), GAM_SPLINE_FEATURES),
		("linear", numeric_pipe(True), GAM_LINEAR_FEATURES),
		("cat", categorical_pipe(), CATEGORICAL_FEATURES),
	])
	return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def threshold_by_f1(y, prob):
	best_t =0.5
	best_f1 = -1
	for t in np.linspace(0.05,0.95,91):
		pred = (prob >= t).astype(int)
		value = f1_score(y, pred, zero_division=0)
		if value > best_f1:
			best_f1 = value
			best_t = t
	return best_t


def evaluate_probability(name, y_train, train_prob, y_test, test_prob):
	threshold = threshold_by_f1(y_train, train_prob)
	test_pred = (test_prob >= threshold).astype(int)
	return {
		"模型": name,
		"阈值": float(threshold),
		"测试AUC": float(roc_auc_score(y_test, test_prob)),
		"测试F1": float(f1_score(y_test, test_pred, zero_division=0)),
		"测试Recall": float(recall_score(y_test, test_pred, zero_division=0)),
		"测试Precision": float(precision_score(y_test, test_pred, zero_division=0)),
		"测试Brier": float(brier_score_loss(y_test, test_prob)),
	}


def fit_predict_classifier(name, model, features, train_df, test_df):
	y_train = train_df["renew_in_H"].to_numpy(dtype=int)
	y_test = test_df["renew_in_H"].to_numpy(dtype=int)
	model.fit(train_df[features], y_train)
	train_prob = model.predict_proba(train_df[features])[:,1]
	test_prob = model.predict_proba(test_df[features])[:,1]
	metric = evaluate_probability(name, y_train, train_prob, y_test, test_prob)
	return metric, test_prob, model


def fit_predict_cox(train_df, test_df):
	features = BASE_FEATURES
	x_train_raw = train_df[features].copy()
	x_test_raw = test_df[features].copy()
	for col in features:
		median = x_train_raw[col].median()
		if pd.isna(median):
			median =0.0
		x_train_raw[col] = x_train_raw[col].fillna(median)
		x_test_raw[col] = x_test_raw[col].fillna(median)
		x_train_raw[col] = np.log1p(np.maximum(x_train_raw[col],0))
		x_test_raw[col] = np.log1p(np.maximum(x_test_raw[col],0))
	means = x_train_raw.mean(axis=0)
	stds = x_train_raw.std(axis=0).replace(0,1)
	x_train = ((x_train_raw - means) / stds).to_numpy(dtype=float)
	x_test = ((x_test_raw - means) / stds).to_numpy(dtype=float)
	train_duration = train_df["duration"].to_numpy(dtype=float)
	train_event = train_df["event"].to_numpy(dtype=int)
	model = CoxBreslowModel(ridge=1e-4).fit(x_train, train_duration, train_event)
	train_prob = model.predict_prob(x_train, train_duration, train_event, x_train)
	test_prob = model.predict_prob(x_train, train_duration, train_event, x_test)
	y_train = train_df["renew_in_H"].to_numpy(dtype=int)
	y_test = test_df["renew_in_H"].to_numpy(dtype=int)
	metric = evaluate_probability("Cox", y_train, train_prob, y_test, test_prob)
	return metric, test_prob


def rf_importance(model, features):
	pre = model.named_steps["preprocess"]
	rf = model.named_steps["model"]
	feature_names = pre.get_feature_names_out()
	rows = []
	for name, value in zip(feature_names, rf.feature_importances_):
		clean = name.split("__",1)[-1]
		matched = None
		for f in features:
			if clean == f or clean.startswith(f + "_"):
				matched = f
				break
		if matched is None:
			matched = clean
		rows.append({"原始变量": matched, "中文含义": RENAME.get(matched, matched), "重要性": value})
	importance = pd.DataFrame(rows).groupby(["原始变量", "中文含义"], as_index=False)["重要性"].sum()
	return importance.sort_values("重要性", ascending=False)


def export_serf_artifacts(model, active_target):
	joblib.dump(model, SERF_MODEL_PATH)
	meta = {
		"model_name": "状态增强随机森林_H60",
		"feature_order": STATE_RF_FEATURES,
		"base_features": BASE_FEATURES,
		"response_features": RESPONSE_FEATURES,
		"response_source_features": RESPONSE_SOURCE_FEATURES,
		"categorical_features": CATEGORICAL_FEATURES,
		"active_target": float(active_target),
		"response_feature_mapping": {
			"active_gap": "max(active_target - A_score,0)",
			"o2o_strength": "A_score * log1p(max(S_score,0))",
			"pt_potential": "log1p(max(anaerobic_count,0)) * (1 - clip(pt_bought_before,0,1))",
			"buy_stickiness": "log1p(max(pt_amount_total_before,0)) / (days_since_last_buy +1)",
			"visit_recency_score": "1 / (days_since_last_visit +1)",
			"active_recency_score": "1 / (days_since_last_active +1)",
			"interact_recency_score": "1 / (days_since_last_interact +1)",
		},
	}
	SERF_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def save_metric_plot(metric_df):
	x = np.arange(len(metric_df))
	width =0.18
	fig, ax = plt.subplots(figsize=(9,5))
	for i, col in enumerate(["测试AUC", "测试F1", "测试Recall", "测试Brier"]):
		ax.bar(x + (i -1.5) * width, metric_df[col], width, label=col)
	ax.set_xticks(x)
	ax.set_xticklabels(metric_df["模型"], rotation=10)
	ax.set_ylim(0,1)
	ax.set_title("状态增强随机森林与对比模型测试表现")
	ax.legend()
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.3_图13_状态增强随机森林与对比模型指标_H60.png", bbox_inches="tight")
	plt.close(fig)


def save_probability_plot(pred_df):
	fig, ax = plt.subplots(figsize=(8,5))
	for col in ["状态增强随机森林_prob", "普通随机森林_prob", "Logistic-GAM_prob", "Cox_prob"]:
		ax.hist(pred_df[col], bins=35, alpha=0.42, label=col.replace("_prob", ""))
	ax.set_xlabel("预测30天内续费概率")
	ax.set_ylabel("测试集样本数")
	ax.set_title("各模型预测续费概率分布")
	ax.legend()
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.3_图14_状态增强随机森林与对比模型概率分布_H60.png", bbox_inches="tight")
	plt.close(fig)


def save_importance_plot(importance):
	plot_df = importance.head(15).sort_values("重要性")
	fig, ax = plt.subplots(figsize=(8,6))
	ax.barh(plot_df["中文含义"], plot_df["重要性"], color="#5b8db8")
	ax.set_xlabel("特征重要性")
	ax.set_title("状态增强随机森林特征重要性Top15")
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.3_图15_状态增强随机森林特征重要性Top15_H60.png", bbox_inches="tight")
	plt.close(fig)


def write_report(metric_df, importance, train_df, test_df, cutoff):
	best = metric_df.sort_values(["测试AUC", "测试Brier"], ascending=[False, True]).iloc[0]
	report = f"""#5.3状态增强随机森林与对比模型结果 H=60

本文件由 `模型输出结果/compare_state_enhanced_rf_h60.py` 自动生成，用于比较本文提出的状态增强随机森林模型与普通随机森林、Logistic-GAM近似模型和 Cox生存模型在 H=60续费概率预测上的表现。

##1. 模型说明

状态增强随机森林在普通行为特征基础上加入两类场景化信息：

1. 用户状态变量 $Z_{{i,t}}$，包括高活跃高消费用户、私教潜在用户和流失风险用户等；
2. 行为响应特征，包括活跃提升空间、线上线下联动强度、私教转化潜力、购买粘性、到店近因得分、活跃近因得分和互动近因得分。

其中，流失风险向量采用论文框架中的最新口径：

$$
L_{{i,t}}=(Stickiness_i,VisitRecency_i,ActiveRecency_i,InteractRecency_i)
$$

其输入特征可表示为：

$$
X_i^{{SERF}}=(A_{{i,t}},S_{{i,t}},M_{{i,t}},L_{{i,t}},PT_{{i,t}},Z_{{i,t}},GapA_i,O2O_i,PTPotential_i,Stickiness_i)
$$

随机森林预测概率为：

$$
\\hat p_i^{{renew,60}}=\\frac1B\\sum_{{b=1}}^B T_b(X_i^{{SERF}})
$$

##2. 样本处理

二分类模型使用 `censored=0` 的完整观察样本。为了与 Cox比较，Cox模型使用训练期全部样本估计生存风险，并在相同测试集上输出 $H=60$续费概率。样本按 `end_date` 做时间切分，切分日期为 `{cutoff.date()}`。

| 数据集 | 样本量 |续费率 |
|---|---:|---:|
|训练集 | {len(train_df):,} | {train_df['renew_in_H'].mean():.4%} |
|测试集 | {len(test_df):,} | {test_df['renew_in_H'].mean():.4%} |

##3. 模型对比指标

{metric_df.to_markdown(index=False)}

##4. 状态增强随机森林特征重要性

{importance.head(20).to_markdown(index=False)}

##5. 模型采用建议

从测试集综合表现看，最优模型为 **{best['模型']}**，其测试 AUC为 {best['测试AUC']:.4f}，测试F1为 {best['测试F1']:.4f}，测试Brier Score为 {best['测试Brier']:.4f}。

若状态增强随机森林相对普通随机森林有提升或保持相近表现，则说明加入用户状态和行为响应特征不会削弱预测效果，同时增强了模型与积分运营场景之间的联系。本文后续可采用状态增强随机森林作为续费概率预测模型，用于积分策略模拟；Cox模型保留为生存分析和右删失处理说明，Logistic-GAM作为非线性解释辅助模型。

##6. 可放入论文的图片

- `5.3_图13_状态增强随机森林与对比模型指标_H60.png`
- `5.3_图14_状态增强随机森林与对比模型概率分布_H60.png`
- `5.3_图15_状态增强随机森林特征重要性Top15_H60.png`
- `serf_renewal_h60_model.joblib`
- `serf_renewal_h60_features.json`
"""
	REPORT_PATH.write_text(report, encoding="utf-8")


def main():
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	all_df = load_data()
	train_df, test_df, cutoff, active_target = split_binary(all_df)
	cox_train_df, cox_test_df = split_cox(all_df, cutoff, active_target)

	metrics = []
	pred_df = test_df[["user_id", "cycle_id", "end_date", "renew_in_H"]].copy()

	serf_metric, serf_prob, serf_model = fit_predict_classifier("状态增强随机森林", make_rf(STATE_RF_FEATURES), STATE_RF_FEATURES, train_df, test_df)
	metrics.append(serf_metric)
	pred_df["状态增强随机森林_prob"] = serf_prob

	plain_metric, plain_prob, _ = fit_predict_classifier("普通随机森林", make_rf(PLAIN_RF_FEATURES), PLAIN_RF_FEATURES, train_df, test_df)
	metrics.append(plain_metric)
	pred_df["普通随机森林_prob"] = plain_prob

	gam_metric, gam_prob, _ = fit_predict_classifier("Logistic-GAM", make_gam(), PLAIN_RF_FEATURES, train_df, test_df)
	metrics.append(gam_metric)
	pred_df["Logistic-GAM_prob"] = gam_prob

	cox_metric, cox_prob = fit_predict_cox(cox_train_df, cox_test_df)
	metrics.append(cox_metric)
	pred_df["Cox_prob"] = cox_prob

	metric_df = pd.DataFrame(metrics)
	importance = rf_importance(serf_model, STATE_RF_FEATURES)

	metric_df.to_csv(METRIC_PATH, index=False, encoding="utf-8-sig")
	importance.to_csv(IMPORTANCE_PATH, index=False, encoding="utf-8-sig")
	pred_df.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")
	export_serf_artifacts(serf_model, active_target)

	save_metric_plot(metric_df)
	save_probability_plot(pred_df)
	save_importance_plot(importance)
	write_report(metric_df, importance, train_df, test_df, cutoff)
	print(f"已生成状态增强随机森林对比结果：{REPORT_PATH}")
	print(f"已导出SERF模型：{SERF_MODEL_PATH}")
	print(f"已导出SERF特征元信息：{SERF_META_PATH}")


if __name__ == "__main__":
	main()

