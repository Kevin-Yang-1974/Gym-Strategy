# -*- coding: utf-8 -*-
"""增强版私教购买预测模型。

目标：
1. 增强私教购买模型的排序能力，重点关注 PR-AUC、TopK Precision/Recall/Lift；
2. 为后续策略模拟导出可复用的私教购买概率模型工件；
3. 避免重新使用旧版 lambda_pt 线性概率加成。
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
	average_precision_score,
	brier_score_loss,
	confusion_matrix,
	f1_score,
	precision_recall_curve,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "数据整理程序输出" / "整理后的数据"
OUT_DIR = BASE_DIR / "模型输出结果"
DATA_PATH = DATA_DIR / "pt_purchase_dataset_30d.csv"

REPORT_PATH = OUT_DIR / "5.4_私教购买增强模型对比结果.md"
METRIC_PATH = OUT_DIR / "5.4_私教购买增强模型对比指标.csv"
TOPK_PATH = OUT_DIR / "5.4_私教购买增强模型TopK指标.csv"
PRED_PATH = OUT_DIR / "5.4_私教购买增强模型测试集预测.csv"
IMPORTANCE_PATH = OUT_DIR / "5.4_私教购买增强模型特征重要性.csv"
CONFUSION_PATH = OUT_DIR / "5.4_私教购买增强模型混淆矩阵.csv"
MODEL_PATH = OUT_DIR / "pt_purchase_enhanced_model.joblib"
FEATURE_META_PATH = OUT_DIR / "pt_purchase_enhanced_features.json"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

BASE_FEATURES = [
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
	"active_member",
	"dte",
	"pt_bought_before",
	"pt_amount_total_before",
	"pt_sessions_bought_before",
	"pt_unit_price_est",
	"A_score",
	"S_score",
	"current_member_type",
	"user_state",
]

ENGINEERING_SOURCE_FEATURES = [
	"days_since_last_visit",
	"days_since_last_buy",
	"days_since_last_active",
	"days_since_last_interact",
	"days_since_last_pt_buy",
]

CATEGORICAL_FEATURES = ["current_member_type", "user_state"]
BASE_NUMERIC_FEATURES = [f for f in BASE_FEATURES if f not in CATEGORICAL_FEATURES]

ENGINEERED_FEATURES = [
	"visit_recency_score",
	"buy_recency_score",
	"active_recency_score",
	"interact_recency_score",
	"pt_recency_score",
	"pt_stickiness_score",
	"pt_session_stickiness_score",
	"active_recent_score",
	"visit_recent_intensity",
	"strength_frequency_score",
	"strength_volume_score",
	"new_pt_strength_potential",
	"o2o_strength_score",
	"pt_share_amount",
	"member_share_amount",
	"discount_intensity",
	"order_per_visit",
	"spend_per_visit",
	"near_expiry_30",
	"near_expiry_60",
	"expired_member_flag",
	"active_member_with_pt_history",
	"active_member_no_pt_history",
]

FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_FEATURES

CORE_FEATURES = [
	"visit_count",
	"stay_hours_total",
	"calorie_total",
	"anaerobic_volume",
	"anaerobic_count",
	"order_amount_total",
	"order_count",
	"order_amount_avg",
	"member_amount",
	"discount_order_ratio",
	"active_member",
	"dte",
	"pt_bought_before",
	"pt_amount_total_before",
	"pt_sessions_bought_before",
	"pt_unit_price_est",
	"A_score",
	"S_score",
	"current_member_type",
	"user_state",
	"visit_recency_score",
	"active_recency_score",
	"pt_recency_score",
	"pt_stickiness_score",
	"strength_frequency_score",
	"strength_volume_score",
	"new_pt_strength_potential",
	"near_expiry_30",
	"active_member_with_pt_history",
	"active_member_no_pt_history",
]

FEATURE_SETS = {
	"核心精简特征": CORE_FEATURES,
	"全量增强特征": FEATURES,
}

RENAME = {
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
	"current_member_type": "当前会员类型",
	"user_state": "用户状态",
	"visit_recency_score": "到店近因得分",
	"buy_recency_score": "购买近因得分",
	"active_recency_score": "活跃近因得分",
	"interact_recency_score": "互动近因得分",
	"pt_recency_score": "私教购买近因得分",
	"pt_stickiness_score": "私教消费粘性",
	"pt_session_stickiness_score": "私教课时粘性",
	"active_recent_score": "近期活跃强度",
	"visit_recent_intensity": "近期到店强度",
	"strength_frequency_score": "无氧频次得分",
	"strength_volume_score": "无氧训练强度",
	"new_pt_strength_potential": "新私教力量训练潜力",
	"o2o_strength_score": "线上线下联动强度",
	"pt_share_amount": "私教消费占比",
	"member_share_amount": "会员消费占比",
	"discount_intensity": "优惠消费强度",
	"order_per_visit": "到店购买频率",
	"spend_per_visit": "到店消费强度",
	"near_expiry_30": "会员30天内到期",
	"near_expiry_60": "会员60天内到期",
	"expired_member_flag": "会员已过期标记",
	"active_member_with_pt_history": "有效会员且有私教历史",
	"active_member_no_pt_history": "有效会员但无私教历史",
}

TOPK_LIST = [100, 200, 500, 1000, 2000, 5000]
NEGATIVE_SAMPLE_RATIO = 60
RANDOM_STATE = 2026


def log1p_array(x):
	return np.log1p(np.maximum(x, 0))


def safe_numeric(df, col, default=0):
	return pd.to_numeric(df[col], errors="coerce").fillna(default)


def add_engineered_features(df):
	out = df.copy()
	for col in list(dict.fromkeys(BASE_NUMERIC_FEATURES + ENGINEERING_SOURCE_FEATURES)):
		out[col] = pd.to_numeric(out[col], errors="coerce")
	for col in CATEGORICAL_FEATURES:
		out[col] = out[col].astype("object").where(out[col].notna(), "缺失")

	visit_days = safe_numeric(out, "days_since_last_visit", 999).clip(lower=0)
	buy_days = safe_numeric(out, "days_since_last_buy", 999).clip(lower=0)
	active_days = safe_numeric(out, "days_since_last_active", 999).clip(lower=0)
	interact_days = safe_numeric(out, "days_since_last_interact", 999).clip(lower=0)
	pt_days = safe_numeric(out, "days_since_last_pt_buy", 999).clip(lower=0)
	active_member = safe_numeric(out, "active_member", 0).clip(0, 1)
	pt_bought = safe_numeric(out, "pt_bought_before", 0).clip(0, 1)
	a_score = safe_numeric(out, "A_score", 0).clip(lower=0)
	s_score = safe_numeric(out, "S_score", 0).clip(lower=0)
	visit_count = safe_numeric(out, "visit_count", 0).clip(lower=0)
	order_count = safe_numeric(out, "order_count", 0).clip(lower=0)
	order_amount = safe_numeric(out, "order_amount_total", 0).clip(lower=0)
	member_amount = safe_numeric(out, "member_amount", 0).clip(lower=0)
	pt_amount = safe_numeric(out, "pt_amount", 0).clip(lower=0)
	pt_total = safe_numeric(out, "pt_amount_total_before", 0).clip(lower=0)
	pt_sessions_total = safe_numeric(out, "pt_sessions_bought_before", 0).clip(lower=0)
	anaerobic_count = safe_numeric(out, "anaerobic_count", 0).clip(lower=0)
	anaerobic_volume = safe_numeric(out, "anaerobic_volume", 0).clip(lower=0)
	discount_ratio = safe_numeric(out, "discount_order_ratio", 0).clip(0, 1)
	dte = safe_numeric(out, "dte", -999)

	out["visit_recency_score"] = 1 / (visit_days + 1)
	out["buy_recency_score"] = 1 / (buy_days + 1)
	out["active_recency_score"] = 1 / (active_days + 1)
	out["interact_recency_score"] = 1 / (interact_days + 1)
	out["pt_recency_score"] = 1 / (pt_days + 1)
	out["pt_stickiness_score"] = np.log1p(pt_total) * out["pt_recency_score"]
	out["pt_session_stickiness_score"] = np.log1p(pt_sessions_total) * out["pt_recency_score"]
	out["active_recent_score"] = a_score * out["active_recency_score"]
	out["visit_recent_intensity"] = np.log1p(visit_count) * out["visit_recency_score"]
	out["strength_frequency_score"] = np.log1p(anaerobic_count)
	out["strength_volume_score"] = np.log1p(anaerobic_volume)
	out["new_pt_strength_potential"] = (1 - pt_bought) * np.log1p(anaerobic_count + anaerobic_volume)
	out["o2o_strength_score"] = a_score * np.log1p(s_score)
	out["pt_share_amount"] = pt_amount / (order_amount + 1)
	out["member_share_amount"] = member_amount / (order_amount + 1)
	out["discount_intensity"] = discount_ratio * np.log1p(order_amount)
	out["order_per_visit"] = order_count / (visit_count + 1)
	out["spend_per_visit"] = order_amount / (visit_count + 1)
	out["near_expiry_30"] = ((active_member == 1) & (dte >= 0) & (dte <= 30)).astype(int)
	out["near_expiry_60"] = ((active_member == 1) & (dte >= 0) & (dte <= 60)).astype(int)
	out["expired_member_flag"] = ((active_member == 0) | (dte < 0)).astype(int)
	out["active_member_with_pt_history"] = ((active_member == 1) & (pt_bought == 1)).astype(int)
	out["active_member_no_pt_history"] = ((active_member == 1) & (pt_bought == 0)).astype(int)
	return out


def load_data():
	df = pd.read_csv(DATA_PATH, low_memory=False)
	df["feature_window_end"] = pd.to_datetime(df["feature_window_end"], errors="coerce")
	df = df.dropna(subset=["feature_window_end"])
	df["y_pt"] = pd.to_numeric(df["y_pt"], errors="coerce").fillna(0).astype(int)
	return add_engineered_features(df)


def split_data(df):
	cutoff = df["feature_window_end"].quantile(0.8)
	train_df = df[df["feature_window_end"] <= cutoff].copy()
	test_df = df[df["feature_window_end"] > cutoff].copy()
	return train_df, test_df, cutoff


def numeric_pipe():
	return Pipeline([
		("imputer", SimpleImputer(strategy="median")),
		("log", FunctionTransformer(log1p_array, feature_names_out="one-to-one")),
	])


def categorical_pipe(sparse_output=True):
	return Pipeline([
		("imputer", SimpleImputer(strategy="most_frequent")),
		("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=sparse_output)),
	])


def split_feature_types(feature_list):
	categorical = [f for f in CATEGORICAL_FEATURES if f in feature_list]
	numeric = [f for f in feature_list if f not in categorical]
	return numeric, categorical


def make_preprocess(feature_list, sparse_categorical=True):
	numeric, categorical = split_feature_types(feature_list)
	transformers = []
	if numeric:
		transformers.append(("num", numeric_pipe(), numeric))
	if categorical:
		transformers.append(("cat", categorical_pipe(sparse_categorical), categorical))
	return ColumnTransformer(transformers, sparse_threshold=0.3 if sparse_categorical else 0.0)


def make_rf(feature_list, n_estimators=220, max_depth=14, min_samples_leaf=40):
	preprocess = ColumnTransformer([
		("num", numeric_pipe(), split_feature_types(feature_list)[0]),
		("cat", categorical_pipe(True), split_feature_types(feature_list)[1]),
	])
	model = RandomForestClassifier(
		n_estimators=n_estimators,
		max_depth=max_depth,
		min_samples_leaf=min_samples_leaf,
		class_weight="balanced_subsample",
		n_jobs=-1,
		random_state=RANDOM_STATE,
	)
	return Pipeline([("preprocess", preprocess), ("model", model)])


def make_baseline_rf(feature_list):
	return make_rf(feature_list, n_estimators=220, max_depth=14, min_samples_leaf=40)


def make_enhanced_rf(feature_list):
	return make_rf(feature_list, n_estimators=240, max_depth=16, min_samples_leaf=25)


def make_extra_trees(feature_list):
	preprocess = make_preprocess(feature_list, sparse_categorical=True)
	model = ExtraTreesClassifier(
		n_estimators=320,
		max_depth=18,
		min_samples_leaf=10,
		class_weight="balanced",
		n_jobs=-1,
		random_state=RANDOM_STATE,
	)
	return Pipeline([("preprocess", preprocess), ("model", model)])


def make_hist_gbdt(feature_list):
	preprocess = make_preprocess(feature_list, sparse_categorical=False)
	model = HistGradientBoostingClassifier(
		loss="log_loss",
		learning_rate=0.05,
		max_iter=220,
		max_leaf_nodes=31,
		min_samples_leaf=20,
		l2_regularization=0.05,
		class_weight=None,
		early_stopping=True,
		random_state=RANDOM_STATE,
	)
	return Pipeline([("preprocess", preprocess), ("model", model)])


def case_control_sample(train_df, ratio=NEGATIVE_SAMPLE_RATIO):
	pos = train_df[train_df["y_pt"] == 1]
	neg = train_df[train_df["y_pt"] == 0]
	n_neg = min(len(neg), len(pos) * ratio)
	neg_sample = neg.sample(n=n_neg, random_state=RANDOM_STATE)
	return pd.concat([pos, neg_sample], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def prior_correct(prob, true_prior, sample_prior):
	prob = np.clip(prob, 1e-8, 1 - 1e-8)
	true_prior = np.clip(true_prior, 1e-8, 1 - 1e-8)
	sample_prior = np.clip(sample_prior, 1e-8, 1 - 1e-8)
	odds = prob / (1 - prob)
	factor = (true_prior / (1 - true_prior)) / (sample_prior / (1 - sample_prior))
	corrected_odds = odds * factor
	return corrected_odds / (1 + corrected_odds)


def threshold_by_f1(y_true, prob):
	precisions, recalls, thresholds = precision_recall_curve(y_true, prob)
	f1_values = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
	if len(thresholds) == 0:
		return 0.5
	best_idx = int(np.nanargmax(f1_values[:-1]))
	return float(thresholds[best_idx])


def topk_metrics(y_true, prob, model_name, feature_set):
	y_true = np.asarray(y_true, dtype=int)
	prob = np.asarray(prob, dtype=float)
	order = np.argsort(-prob)
	base_rate = y_true.mean()
	rows = []
	for k in TOPK_LIST:
		k_eff = min(k, len(y_true))
		idx = order[:k_eff]
		tp = int(y_true[idx].sum())
		precision = tp / k_eff
		recall = tp / max(int(y_true.sum()), 1)
		lift = precision / max(base_rate, 1e-12)
		rows.append({
			"特征集": feature_set,
			"模型": model_name,
			"TopK": int(k_eff),
			"命中正样本数": tp,
			"Precision@K": precision,
			"Recall@K": recall,
			"Lift@K": lift,
		})
	return rows


def evaluate_model(name, model, feature_set, feature_list, train_df, test_df, fit_df=None, prior_correction=False):
	if fit_df is None:
		fit_df = train_df
	y_fit = fit_df["y_pt"].to_numpy(dtype=int)
	y_train = train_df["y_pt"].to_numpy(dtype=int)
	y_test = test_df["y_pt"].to_numpy(dtype=int)

	model.fit(fit_df[feature_list], y_fit)
	fit_prob = model.predict_proba(fit_df[feature_list])[:, 1]
	train_prob = model.predict_proba(train_df[feature_list])[:, 1]
	test_prob = model.predict_proba(test_df[feature_list])[:, 1]
	if prior_correction:
		train_prior = float(train_df["y_pt"].mean())
		fit_prior = float(fit_df["y_pt"].mean())
		fit_prob = prior_correct(fit_prob, train_prior, fit_prior)
		train_prob = prior_correct(train_prob, train_prior, fit_prior)
		test_prob = prior_correct(test_prob, train_prior, fit_prior)

	threshold = threshold_by_f1(y_train, train_prob)
	test_pred = (test_prob >= threshold).astype(int)
	tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
	train_pr_auc = float(average_precision_score(y_train, train_prob))
	test_pr_auc = float(average_precision_score(y_test, test_prob))
	metric = {
		"特征集": feature_set,
		"模型": name,
		"训练方式": "case_control" if fit_df is not train_df else "full",
		"特征数": int(len(feature_list)),
		"训练样本数": int(len(fit_df)),
		"训练正样本数": int(y_fit.sum()),
		"训练样本正样本比例": float(y_fit.mean()),
		"真实训练正样本比例": float(y_train.mean()),
		"测试正样本数": int(y_test.sum()),
		"阈值": float(threshold),
		"Fit_PR_AUC": float(average_precision_score(y_fit, fit_prob)),
		"Train_PR_AUC": train_pr_auc,
		"PR_AUC": test_pr_auc,
		"PR_AUC泛化差": float(train_pr_auc - test_pr_auc),
		"ROC_AUC": float(roc_auc_score(y_test, test_prob)),
		"Brier": float(brier_score_loss(y_test, test_prob)),
		"F1": float(f1_score(y_test, test_pred, zero_division=0)),
		"Recall": float(recall_score(y_test, test_pred, zero_division=0)),
		"Precision": float(precision_score(y_test, test_pred, zero_division=0)),
		"TP": int(tp),
		"FP": int(fp),
		"TN": int(tn),
		"FN": int(fn),
		"预测正样本数": int(test_pred.sum()),
		"是否先验校正": bool(prior_correction),
	}
	pred_part = pd.DataFrame({
		"user_id": test_df["user_id"].to_numpy(),
		"feature_period": test_df["feature_period"].to_numpy(),
		"label_period": test_df["label_period"].to_numpy(),
		"y_pt": y_test,
		f"{feature_set}_{name}_prob": test_prob,
		f"{feature_set}_{name}_pred": test_pred,
	})
	confusion = pd.DataFrame([
		{"特征集": feature_set, "模型": name, "真实": 0, "预测": 0, "样本数": int(tn)},
		{"特征集": feature_set, "模型": name, "真实": 0, "预测": 1, "样本数": int(fp)},
		{"特征集": feature_set, "模型": name, "真实": 1, "预测": 0, "样本数": int(fn)},
		{"特征集": feature_set, "模型": name, "真实": 1, "预测": 1, "样本数": int(tp)},
	])
	return metric, pred_part, confusion, model, topk_metrics(y_test, test_prob, name, feature_set)


def aggregate_feature_importance(model, name, feature_set, feature_list):
	if not hasattr(model.named_steps["model"], "feature_importances_"):
		return pd.DataFrame()
	pre = model.named_steps["preprocess"]
	est = model.named_steps["model"]
	feature_names = pre.get_feature_names_out()
	rows = []
	for feature_name, value in zip(feature_names, est.feature_importances_):
		clean = feature_name.split("__", 1)[-1]
		matched = None
		for feature in FEATURES:
			if clean == feature or clean.startswith(feature + "_"):
				matched = feature
				break
		if matched is None:
			matched = clean
		rows.append({
			"特征集": feature_set,
			"模型": name,
			"原始变量": matched,
			"中文含义": RENAME.get(matched, matched),
			"重要性": float(value),
		})
	out = pd.DataFrame(rows).groupby(["特征集", "模型", "原始变量", "中文含义"], as_index=False)["重要性"].sum()
	return out.sort_values(["特征集", "模型", "重要性"], ascending=[True, True, False])


def save_metric_plot(metric_df):
	plot_df = metric_df.sort_values("PR_AUC", ascending=False).reset_index(drop=True)
	x = np.arange(len(plot_df))
	width = 0.18
	fig, ax = plt.subplots(figsize=(11, 5))
	for idx, col in enumerate(["PR_AUC", "ROC_AUC", "F1", "Recall"]):
		ax.bar(x + (idx - 1.5) * width, plot_df[col], width, label=col)
	ax.set_xticks(x)
	ax.set_xticklabels(plot_df["特征集"] + "\n" + plot_df["模型"], rotation=20, ha="right")
	ax.set_ylim(0, 1)
	ax.set_title("增强私教购买模型测试集指标对比")
	ax.legend()
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.4_图4_增强私教购买模型指标对比.png", bbox_inches="tight")
	plt.close(fig)


def save_pr_curve(pred_df, metric_df):
	fig, ax = plt.subplots(figsize=(7, 5))
	y = pred_df["y_pt"].to_numpy(dtype=int)
	for _, row in metric_df.sort_values("PR_AUC", ascending=False).iterrows():
		name = row["模型"]
		feature_set = row["特征集"]
		prob_col = f"{feature_set}_{name}_prob"
		prob = pred_df[prob_col].to_numpy()
		precision, recall, _ = precision_recall_curve(y, prob)
		ap = average_precision_score(y, prob)
		ax.plot(recall, precision, label=f"{feature_set}-{name} PR-AUC={ap:.4f}")
	ax.set_xlabel("Recall")
	ax.set_ylabel("Precision")
	ax.set_title("增强私教购买模型PR曲线")
	ax.legend()
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.4_图5_增强私教购买模型PR曲线.png", bbox_inches="tight")
	plt.close(fig)


def save_topk_plot(topk_df):
	plot_df = topk_df[topk_df["TopK"].isin([500, 1000, 2000, 5000])].copy()
	fig, ax = plt.subplots(figsize=(9, 5))
	for (feature_set, name), group in plot_df.groupby(["特征集", "模型"]):
		group = group.sort_values("TopK")
		ax.plot(group["TopK"], group["Lift@K"], marker="o", label=f"{feature_set}-{name}")
	ax.set_xlabel("TopK")
	ax.set_ylabel("Lift@K")
	ax.set_title("增强私教购买模型TopK提升倍数")
	ax.legend()
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.4_图6_增强私教购买模型TopK_Lift.png", bbox_inches="tight")
	plt.close(fig)


def save_overfit_plot(metric_df):
	plot_df = metric_df.sort_values("PR_AUC泛化差", ascending=False).reset_index(drop=True)
	fig, ax = plt.subplots(figsize=(10, 5))
	labels = plot_df["特征集"] + "\n" + plot_df["模型"]
	ax.bar(labels, plot_df["PR_AUC泛化差"], color="#b85c5c")
	ax.axhline(0, color="#444444", linewidth=0.8)
	ax.set_ylabel("Train PR-AUC - Test PR-AUC")
	ax.set_title("私教购买模型过拟合风险对照")
	ax.tick_params(axis="x", rotation=20)
	fig.tight_layout()
	fig.savefig(OUT_DIR / "5.4_图7_增强私教购买模型过拟合风险.png", bbox_inches="tight")
	plt.close(fig)


def write_report(metric_df, topk_df, importance_df, confusion_df, train_df, test_df, cutoff, best_key):
	best_feature_set, best_name = best_key
	best = metric_df[(metric_df["特征集"] == best_feature_set) & (metric_df["模型"] == best_name)].iloc[0]
	feature_compare = metric_df[[
		"特征集",
		"模型",
		"特征数",
		"训练方式",
		"Train_PR_AUC",
		"PR_AUC",
		"PR_AUC泛化差",
		"ROC_AUC",
		"Brier",
		"F1",
		"Recall",
		"Precision",
	]].copy()
	topk_best = topk_df[(topk_df["特征集"] == best_feature_set) & (topk_df["模型"] == best_name)].copy()
	report = f"""#5.4 增强版私教购买模型对比结果

本文件由 `模型输出结果/compare_pt_purchase_models_enhanced_30d.py` 自动生成，用于提升私教购买预测模型的排序能力，并为后续积分策略模拟准备可在线推理的私教购买概率工件。

##1. 增强思路

原始私教购买模型面对极度稀有事件，测试集正样本比例仅约 {test_df['y_pt'].mean():.4%}。增强版模型主要做三项改进：

1. 增加近因、粘性、无氧训练潜力、会员临期和消费占比等业务特征；
2. 增加“核心精简特征”和“全量增强特征”的消融对照，检验特征过多是否导致过拟合；
3. 引入适合稀有事件排序的 case-control 训练方式，并对概率做先验比例校正；
4. 除 PR-AUC、ROC-AUC、F1 外，补充 TopK Precision、Recall、Lift 和 Brier Score，用于衡量模型识别高潜私教购买用户及输出概率的能力。

样本仍按 `feature_window_end` 做时间切分，切分日期为 `{cutoff.date()}`。

| 数据集 | 样本量 | 正样本数 | 正样本比例 |
|---|---:|---:|---:|
| 训练集 | {len(train_df):,} | {int(train_df['y_pt'].sum()):,} | {train_df['y_pt'].mean():.4%} |
| 测试集 | {len(test_df):,} | {int(test_df['y_pt'].sum()):,} | {test_df['y_pt'].mean():.4%} |

##2. 特征集规模与过拟合风险

本节用 `Train_PR_AUC - Test_PR_AUC` 作为过拟合风险的直接观察量。若全量增强特征的测试 PR-AUC 没有超过核心精简特征，且泛化差明显更大，则说明“输入特征太多”确实可能损害泛化能力。

{feature_compare.to_markdown(index=False)}

##3. 模型评价指标

{metric_df.to_markdown(index=False)}

##4. TopK识别效果

{topk_df.to_markdown(index=False)}

当前最佳模型在 TopK 场景下的命中效果如下：

{topk_best.to_markdown(index=False)}

##5. 混淆矩阵

{confusion_df.to_markdown(index=False)}

##6. 特征重要性

{importance_df.head(30).to_markdown(index=False) if not importance_df.empty else '当前最佳模型不提供树模型特征重要性。'}

##7. 当前最佳模型

按测试集 PR-AUC 优先选择，当前增强版最佳模型为 **{best_feature_set}-{best_name}**，测试集 PR-AUC 为 {best['PR_AUC']:.4f}，ROC-AUC 为 {best['ROC_AUC']:.4f}，Brier Score 为 {best['Brier']:.6f}，F1 为 {best['F1']:.4f}，Recall 为 {best['Recall']:.4f}，Precision 为 {best['Precision']:.4f}。

该模型已导出为：

- `pt_purchase_enhanced_model.joblib`
- `pt_purchase_enhanced_features.json`

后续策略模拟中，可以用该模型替代当前固定的私教购买基线概率，并在积分策略改变活跃、互动和无氧训练相关特征后，重新推理策略后的私教购买概率。这样 TOPSIS 中“私教购买概率提升”将不再是全0指标。

需要注意的是，私教购买正样本极少，模型结果应主要用于高潜用户排序和情景模拟中的概率输入，不能解释为积分对私教购买的真实因果效应。

##8. 可放入论文的图片

- `5.4_图4_增强私教购买模型指标对比.png`
- `5.4_图5_增强私教购买模型PR曲线.png`
- `5.4_图6_增强私教购买模型TopK_Lift.png`
- `5.4_图7_增强私教购买模型过拟合风险.png`
"""
	REPORT_PATH.write_text(report, encoding="utf-8")


def save_artifacts(best_model, best_key, metric_df, fit_df, train_df, feature_list):
	best_feature_set, best_name = best_key
	meta = {
		"model_name": best_name,
		"feature_set": best_feature_set,
		"feature_order": feature_list,
		"base_features": BASE_FEATURES,
		"engineering_source_features": ENGINEERING_SOURCE_FEATURES,
		"engineered_features": ENGINEERED_FEATURES,
		"core_features": CORE_FEATURES,
		"numeric_features": split_feature_types(feature_list)[0],
		"categorical_features": CATEGORICAL_FEATURES,
		"target": "y_pt",
		"prediction_window_days": 30,
		"requires_engineering": True,
		"engineering_function": "add_engineered_features in compare_pt_purchase_models_enhanced_30d.py",
		"true_train_prior": float(train_df["y_pt"].mean()),
		"fit_sample_prior": float(fit_df["y_pt"].mean()),
		"prior_correction": bool(metric_df.loc[(metric_df["特征集"] == best_feature_set) & (metric_df["模型"] == best_name), "是否先验校正"].iloc[0]),
		"metrics": metric_df.loc[(metric_df["特征集"] == best_feature_set) & (metric_df["模型"] == best_name)].iloc[0].to_dict(),
	}
	joblib.dump(best_model, MODEL_PATH)
	FEATURE_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	df = load_data()
	train_df, test_df, cutoff = split_data(df)
	cc_train = case_control_sample(train_df)

	model_specs = []
	for feature_set, feature_list in FEATURE_SETS.items():
		model_specs.extend([
			(feature_set, feature_list, "增强随机森林", make_enhanced_rf(feature_list), train_df, False),
			(feature_set, feature_list, "CaseControl-ExtraTrees", make_extra_trees(feature_list), cc_train, True),
			(feature_set, feature_list, "CaseControl-HistGBDT", make_hist_gbdt(feature_list), cc_train, True),
		])

	metrics = []
	pred_df = None
	confusions = []
	topk_rows = []
	importance_parts = []
	fitted_models = {}
	fit_frames = {}
	feature_lists = {}

	for feature_set, feature_list, name, model, fit_df, prior_correction in model_specs:
		model_key = (feature_set, name)
		print(f"训练模型：{feature_set}-{name}，特征数={len(feature_list)}，训练样本数={len(fit_df):,}，正样本数={int(fit_df['y_pt'].sum()):,}")
		metric, part_pred, confusion, fitted, topk = evaluate_model(
			name,
			model,
			feature_set,
			feature_list,
			train_df,
			test_df,
			fit_df=fit_df,
			prior_correction=prior_correction,
		)
		metrics.append(metric)
		confusions.append(confusion)
		topk_rows.extend(topk)
		fitted_models[model_key] = fitted
		fit_frames[model_key] = fit_df
		feature_lists[model_key] = feature_list
		imp = aggregate_feature_importance(fitted, name, feature_set, feature_list)
		if not imp.empty:
			importance_parts.append(imp)
		if pred_df is None:
			pred_df = part_pred
		else:
			cols = [c for c in part_pred.columns if c.endswith("_prob") or c.endswith("_pred")]
			pred_df = pd.concat([pred_df, part_pred[cols]], axis=1)

	metric_df = pd.DataFrame(metrics).sort_values(["PR_AUC", "F1"], ascending=[False, False]).reset_index(drop=True)
	confusion_df = pd.concat(confusions, ignore_index=True)
	topk_df = pd.DataFrame(topk_rows)
	importance_df = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()

	best_key = (metric_df.iloc[0]["特征集"], metric_df.iloc[0]["模型"])
	best_model = fitted_models[best_key]
	best_fit_df = fit_frames[best_key]
	best_feature_list = feature_lists[best_key]

	metric_df.to_csv(METRIC_PATH, index=False, encoding="utf-8-sig")
	pred_df.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")
	topk_df.to_csv(TOPK_PATH, index=False, encoding="utf-8-sig")
	confusion_df.to_csv(CONFUSION_PATH, index=False, encoding="utf-8-sig")
	if not importance_df.empty:
		importance_df.to_csv(IMPORTANCE_PATH, index=False, encoding="utf-8-sig")

	save_metric_plot(metric_df)
	save_pr_curve(pred_df, metric_df)
	save_topk_plot(topk_df)
	save_overfit_plot(metric_df)
	write_report(metric_df, topk_df, importance_df, confusion_df, train_df, test_df, cutoff, best_key)
	save_artifacts(best_model, best_key, metric_df, best_fit_df, train_df, best_feature_list)
	print(f"已生成增强版私教购买模型结果：{REPORT_PATH}")
	print(f"已导出增强版私教购买模型：{MODEL_PATH}")
	print(f"已导出增强版私教购买模型元信息：{FEATURE_META_PATH}")


if __name__ == "__main__":
	main()
