# -*- coding: utf-8 -*-
"""候选积分策略模拟、经营约束筛选、熵权-TOPSIS排序与多延期观察窗口分析。"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "数据整理程序输出" / "整理后的数据"
OUT_DIR = BASE_DIR / "模型输出结果"

RENEW_DATA_PATH = DATA_DIR / "renewal_survival_dataset_H30.csv"
PT_DATA_PATH = DATA_DIR / "pt_purchase_dataset_30d.csv"
SERF_MODEL_PATH = OUT_DIR / "serf_renewal_h30_model.joblib"
SERF_META_PATH = OUT_DIR / "serf_renewal_h30_features.json"
PT_MODEL_PATH = OUT_DIR / "pt_purchase_enhanced_model.joblib"
PT_META_PATH = OUT_DIR / "pt_purchase_enhanced_features.json"

STRATEGY_DETAIL_PATH = OUT_DIR / "5.5_候选积分策略模拟明细.csv"
STRATEGY_SUMMARY_PATH = OUT_DIR / "5.5_候选积分策略汇总.csv"
TOPSIS_PATH = OUT_DIR / "5.6_熵权TOPSIS策略排序.csv"
REPORT_PATH = OUT_DIR / "5.5_5.6_积分策略模拟与TOPSIS结果.md"

MULTI_WINDOW_SUMMARY_PATH = OUT_DIR / "5.6_多延迟观察窗口方案汇总.csv"
MULTI_WINDOW_TRAJECTORY_PATH = OUT_DIR / "5.6_多延迟观察窗口轨迹.csv"
S5_VS_S0_PATH = OUT_DIR / "5.6_S5_vs_S0利润差.csv"
S5_VS_S0_CROSSOVER_PATH = OUT_DIR / "5.6_S5_vs_S0交叉点表.csv"
S3_S4_S5_COMPARISON_PATH = OUT_DIR / "5.6_S3_S4_S5私教模型接入后对比.csv"
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

LONG_HORIZON_CARRY_SCENARIOS = [
	{"观察期标签": "L1", "carry_a":0.20},
	{"观察期标签": "L2", "carry_a":0.35},
	{"观察期标签": "L3", "carry_a":0.50},
	{"观察期标签": "L4", "carry_a":0.70},
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
	"Z1_高活跃高消费用户": {"rho":0.25, "q":0.18, "pref_p0":10, "pref_p1":4},
	"Z6_私教潜在用户": {"rho":0.45, "q":0.28, "pref_p0":10, "pref_p1":4},
	"Z7_流失风险用户": {"rho":0.35, "q":0.22, "pref_p0":12, "pref_p1":5},
	"其他": {"rho":0.30, "q":0.20, "pref_p0":10, "pref_p1":4},
}

SEMANTIC_GAMMA =0.5
FEATURE_UPDATE_EPS =1e-9
SERF_META_REQUIRED_KEYS = ["feature_order", "base_features", "response_features", "categorical_features", "active_target"]
SERF_RESPONSE_FEATURES = [
	"active_gap",
	"o2o_strength",
	"pt_potential",
	"buy_stickiness",
	"visit_recency_score",
	"active_recency_score",
	"interact_recency_score",
]
SERF_CATEGORICAL_MISSING_TOKEN = "缺失"
OFFLINE_ACTIVITY_FEATURES = [
	"visit_count",
	"stay_hours_total",
	"run_distance_total",
	"calorie_total",
	"anaerobic_volume",
]
ONLINE_ACTIVITY_FEATURES = [
	"post_count",
	"like_given_count",
	"comment_given_count",
	"like_received_count",
	"comment_received_count",
]
SERF_L_RESPONSE_FEATURES = [
	"buy_stickiness",
	"visit_recency_score",
	"active_recency_score",
	"interact_recency_score",
]
SERF_RECENCY_SCORE_FEATURES = [
	"visit_recency_score",
	"active_recency_score",
	"interact_recency_score",
]
PT_META_REQUIRED_KEYS = [
	"feature_order",
	"base_features",
	"categorical_features",
	"numeric_features",
	"requires_engineering",
	"prior_correction",
]
PT_CATEGORICAL_MISSING_TOKEN = "缺失"


def load_serf_artifacts():
	meta = json.loads(SERF_META_PATH.read_text(encoding="utf-8"))
	for key in SERF_META_REQUIRED_KEYS:
		if key not in meta:
			raise KeyError(f"SERF 元信息缺少关键字段: {key}")
	if list(meta["response_features"]) != SERF_RESPONSE_FEATURES:
		raise ValueError("SERF 响应特征列表与训练导出元信息不一致，已停止打分以避免 train/serve skew。")
	model = joblib.load(SERF_MODEL_PATH)
	enable_parallel_prediction(model)
	return model, meta


def load_pt_artifacts():
	meta = json.loads(PT_META_PATH.read_text(encoding="utf-8"))
	for key in PT_META_REQUIRED_KEYS:
		if key not in meta:
			raise KeyError(f"私教购买模型元信息缺少关键字段: {key}")
	if not meta.get("requires_engineering", False):
		raise ValueError("私教购买模型元信息显示无需工程特征，但当前策略模拟按增强特征口径打分，已停止。")
	model = joblib.load(PT_MODEL_PATH)
	enable_parallel_prediction(model)
	return model, meta


def enable_parallel_prediction(model):
	if not hasattr(model, "get_params") or not hasattr(model, "set_params"):
		return
	params = model.get_params()
	parallel_params = {
		name: -1
		for name in params
		if name == "n_jobs" or name.endswith("__n_jobs")
	}
	if parallel_params:
		model.set_params(**parallel_params)


def log1p_array(x):
	return np.log1p(np.maximum(x,0))


def add_serf_response_features(df, active_target):
	# 必须与 compare_state_enhanced_rf_h30.py 中的 add_response_features 保持一致。
	out = df.copy()
	out["active_gap"] = np.maximum(active_target - out["A_score"].fillna(0),0)
	out["o2o_strength"] = out["A_score"].fillna(0) * np.log1p(np.maximum(out["S_score"].fillna(0),0))
	out["pt_potential"] = np.log1p(np.maximum(out["anaerobic_count"].fillna(0),0)) * (1 - out["pt_bought_before"].fillna(0).clip(0,1))
	out["buy_stickiness"] = np.log1p(np.maximum(out["pt_amount_total_before"].fillna(0),0)) / (out["days_since_last_buy"].fillna(999) +1)
	out["visit_recency_score"] =1 / (out["days_since_last_visit"].fillna(999) +1)
	out["active_recency_score"] =1 / (out["days_since_last_active"].fillna(999) +1)
	out["interact_recency_score"] =1 / (out["days_since_last_interact"].fillna(999) +1)
	return out


def safe_numeric(df, col, default=0):
	if col not in df.columns:
		return pd.Series(default, index=df.index)
	return pd.to_numeric(df[col], errors="coerce").fillna(default)


def add_pt_engineered_features(df, meta):
	# 必须与 compare_pt_purchase_models_enhanced_30d.py 中的 add_engineered_features 保持一致。
	out = df.copy()
	for col in meta["numeric_features"]:
		if col in out.columns:
			out[col] = pd.to_numeric(out[col], errors="coerce")
	for col in meta["categorical_features"]:
		if col in out.columns:
			out[col] = out[col].astype("object").where(out[col].notna(), PT_CATEGORICAL_MISSING_TOKEN)

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


def prepare_pt_scoring_frame(df, meta):
	required = list(dict.fromkeys(meta["base_features"] + meta.get("engineering_source_features", []) + meta["categorical_features"]))
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise KeyError(f"私教购买模型推理缺少字段: {missing}")
	frame = add_pt_engineered_features(df, meta)
	for col in meta["categorical_features"]:
		frame[col] = frame[col].astype("object").where(frame[col].notna(), PT_CATEGORICAL_MISSING_TOKEN)
	missing_after_engineering = [col for col in meta["feature_order"] if col not in frame.columns]
	if missing_after_engineering:
		raise KeyError(f"私教购买模型特征构造后仍缺少字段: {missing_after_engineering}")
	return frame[meta["feature_order"]]


def prior_correct(prob, true_prior, sample_prior):
	prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-8, 1 - 1e-8)
	true_prior = np.clip(float(true_prior), 1e-8, 1 - 1e-8)
	sample_prior = np.clip(float(sample_prior), 1e-8, 1 - 1e-8)
	odds = prob / (1 - prob)
	factor = (true_prior / (1 - true_prior)) / (sample_prior / (1 - sample_prior))
	return (factor * odds) / (1 + factor * odds)


def score_pt_probability(df, model, meta):
	x = prepare_pt_scoring_frame(df, meta)
	prob = model.predict_proba(x)[:,1]
	if meta.get("prior_correction", False):
		prob = prior_correct(prob, meta["true_train_prior"], meta["fit_sample_prior"])
	return pd.Series(prob, index=df.index, name="p_pt").clip(0,1)


def build_pt_fast_context(pop, model, meta):
	scoring_source = pop.copy()
	if "pt_user_state" in scoring_source.columns:
		scoring_source["user_state"] = scoring_source["pt_user_state"]
	base_frame = prepare_pt_scoring_frame(scoring_source, meta)
	preprocess = model.named_steps["preprocess"]
	num_pipeline = preprocess.named_transformers_["num"]
	cat_pipeline = preprocess.named_transformers_["cat"]
	num_cols = list(preprocess.transformers_[0][2])
	cat_cols = list(preprocess.transformers_[1][2])
	base_numeric = base_frame[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
	source_cols = list(meta.get("engineering_source_features", []))
	source_numeric = {
		col: pd.to_numeric(scoring_source[col], errors="coerce").to_numpy(dtype=np.float64)
		for col in source_cols
		if col in scoring_source.columns
	}
	cat_encoded = cat_pipeline.transform(base_frame[cat_cols])
	if hasattr(cat_encoded, "toarray"):
		cat_encoded = cat_encoded.toarray()
	return {
		"model": model.named_steps["model"],
		"num_cols": num_cols,
		"num_index": {col: idx for idx, col in enumerate(num_cols)},
		"base_numeric": base_numeric,
		"num_medians": num_pipeline.named_steps["imputer"].statistics_.astype(np.float64),
		"cat_encoded": np.asarray(cat_encoded, dtype=np.float64),
		"source_numeric": source_numeric,
		"prior_correction": bool(meta.get("prior_correction", False)),
		"true_train_prior": float(meta.get("true_train_prior", 0.0)),
		"fit_sample_prior": float(meta.get("fit_sample_prior", 0.0)),
	}


def active_response_scales(pop, delta_a, delta_s):
	a_old = pd.to_numeric(pop["A_score"], errors="coerce").fillna(0).clip(lower=0)
	s_old = pd.to_numeric(pop["S_score"], errors="coerce").fillna(0).clip(lower=0)
	delta_a = pd.Series(delta_a, index=pop.index).fillna(0).clip(lower=0)
	delta_s = pd.Series(delta_s, index=pop.index).fillna(0).clip(lower=0)
	offline_scale = 1 + SEMANTIC_GAMMA * np.log1p(delta_a / (a_old + FEATURE_UPDATE_EPS))
	online_scale = 1 + SEMANTIC_GAMMA * np.log1p(delta_s / (s_old + FEATURE_UPDATE_EPS))
	return offline_scale, online_scale


def fast_score_pt_counterfactual_probability(pop, delta_a, delta_s, context):
	delta_a_series = pd.Series(delta_a, index=pop.index).fillna(0).clip(lower=0)
	delta_s_series = pd.Series(delta_s, index=pop.index).fillna(0).clip(lower=0)
	if float(delta_a_series.sum() + delta_s_series.sum()) <= EPS:
		return pop["p_pt_base"]
	idx = context["num_index"]
	raw = context["base_numeric"].copy()
	delta_a_arr = delta_a_series.to_numpy(dtype=np.float64)
	delta_s_arr = delta_s_series.to_numpy(dtype=np.float64)
	source_overrides = {}

	def get_raw(col, default=0.0, lower=None, upper=None):
		if col in source_overrides:
			arr = source_overrides[col]
		elif col not in idx:
			source_numeric = context.get("source_numeric", {})
			if col in source_numeric:
				arr = np.nan_to_num(source_numeric[col], nan=default)
			else:
				arr = np.full(len(delta), default, dtype=np.float64)
		else:
			arr = np.nan_to_num(raw[:,idx[col]], nan=default)
		if lower is not None:
			arr = np.maximum(arr, lower)
		if upper is not None:
			arr = np.minimum(arr, upper)
		return arr

	def set_raw(col, value):
		if col in idx:
			raw[:,idx[col]] = value

	a_old = get_raw("A_score", 0.0, lower=0.0)
	s_old = get_raw("S_score", 0.0, lower=0.0)
	offline_scale = 1 + SEMANTIC_GAMMA * np.log1p(delta_a_arr / (a_old + FEATURE_UPDATE_EPS))
	online_scale = 1 + SEMANTIC_GAMMA * np.log1p(delta_s_arr / (s_old + FEATURE_UPDATE_EPS))
	a_new = a_old + delta_a_arr
	s_new = s_old + delta_s_arr
	set_raw("A_score", a_new)
	if "S_score" in idx:
		set_raw("S_score", s_new)
	for col in OFFLINE_ACTIVITY_FEATURES:
		if col in idx:
			set_raw(col, get_raw(col, 0.0, lower=0.0) * offline_scale)
	for col in ONLINE_ACTIVITY_FEATURES:
		if col in idx:
			set_raw(col, get_raw(col, 0.0, lower=0.0) * online_scale)
	for col in ["days_since_last_visit", "days_since_last_active"]:
		updated = get_raw(col, 999.0, lower=0.0) / np.where(offline_scale == 0, 1, offline_scale)
		if col in idx:
			set_raw(col, updated)
		else:
			source_overrides[col] = updated
	updated_interact_days = get_raw("days_since_last_interact", 999.0, lower=0.0) / np.where(online_scale == 0, 1, online_scale)
	if "days_since_last_interact" in idx:
		set_raw("days_since_last_interact", updated_interact_days)
	else:
		source_overrides["days_since_last_interact"] = updated_interact_days

	visit_days = get_raw("days_since_last_visit", 999.0, lower=0.0)
	buy_days = get_raw("days_since_last_buy", 999.0, lower=0.0)
	active_days = get_raw("days_since_last_active", 999.0, lower=0.0)
	interact_days = get_raw("days_since_last_interact", 999.0, lower=0.0)
	pt_days = get_raw("days_since_last_pt_buy", 999.0, lower=0.0)
	active_member = get_raw("active_member", 0.0, lower=0.0, upper=1.0)
	pt_bought = get_raw("pt_bought_before", 0.0, lower=0.0, upper=1.0)
	a_score = get_raw("A_score", 0.0, lower=0.0)
	s_score = get_raw("S_score", 0.0, lower=0.0)
	visit_count = get_raw("visit_count", 0.0, lower=0.0)
	order_count = get_raw("order_count", 0.0, lower=0.0)
	order_amount = get_raw("order_amount_total", 0.0, lower=0.0)
	member_amount = get_raw("member_amount", 0.0, lower=0.0)
	pt_amount = get_raw("pt_amount", 0.0, lower=0.0)
	pt_total = get_raw("pt_amount_total_before", 0.0, lower=0.0)
	pt_sessions_total = get_raw("pt_sessions_bought_before", 0.0, lower=0.0)
	anaerobic_count = get_raw("anaerobic_count", 0.0, lower=0.0)
	anaerobic_volume = get_raw("anaerobic_volume", 0.0, lower=0.0)
	discount_ratio = get_raw("discount_order_ratio", 0.0, lower=0.0, upper=1.0)
	dte = get_raw("dte", -999.0)

	set_raw("visit_recency_score", 1 / (visit_days + 1))
	set_raw("buy_recency_score", 1 / (buy_days + 1))
	set_raw("active_recency_score", 1 / (active_days + 1))
	set_raw("interact_recency_score", 1 / (interact_days + 1))
	pt_recency_score = 1 / (pt_days + 1)
	set_raw("pt_recency_score", pt_recency_score)
	set_raw("pt_stickiness_score", np.log1p(pt_total) * pt_recency_score)
	set_raw("pt_session_stickiness_score", np.log1p(pt_sessions_total) * pt_recency_score)
	set_raw("active_recent_score", a_score / (active_days + 1))
	set_raw("visit_recent_intensity", np.log1p(visit_count) / (visit_days + 1))
	set_raw("strength_frequency_score", np.log1p(anaerobic_count))
	set_raw("strength_volume_score", np.log1p(anaerobic_volume))
	set_raw("new_pt_strength_potential", (1 - pt_bought) * np.log1p(anaerobic_count + anaerobic_volume))
	set_raw("o2o_strength_score", a_score * np.log1p(s_score))
	set_raw("pt_share_amount", pt_amount / (order_amount + 1))
	set_raw("member_share_amount", member_amount / (order_amount + 1))
	set_raw("discount_intensity", discount_ratio * np.log1p(order_amount))
	set_raw("order_per_visit", order_count / (visit_count + 1))
	set_raw("spend_per_visit", order_amount / (visit_count + 1))
	set_raw("near_expiry_30", ((active_member == 1) & (dte >= 0) & (dte <= 30)).astype(float))
	set_raw("near_expiry_60", ((active_member == 1) & (dte >= 0) & (dte <= 60)).astype(float))
	set_raw("expired_member_flag", ((active_member == 0) | (dte < 0)).astype(float))
	set_raw("active_member_with_pt_history", ((active_member == 1) & (pt_bought == 1)).astype(float))
	set_raw("active_member_no_pt_history", ((active_member == 1) & (pt_bought == 0)).astype(float))

	num = np.where(np.isnan(raw), context["num_medians"], raw)
	num = log1p_array(num)
	x = np.concatenate([num, context["cat_encoded"]], axis=1)
	prob = context["model"].predict_proba(x)[:,1]
	if context["prior_correction"]:
		prob = prior_correct(prob, context["true_train_prior"], context["fit_sample_prior"])
	return pd.Series(prob, index=pop.index, name="p_pt_new").clip(0,1)


def prepare_serf_scoring_frame(df, meta):
	base_features = meta["base_features"]
	response_source_features = meta.get("response_source_features", [])
	categorical_features = meta["categorical_features"]
	required = list(dict.fromkeys(base_features + response_source_features + categorical_features))
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise KeyError(f"SERF 推理缺少字段: {missing}")
	frame = df.copy()
	for col in base_features:
		frame[col] = pd.to_numeric(frame[col], errors="coerce")
	for col in response_source_features:
		frame[col] = pd.to_numeric(frame[col], errors="coerce")
	for col in categorical_features:
		frame[col] = frame[col].astype("object").where(frame[col].notna(), SERF_CATEGORICAL_MISSING_TOKEN)
	frame = add_serf_response_features(frame, float(meta["active_target"]))
	feature_order = meta["feature_order"]
	missing_after_engineering = [col for col in feature_order if col not in frame.columns]
	if missing_after_engineering:
		raise KeyError(f"SERF 特征构造后仍缺少字段: {missing_after_engineering}")
	return frame[feature_order]


def score_serf_renewal_probability(df, model=None, meta=None):
	if model is None or meta is None:
		model, meta = load_serf_artifacts()
	x = prepare_serf_scoring_frame(df, meta)
	prob = model.predict_proba(x)[:,1]
	return pd.Series(prob, index=df.index, name="p_renew_base")


def build_serf_fast_context(pop, model, meta):
	scoring_source = pop.copy()
	if "serf_user_state" in scoring_source.columns:
		scoring_source["user_state"] = scoring_source["serf_user_state"]
	base_frame = prepare_serf_scoring_frame(scoring_source, meta)
	preprocess = model.named_steps["preprocess"]
	num_pipeline = preprocess.named_transformers_["num"]
	cat_pipeline = preprocess.named_transformers_["cat"]
	num_cols = list(preprocess.transformers_[0][2])
	cat_cols = list(preprocess.transformers_[1][2])
	base_numeric = base_frame[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
	cat_encoded = cat_pipeline.transform(base_frame[cat_cols])
	if hasattr(cat_encoded, "toarray"):
		cat_encoded = cat_encoded.toarray()
	return {
		"model": model.named_steps["model"],
		"num_cols": num_cols,
		"num_index": {col: idx for idx, col in enumerate(num_cols)},
		"base_numeric": base_numeric,
		"num_medians": num_pipeline.named_steps["imputer"].statistics_.astype(np.float64),
		"cat_encoded": np.asarray(cat_encoded, dtype=np.float64),
		"active_target": float(meta["active_target"]),
	}


def build_serf_counterfactual_source(df, delta_a, delta_s):
	out = df.copy()
	delta_a = pd.Series(delta_a, index=out.index).fillna(0).clip(lower=0)
	delta_s = pd.Series(delta_s, index=out.index).fillna(0).clip(lower=0)
	offline_scale, online_scale = active_response_scales(out, delta_a, delta_s)
	out["A_score"] = pd.to_numeric(out["A_score"], errors="coerce").fillna(0).clip(lower=0) + delta_a
	out["S_score"] = pd.to_numeric(out["S_score"], errors="coerce").fillna(0).clip(lower=0) + delta_s
	for col in OFFLINE_ACTIVITY_FEATURES:
		if col in out.columns:
			out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0) * offline_scale
	for col in ONLINE_ACTIVITY_FEATURES:
		if col in out.columns:
			out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0) * online_scale
	if "serf_user_state" in out.columns:
		out["user_state"] = out["serf_user_state"]
	return out, offline_scale, online_scale


def build_pt_counterfactual_source(df, delta_a, delta_s):
	out = df.copy()
	delta_a = pd.Series(delta_a, index=out.index).fillna(0).clip(lower=0)
	delta_s = pd.Series(delta_s, index=out.index).fillna(0).clip(lower=0)
	offline_scale, online_scale = active_response_scales(out, delta_a, delta_s)
	out["A_score"] = pd.to_numeric(out["A_score"], errors="coerce").fillna(0).clip(lower=0) + delta_a
	if "S_score" in out.columns:
		out["S_score"] = pd.to_numeric(out["S_score"], errors="coerce").fillna(0).clip(lower=0) + delta_s
	for col in OFFLINE_ACTIVITY_FEATURES:
		if col in out.columns:
			out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0) * offline_scale
	for col in ONLINE_ACTIVITY_FEATURES:
		if col in out.columns:
			out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0) * online_scale
	for col in ["days_since_last_visit", "days_since_last_active"]:
		if col in out.columns:
			base_days = pd.to_numeric(out[col], errors="coerce").fillna(999).clip(lower=0)
			out[col] = (base_days / offline_scale.replace(0, 1)).clip(lower=0)
	if "days_since_last_interact" in out.columns:
		base_days = pd.to_numeric(out["days_since_last_interact"], errors="coerce").fillna(999).clip(lower=0)
		out["days_since_last_interact"] = (base_days / online_scale.replace(0, 1)).clip(lower=0)
	if "pt_user_state" in out.columns:
		out["user_state"] = out["pt_user_state"]
	return out


def cached_pt_counterfactual_probability(pop, delta_a, delta_s, model, meta, cache, cache_key, pt_context=None):
	delta_a_series = pd.Series(delta_a, index=pop.index).fillna(0).clip(lower=0)
	delta_s_series = pd.Series(delta_s, index=pop.index).fillna(0).clip(lower=0)
	if float(delta_a_series.sum() + delta_s_series.sum()) <= EPS:
		return pop["p_pt_base"]
	prob_key = ("pt_prob", *cache_key)
	if cache is not None and prob_key in cache:
		return cache[prob_key]
	if pt_context is not None:
		prob = fast_score_pt_counterfactual_probability(pop, delta_a_series, delta_s_series, pt_context)
	else:
		source = build_pt_counterfactual_source(pop, delta_a_series, delta_s_series)
		prob = score_pt_probability(source, model, meta)
	if cache is not None:
		cache[prob_key] = prob
	return prob


def cached_pt_counterfactual_metrics(pop, delta_a, delta_s, price_p, model, meta, cache, cache_key, pt_context=None):
	delta_a_series = pd.Series(delta_a, index=pop.index).fillna(0).clip(lower=0)
	delta_s_series = pd.Series(delta_s, index=pop.index).fillna(0).clip(lower=0)
	if float(delta_a_series.sum() + delta_s_series.sum()) <= EPS:
		return float((pop["p_pt_base"] * price_p).sum()), 0.0
	metrics_key = ("pt_metrics", *cache_key)
	if cache is not None and metrics_key in cache:
		return cache[metrics_key]
	prob = cached_pt_counterfactual_probability(pop, delta_a_series, delta_s_series, model, meta, cache, cache_key, pt_context)
	metrics = (
		float((prob * price_p).sum()),
		float((prob - pop["p_pt_base"]).sum()),
	)
	if cache is not None:
		cache[metrics_key] = metrics
	return metrics


def prepare_serf_counterfactual_scoring_frame(df, delta_a, delta_s, meta):
	source, offline_scale, online_scale = build_serf_counterfactual_source(df, delta_a, delta_s)
	base_source = df.copy()
	if "serf_user_state" in base_source.columns:
		base_source["user_state"] = base_source["serf_user_state"]
	base_frame = prepare_serf_scoring_frame(base_source, meta)
	frame = prepare_serf_scoring_frame(source, meta)
	for col in SERF_L_RESPONSE_FEATURES:
		if col in frame.columns:
			scale = online_scale if col == "interact_recency_score" else offline_scale
			frame[col] = pd.to_numeric(base_frame[col], errors="coerce").fillna(0).clip(lower=0) * scale
	for col in SERF_RECENCY_SCORE_FEATURES:
		if col in frame.columns:
			frame[col] = frame[col].clip(0,1)
	return frame[meta["feature_order"]]


def score_serf_counterfactual_renewal_probability(df, delta_a, delta_s, model, meta):
	x = prepare_serf_counterfactual_scoring_frame(df, delta_a, delta_s, meta)
	prob = model.predict_proba(x)[:,1]
	return pd.Series(prob, index=df.index, name="p_renew_new").clip(0,1)


def fast_score_serf_counterfactual_probability(pop, delta_a, delta_s, context):
	delta_a_series = pd.Series(delta_a, index=pop.index).fillna(0).clip(lower=0)
	delta_s_series = pd.Series(delta_s, index=pop.index).fillna(0).clip(lower=0)
	if float(delta_a_series.sum() + delta_s_series.sum()) <= EPS:
		return pop["p_renew_base"]
	idx = context["num_index"]
	raw = context["base_numeric"].copy()
	delta_a_arr = delta_a_series.to_numpy(dtype=np.float64)
	delta_s_arr = delta_s_series.to_numpy(dtype=np.float64)
	a_old = np.nan_to_num(context["base_numeric"][:,idx["A_score"]], nan=0.0)
	a_old = np.maximum(a_old,0)
	s_old = np.zeros(len(delta_a_arr), dtype=np.float64)
	if "S_score" in idx:
		s_old = np.nan_to_num(context["base_numeric"][:,idx["S_score"]], nan=0.0)
		s_old = np.maximum(s_old,0)
	offline_scale = 1 + SEMANTIC_GAMMA * np.log1p(delta_a_arr / (a_old + FEATURE_UPDATE_EPS))
	online_scale = 1 + SEMANTIC_GAMMA * np.log1p(delta_s_arr / (s_old + FEATURE_UPDATE_EPS))
	a_new = a_old + delta_a_arr
	s_new = s_old + delta_s_arr
	raw[:,idx["A_score"]] = a_new
	if "S_score" in idx:
		raw[:,idx["S_score"]] = s_new
	for col in OFFLINE_ACTIVITY_FEATURES:
		if col in idx:
			base_value = np.nan_to_num(context["base_numeric"][:,idx[col]], nan=0.0)
			raw[:,idx[col]] = np.maximum(base_value,0) * offline_scale
	for col in ONLINE_ACTIVITY_FEATURES:
		if col in idx:
			base_value = np.nan_to_num(context["base_numeric"][:,idx[col]], nan=0.0)
			raw[:,idx[col]] = np.maximum(base_value,0) * online_scale
	if "active_gap" in idx:
		raw[:,idx["active_gap"]] = np.maximum(context["active_target"] - a_new,0)
	if "o2o_strength" in idx:
		raw[:,idx["o2o_strength"]] = a_new * np.log1p(np.maximum(s_new,0))
	for col in SERF_L_RESPONSE_FEATURES:
		if col in idx:
			base_value = np.nan_to_num(context["base_numeric"][:,idx[col]], nan=0.0)
			scale = online_scale if col == "interact_recency_score" else offline_scale
			raw[:,idx[col]] = np.maximum(base_value,0) * scale
	for col in SERF_RECENCY_SCORE_FEATURES:
		if col in idx:
			raw[:,idx[col]] = np.clip(raw[:,idx[col]],0,1)
	num = np.where(np.isnan(raw), context["num_medians"], raw)
	num = log1p_array(num)
	x = np.concatenate([num, context["cat_encoded"]], axis=1)
	prob = context["model"].predict_proba(x)[:,1]
	return pd.Series(prob, index=pop.index, name="p_renew_new").clip(0,1)


def score_serf_counterfactual_probability(pop, delta_a, delta_s, model, meta, serf_context=None):
	if serf_context is not None:
		return fast_score_serf_counterfactual_probability(pop, delta_a, delta_s, serf_context)
	return score_serf_counterfactual_renewal_probability(pop, delta_a, delta_s, model, meta)


def cached_serf_counterfactual_probability(pop, delta_a, delta_s, model, meta, cache, cache_key, serf_context=None):
	delta_a_series = pd.Series(delta_a, index=pop.index).fillna(0).clip(lower=0)
	delta_s_series = pd.Series(delta_s, index=pop.index).fillna(0).clip(lower=0)
	if float(delta_a_series.sum() + delta_s_series.sum()) <= EPS:
		return pop["p_renew_base"]
	prob_key = ("prob", *cache_key)
	if cache is not None and prob_key in cache:
		return cache[prob_key]
	prob = score_serf_counterfactual_probability(
		pop,
		delta_a_series,
		delta_s_series,
		model=model,
		meta=meta,
		serf_context=serf_context,
	)
	if cache is not None:
		cache[prob_key] = prob
	return prob


def cached_serf_counterfactual_metrics(pop, delta_a, delta_s, price_m, model, meta, cache, cache_key, serf_context=None):
	delta_a_series = pd.Series(delta_a, index=pop.index).fillna(0).clip(lower=0)
	delta_s_series = pd.Series(delta_s, index=pop.index).fillna(0).clip(lower=0)
	if float(delta_a_series.sum() + delta_s_series.sum()) <= EPS:
		return float((pop["p_renew_base"] * price_m).sum()), 0.0
	metrics_key = ("metrics", *cache_key)
	if cache is not None and metrics_key in cache:
		return cache[metrics_key]
	prob = score_serf_counterfactual_probability(
		pop,
		delta_a_series,
		delta_s_series,
		model=model,
		meta=meta,
		serf_context=serf_context,
	)
	metrics = (
		float((prob * price_m).sum()),
		float((prob - pop["p_renew_base"]).sum()),
	)
	if cache is not None:
		cache[metrics_key] = metrics
	return metrics


def group_bonus(strategy, state):
	if state == "Z1_高活跃高消费用户":
		return strategy["bonus_z1"]
	if state == "Z6_私教潜在用户":
		return strategy["bonus_z6"]
	if state == "Z7_流失风险用户":
		return strategy["bonus_z7"]
	return min(strategy["bonus_z6"], strategy["bonus_z7"])


def compute_response_params(pop):
	# 修复Bug1: 取前 25% 高活跃用户的均值作为 target，而不是全体的 75分位数
	q75 = pop["A_score"].quantile(0.75)
	if q75 <= 0:
		q75 = pop["A_score"].mean() + 1e-5  # 防止全体为0的极端情况
	target_a = pop.loc[pop["A_score"] >= q75, "A_score"].mean()
	q75_s = pop["S_score"].quantile(0.75)
	if q75_s <= 0:
		q75_s = pop["S_score"].mean() + 1e-5
	target_s = pop.loc[pop["S_score"] >= q75_s, "S_score"].mean()

	params = {}
	for state in GROUP_PARAMS:
		mask = pop["user_state"] == state
		if mask.sum() == 0:
			mean_a = pop["A_score"].mean()
			mean_s = pop["S_score"].mean()
		else:
			mean_a = pop.loc[mask, "A_score"].mean()
			mean_s = pop.loc[mask, "S_score"].mean()

		base = GROUP_PARAMS[state]
		# 给一个保底的活跃提升空间 (2.0)，防止活跃用户 alpha 为 0 导致发积分失效
		alpha_base = max(base["rho"] * (target_a - mean_a), base["rho"] * 2.0)
		alpha_s_base = max(base["rho"] * (target_s - mean_s), base["rho"] * 2.0)
		beta_p0_base = -np.log(1 - base["q"]) / base["pref_p0"]
		beta_p1_base = -np.log(1 - base["q"]) / base["pref_p1"]
		params[state] = {
			"alpha_a_base": alpha_base,
			"alpha_s_base": alpha_s_base,
			"beta_p0_base": beta_p0_base,
			"beta_p1_base": beta_p1_base,
			**base,
		}
	return params


def build_scenario(carry_scenario, delay_count):
	return {
		"观察期标签": carry_scenario["观察期标签"],
		"delay_window_count": delay_count,
		"总窗口数":1 + delay_count,
		"carry_a": carry_scenario["carry_a"],
	}


def simulate_strategy(
	pop,
	strategy,
	response_params,
	scenario,
	serf_model,
	serf_meta,
	pt_model,
	pt_meta,
	serf_cache=None,
	pt_cache=None,
	serf_context=None,
	pt_context=None,
):
		carry_a = scenario["carry_a"]
		delay_window_count = scenario["delay_window_count"]

		df = pop[["user_id", "user_state", "p_renew_base", "p_pt_base", "Price_m", "Price_p", "N_eff", "I_eff",
				  "A_score", "S_score"]].copy()
		df["bonus"] = df["user_state"].map(lambda s: group_bonus(strategy, s))
		alpha_a = df["user_state"].map(lambda s: response_params[s]["alpha_a_base"] * strategy["rho_scale"])
		alpha_s = df["user_state"].map(lambda s: response_params[s]["alpha_s_base"] * strategy["rho_scale"])
		beta_p0 = df["user_state"].map(lambda s: response_params[s]["beta_p0_base"] * strategy["q_scale"])
		beta_p1 = df["user_state"].map(lambda s: response_params[s]["beta_p1_base"] * strategy["q_scale"])
		df["delta_A_w1"] = alpha_a * (1 - np.exp(-beta_p0 * strategy["p0"]))
		df["delta_S_w1"] = alpha_s * (1 - np.exp(-beta_p1 * strategy["p1"]))
		offline_scale_w1, online_scale_w1 = active_response_scales(pop, df["delta_A_w1"], df["delta_S_w1"])
		df["N_eff_new"] = (df["N_eff"] * offline_scale_w1).clip(upper=12)
		df["I_eff_new"] = (df["I_eff"] * online_scale_w1).clip(upper=20)
		df["points_w1"] = np.minimum(
			strategy["p0"] * df["N_eff_new"] + strategy["p1"] * df["I_eff_new"] + df["bonus"],
			strategy["pmax"],
		)

		p_base = {"renew": df["p_renew_base"], "pt": df["p_pt_base"]}
		p_renew_curr = cached_serf_counterfactual_probability(
			pop,
			df["delta_A_w1"],
			df["delta_S_w1"],
			model=serf_model,
			meta=serf_meta,
			cache=serf_cache,
			cache_key=("active", strategy["策略编号"]),
			serf_context=serf_context,
		)
		p_pt_curr = cached_pt_counterfactual_probability(
			pop,
			df["delta_A_w1"],
			df["delta_S_w1"],
			model=pt_model,
			meta=pt_meta,
			cache=pt_cache,
			cache_key=("active", strategy["策略编号"]),
			pt_context=pt_context,
		)
		p_renew_lift_curr = float((p_renew_curr - df["p_renew_base"]).sum())
		p_pt_lift_curr = float((p_pt_curr - df["p_pt_base"]).sum())

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
			"window_index": 1,
			"window_type": "active",
			"window_weight": WINDOW1_WEIGHT,
			"会员收入": rm_w1,
			"私教收入": rp_w1,
			"积分成本": cost_w1,
			"平台利润贡献": 0.35 * rm_w1 + 0.30 * rp_w1 - cost_w1,
			"累计平台利润": 0.35 * rm_w1 + 0.30 * rp_w1 - cost_w1,
			"窗口末续费概率提升": p_renew_lift_curr,
			"窗口末私教购买概率提升": p_pt_lift_curr,
			"窗口私教收入提升": float((WINDOW1_WEIGHT * (p_pt_curr - df["p_pt_base"]) * df["Price_p"]).sum()),
			"窗口活跃提升": float(df["delta_A_w1"].sum()),
			"窗口线上活跃提升": float(df["delta_S_w1"].sum()),
			"窗口结算积分": float(df["points_w1"].sum()),
		}]

		delayed_rm_total = 0.0
		delayed_rp_total = 0.0
		delta_a_total = float(df["delta_A_w1"].sum())
		delta_s_total = float(df["delta_S_w1"].sum())
		delta_a_prev = df["delta_A_w1"]
		delta_s_prev = df["delta_S_w1"]
		cum_profit = trajectory_rows[0]["累计平台利润"]

		for window_idx in range(2, delay_window_count + 2):
			delta_a_curr = carry_a * delta_a_prev
			delta_s_curr = carry_a * delta_s_prev
			delay_step = window_idx - 1

			renew_income_sum, p_renew_lift_curr = cached_serf_counterfactual_metrics(
				pop,
				delta_a_curr,
				delta_s_curr,
				df["Price_m"],
				model=serf_model,
				meta=serf_meta,
				cache=serf_cache,
				cache_key=("delayed", strategy["策略编号"], float(carry_a), delay_step),
				serf_context=serf_context,
			)
			pt_income_sum, p_pt_lift_curr = cached_pt_counterfactual_metrics(
				pop,
				delta_a_curr,
				delta_s_curr,
				df["Price_p"],
				model=pt_model,
				meta=pt_meta,
				cache=pt_cache,
				cache_key=("delayed", strategy["策略编号"], float(carry_a), delay_step),
				pt_context=pt_context,
			)

			rm_curr = DELAYED_WINDOW_WEIGHT * renew_income_sum
			rp_curr = DELAYED_WINDOW_WEIGHT * pt_income_sum
			profit_curr = 0.35 * rm_curr + 0.30 * rp_curr
			cum_profit += profit_curr
			delayed_rm_total += rm_curr
			delayed_rp_total += rp_curr
			delta_a_total += float(delta_a_curr.sum())
			delta_s_total += float(delta_s_curr.sum())

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
				"积分成本": 0.0,
				"平台利润贡献": profit_curr,
				"累计平台利润": cum_profit,
				"窗口末续费概率提升": p_renew_lift_curr,
				"窗口末私教购买概率提升": p_pt_lift_curr,
				"窗口私教收入提升": float(DELAYED_WINDOW_WEIGHT * (pt_income_sum - (df["p_pt_base"] * df["Price_p"]).sum())),
				"窗口活跃提升": float(delta_a_curr.sum()),
				"窗口线上活跃提升": float(delta_s_curr.sum()),
				"窗口结算积分": 0.0,
			})
			delta_a_prev = delta_a_curr
			delta_s_prev = delta_s_curr

		rm_total = rm_w1 + delayed_rm_total
		rp_total = rp_w1 + delayed_rp_total
		cost_total = cost_w1
		profit_total = 0.35 * rm_total + 0.30 * rp_total - cost_total
		cost_rate = cost_total / max(rm_total + rp_total, EPS)

		summary = {
			"观察期标签": scenario["观察期标签"],
			"delay_window_count": delay_window_count,
			"总窗口数": scenario["总窗口数"],
			"窗口1权重": WINDOW1_WEIGHT,
			"延期窗口权重": DELAYED_WINDOW_WEIGHT,
			"延续活跃系数": carry_a,
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
			"总线上活跃提升": delta_s_total,
			"最终续费概率提升": p_renew_lift_curr,
			"最终私教购买概率提升": p_pt_lift_curr,
			"私教期望购买人数提升": p_pt_lift_curr,
			"私教收入提升": float(rp_total - (WINDOW1_WEIGHT + delay_window_count * DELAYED_WINDOW_WEIGHT) * (df["p_pt_base"] * df["Price_p"]).sum()),
			"成本收入比": cost_rate,
			"是否利润非负": profit_total >= 0,
			"是否满足成本约束": cost_rate <= ETA,
		}
		trajectory_df = pd.DataFrame(trajectory_rows)
		return summary, trajectory_df


def compute_s0_gap_table(summary_df):
	base = summary_df[summary_df["策略编号"] == "S0"][["观察期标签", "delay_window_count", "平台利润"]].rename(columns={"平台利润": "S0平台利润"})
	s5 = summary_df[summary_df["策略编号"] == "S5"][["观察期标签", "delay_window_count", "总窗口数", "延续活跃系数", "平台利润"]].rename(columns={"平台利润": "S5平台利润"})
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
			"最大测试延期窗口数": int(last["delay_window_count"]),
			"最大测试总窗口数": int(last["总窗口数"]),
			"最大测试窗口内是否超过S0": has_cross,
			"首次超过S0的delay_window_count": first_delay,
			"首次超过S0的总窗口数": first_total,
			"首次超过S0时利润差": first_gap,
			"最大测试窗口时利润差": float(last["利润差_S5减S0"]),
		})
	return pd.DataFrame(rows)


def compute_focus_strategy_comparison(summary_df):
	focus = summary_df[summary_df["策略编号"].isin(["S3", "S4", "S5"])].copy()
	if focus.empty:
		return focus
	cols = [
		"观察期标签",
		"delay_window_count",
		"总窗口数",
		"策略编号",
		"策略名称",
		"平台利润",
		"私教收入",
		"私教收入提升",
		"最终私教购买概率提升",
		"私教期望购买人数提升",
		"积分成本",
		"成本收入比",
		"是否满足成本约束",
	]
	return focus[cols].sort_values(["观察期标签", "delay_window_count", "策略编号"])


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

	focus_df = plot_df[plot_df["策略编号"].isin(["S3", "S4", "S5"])].copy()
	if not focus_df.empty:
		fig, axes = plt.subplots(1, 3, figsize=(12,4))
		colors = ["#7aa6c2", "#8ab17d", "#d49a6a"]
		axes[0].bar(focus_df["策略编号"], focus_df["平台利润"], color=colors)
		axes[0].set_title("平台利润")
		axes[0].set_ylabel("金额")
		axes[1].bar(focus_df["策略编号"], focus_df["私教收入提升"], color=colors)
		axes[1].set_title("私教收入提升")
		axes[2].bar(focus_df["策略编号"], focus_df["最终私教购买概率提升"], color=colors)
		axes[2].set_title("私教期望购买人数提升")
		for ax in axes:
			ax.grid(axis="y", alpha=0.25)
		fig.suptitle("接入增强私教购买模型后的S3/S4/S5对比")
		fig.tight_layout()
		fig.savefig(OUT_DIR / "5.6_图2_S3_S4_S5私教模型接入后对比.png", bbox_inches="tight")
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

会员续费基线概率不再来自预测 CSV 中的 `普通随机森林_prob`，而是直接使用导出的 SERF 工件 `serf_renewal_h30_model.joblib` 在线推理得到的 $\\hat p_i^{{renew,30}}$。策略实施后，脚本先用线下积分系数 $p_0$ 和线上积分系数 $p_1$ 触发用户线下活跃响应 $\\Delta A_i$ 和线上互动响应 $\\Delta S_i$；再将到店次数、在馆时长、跑步距离、运动消耗、无氧训练量等线下特征，以及发帖、点赞、评论、被点赞、被评论等线上特征按响应强度等比放大；随后按响应后的有效线下和线上行为计算窗口 1 实际获得积分；最后重新调用同一 SERF 工件和增强私教购买模型得到反事实续费概率与私教购买概率，不再使用 `lambda_renew` 或 `lambda_pt` 线性加成。由于历史数据中没有真实积分干预记录，积分带来的活跃提升和概率变化属于基于响应假设的情景模拟结果，不解释为真实因果效应。

##2. 两阶段窗口设定

本次基准模拟将策略效果划分为两个时间窗口：窗口1为返利机制实施期，窗口2为返利结束后的延续观察期。窗口1计入积分发放与兑换成本，窗口2不再新增积分成本，但保留窗口1形成的部分活跃提升惯性，并据此构造延期窗口的 SERF 反事实续费特征。

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

本文件由 `模型输出结果/simulate_points_strategy_topsis.py` 自动生成，用于比较在**窗口1权重=1.0、每个延期观察窗口权重=1.0**时，S5 相对无积分基准策略 S0 的长期累计利润变化。续费基线概率沿用主报告中的同一套 SERF 工件在线推理结果；策略后的续费概率由更新后的反事实特征再次输入 SERF 得到，而不是使用 `lambda_renew` 加成。

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


def load_population():
	model, meta = load_serf_artifacts()
	pt_model, pt_meta = load_pt_artifacts()
	renew_data_cols = list(dict.fromkeys([
		"user_id",
		"cycle_id",
		"end_date",
		*meta["base_features"],
		*meta.get("response_source_features", []),
		*meta["categorical_features"],
		*pt_meta["base_features"],
		*pt_meta.get("engineering_source_features", []),
		*pt_meta["categorical_features"],
		"current_member_price",
		"visit_count",
		"post_count",
		"like_given_count",
		"comment_given_count",
		"A_score",
		"S_score",
		"user_state",
	]))
	renew_data = pd.read_csv(RENEW_DATA_PATH, usecols=renew_data_cols, low_memory=False)
	renew_data["end_date"] = pd.to_datetime(renew_data["end_date"], errors="coerce")
	pop = renew_data.copy()
	pop["p_renew_base"] = score_serf_renewal_probability(renew_data, model=model, meta=meta)
	pop["serf_user_state"] = pop["user_state"]
	pop["pt_user_state"] = pop["user_state"]

	pt_data_cols = ["user_id", "pt_unit_price_est", "pt_purchase_amount_next", "y_pt"]
	pt_data = pd.read_csv(PT_DATA_PATH, usecols=pt_data_cols, low_memory=False)
	positive_amount_mean = pt_data.loc[pt_data["y_pt"] == 1, "pt_purchase_amount_next"].replace(0, np.nan).dropna().mean()
	if pd.isna(positive_amount_mean):
		positive_amount_mean = 300.0
	pt_price_by_user = pt_data.groupby("user_id", as_index=False)["pt_unit_price_est"].max().rename(
		columns={"pt_unit_price_est": "pt_unit_price_est_hist"})
	pop = pop.merge(pt_price_by_user, on="user_id", how="left")
	pop["pt_unit_price_est"] = pd.to_numeric(pop["pt_unit_price_est"], errors="coerce").combine_first(
		pd.to_numeric(pop["pt_unit_price_est_hist"], errors="coerce"))

	pop["p_renew_base"] = pop["p_renew_base"].fillna(pop["p_renew_base"].mean()).clip(0, 1)
	pop["Price_m"] = pd.to_numeric(pop["current_member_price"], errors="coerce").fillna(
		pop["current_member_price"].median()).clip(lower=1)
	pop["Price_p"] = pd.to_numeric(pop["pt_unit_price_est"], errors="coerce").fillna(0)
	pop.loc[pop["Price_p"] <= 0, "Price_p"] = positive_amount_mean
	pop["Price_p"] = pop["Price_p"].clip(lower=1)
	pt_scoring_pop = pop.copy()
	pt_scoring_pop["user_state"] = pt_scoring_pop["pt_user_state"]
	pop["p_pt_base"] = score_pt_probability(pt_scoring_pop, pt_model, pt_meta)
	pop["p_pt_base"] = pop["p_pt_base"].fillna(pop["p_pt_base"].mean()).clip(0, 1)
	pop["visit_count"] = pd.to_numeric(pop["visit_count"], errors="coerce").fillna(0).clip(lower=0)
	for col in ["post_count", "like_given_count", "comment_given_count", "A_score", "S_score"]:
		pop[col] = pd.to_numeric(pop[col], errors="coerce").fillna(0).clip(lower=0)
	# user_state 在 SERF 打分之后再映射到策略分组，避免干扰训练时的类别口径。
	pop["user_state"] = pop["user_state"].fillna("其他")
	pop.loc[~pop["user_state"].isin(GROUP_PARAMS.keys()), "user_state"] = "其他"
	pop["N_eff"] = pop["visit_count"].clip(upper=12)
	pop["I_eff"] = (pop["post_count"] + pop["like_given_count"] + pop["comment_given_count"]).clip(upper=20)
	return pop, model, meta, pt_model, pt_meta


def main():
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	pop, serf_model, serf_meta, pt_model, pt_meta = load_population()
	response_params = compute_response_params(pop)
	serf_context = build_serf_fast_context(pop, serf_model, serf_meta)
	pt_context = build_pt_fast_context(pop, pt_model, pt_meta)
	serf_cache = {}
	pt_cache = {}

	base_scenario = build_scenario(LONG_HORIZON_CARRY_SCENARIOS[1], BASE_DELAY_COUNT)
	base_summaries = []
	base_details = []
	for strategy in STRATEGIES:
		summary, detail = simulate_strategy(
			pop,
			strategy,
			response_params,
			base_scenario,
			serf_model,
			serf_meta,
			pt_model,
			pt_meta,
			serf_cache,
			pt_cache,
			serf_context,
			pt_context,
		)
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
				summary, detail = simulate_strategy(
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
				all_summaries.append(summary)
				all_trajectories.append(detail)

	multi_summary_df = pd.DataFrame(all_summaries)
	multi_trajectory_df = pd.concat(all_trajectories, ignore_index=True)
	gap_df = compute_s0_gap_table(multi_summary_df)
	crossover_df = compute_crossover_table(gap_df)
	focus_comparison_df = compute_focus_strategy_comparison(multi_summary_df)

	multi_summary_df.to_csv(MULTI_WINDOW_SUMMARY_PATH, index=False, encoding="utf-8-sig")
	multi_trajectory_df.to_csv(MULTI_WINDOW_TRAJECTORY_PATH, index=False, encoding="utf-8-sig")
	gap_df.to_csv(S5_VS_S0_PATH, index=False, encoding="utf-8-sig")
	crossover_df.to_csv(S5_VS_S0_CROSSOVER_PATH, index=False, encoding="utf-8-sig")
	focus_comparison_df.to_csv(S3_S4_S5_COMPARISON_PATH, index=False, encoding="utf-8-sig")
	write_multi_window_report(multi_summary_df, gap_df, crossover_df)

	print(f"已生成积分策略模拟与TOPSIS结果：{REPORT_PATH}")
	print(f"已生成多延迟观察窗口分析：{MULTI_WINDOW_REPORT_PATH}")
	print(f"已生成S3/S4/S5私教模型接入后对比：{S3_S4_S5_COMPARISON_PATH}")


if __name__ == "__main__":
	main()
