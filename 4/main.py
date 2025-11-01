# callcenter_pipeline_fixed.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy

# --- LOAD DATA ---
train_df = pd.read_csv("train.csv", parse_dates=["create_time"])
webstat_df = pd.read_csv("t1_webstat.csv", parse_dates=["date_time"])

# --- LABEL ---
y = train_df["is_callcenter"]

# --- PAGEVIEW PROCESSING ---
# Ensure proper ordering
webstat_df = webstat_df.sort_values(["sessionkey_id", "date_time"])

# Get max page number per session
max_page_per_session = webstat_df.groupby("sessionkey_id")["pageview_number"].max().rename("max_pageview_number")
train_df = train_df.merge(max_page_per_session, on="sessionkey_id", how="left")

# Get page number of order session
page_at_order = webstat_df.groupby("sessionkey_id")["pageview_number"].last().rename("pageview_number")
train_df = train_df.merge(page_at_order, on="sessionkey_id", how="left")

# Compute pages_after_order
train_df["pages_after_order"] = train_df["max_pageview_number"] - train_df["pageview_number"]
train_df["pages_after_order"] = train_df["pages_after_order"].clip(lower=0)

# --- COMPUTE order_vs_session_gap_min ---
session_last_time = webstat_df.groupby("sessionkey_id")["date_time"].max().rename("session_last_time")
train_df = train_df.merge(session_last_time, on="sessionkey_id", how="left")
train_df["order_vs_session_gap_min"] = (train_df["create_time"] - train_df["session_last_time"]).dt.total_seconds() / 60

# --- FEATURE ENGINEERING ---
train_df["order_hour"] = train_df["create_time"].dt.hour
train_df["order_dow"] = train_df["create_time"].dt.dayofweek
train_df["order_day"] = train_df["create_time"].dt.day
train_df["order_month"] = train_df["create_time"].dt.month
train_df["order_week"] = train_df["create_time"].dt.isocalendar().week.astype(int)
russian_holidays = {(1, 1), (1, 7), (5, 1), (5, 9), (6, 12), (11, 4)}
train_df["is_holiday"] = train_df["create_time"].apply(lambda dt: (dt.month, dt.day) in russian_holidays).astype(int)

train_df["gap_bucket"] = pd.cut(
    train_df["order_vs_session_gap_min"].fillna(9999),
    bins=[-1e6, 0, 5, 30, 60, 300, 1e6],
    labels=["<=0", "0-5", "5-30", "30-60", "60-300", "300+"]
)

# Combo interaction features
train_df["combo_utm4_gap5_30"] = ((train_df["utm_medium"] == 4) & (train_df["gap_bucket"] == "5-30")).astype(int)
train_df["combo_utm5_gap5_30"] = ((train_df["utm_medium"] == 5) & (train_df["gap_bucket"] == "5-30")).astype(int)
train_df["combo_utm1_gap5_30"] = ((train_df["utm_medium"] == 1) & (train_df["gap_bucket"] == "5-30")).astype(int)

train_df["price_bin"] = pd.cut(train_df["price"], bins=[0, 500, 2000, 10000, 1e6], labels=["<500", "500-2k", "2k-10k", "10k+"])
train_df["qty_bin"] = pd.cut(train_df["goods_qty"], bins=[0, 1, 3, 10, 100], labels=["1", "2-3", "4-10", "10+"])
train_df["pages_bin"] = pd.cut(train_df["pages_after_order"], bins=[-1, 0, 5, 10, 30, 1000], labels=["0", "1-5", "6-10", "11-30", "30+"])
train_df["rating_bin"] = pd.cut(train_df["rating_count"], bins=[-1, 0, 5, 20, 1000], labels=["0", "1-5", "6-20", "20+"])

train_df["combo_root1504_no_after"] = ((train_df["root_id"] == 1504) & (train_df["pages_bin"] == "0")).astype(int)
train_df["combo_price10k_bulk"] = ((train_df["price_bin"] == "10k+") & (train_df["qty_bin"].isin(["2-3", "10+"]))).astype(int)
train_df["combo_no_reviews_no_pages"] = ((train_df["rating_bin"] == "0") & (train_df["pages_bin"] == "0")).astype(int)

# Session-level features
session_duration = webstat_df.groupby("sessionkey_id")["pageview_duration_sec"].sum().rename("session_duration_sec")
session_pages = webstat_df.groupby("sessionkey_id")["page_type"].count().rename("session_total_pages")
session_entropy = webstat_df.groupby("sessionkey_id")["page_type"].apply(lambda x: entropy(x.value_counts(normalize=True))).rename("session_page_entropy")
session_std = webstat_df.groupby("sessionkey_id")["pageview_duration_sec"].std().rename("session_dwell_std")
webstat_df["is_help_page"] = webstat_df["page_type"].isin([10, 11, 12, 13])
max_help_dwell = webstat_df[webstat_df["is_help_page"]].groupby("sessionkey_id")["pageview_duration_sec"].max().rename("max_help_dwell_sec")

train_df = train_df.merge(session_duration, on="sessionkey_id", how="left")
train_df = train_df.merge(session_pages, on="sessionkey_id", how="left")
train_df = train_df.merge(session_entropy, on="sessionkey_id", how="left")
train_df = train_df.merge(session_std, on="sessionkey_id", how="left")
train_df = train_df.merge(max_help_dwell, on="sessionkey_id", how="left")
train_df["session_speed_sec_per_page"] = train_df["session_duration_sec"] / train_df["session_total_pages"]
train_df["max_help_dwell_sec"] = train_df["max_help_dwell_sec"].clip(lower=0)

# Final features
features = [
    'order_hour', 'order_dow', 'order_day', 'order_month', 'order_week',
    'is_holiday', 'session_duration_sec', 'session_total_pages', 'session_speed_sec_per_page',
    'combo_utm4_gap5_30', 'combo_utm5_gap5_30', 'combo_utm1_gap5_30',
    'combo_root1504_no_after', 'combo_price10k_bulk', 'combo_no_reviews_no_pages',
    'session_page_entropy', 'session_dwell_std', 'max_help_dwell_sec'
]

X = train_df[features].copy()
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].fillna(X[col].mode()[0])
for col in X.select_dtypes(include="number").columns:
    X[col] = X[col].fillna(X[col].mean())

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include="number").columns.tolist()

pipe = Pipeline([
    ("preprocess", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])),
    ("clf", LogisticRegression(solver="liblinear", max_iter=500))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
for train_idx, valid_idx in cv.split(X, y):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict_proba(X_valid)[:, 1]
    aucs.append(roc_auc_score(y_valid, y_pred))
    
print(aucs, np.mean(aucs))

test_df = pd.read_csv("test.csv", parse_dates=["create_time"])

# Merge session data into test
test_df = test_df.merge(max_page_per_session, on="sessionkey_id", how="left")
test_df = test_df.merge(page_at_order, on="sessionkey_id", how="left")
test_df["pages_after_order"] = test_df["max_pageview_number"] - test_df["pageview_number"]
test_df["pages_after_order"] = test_df["pages_after_order"].clip(lower=0)

test_df = test_df.merge(session_last_time, on="sessionkey_id", how="left")
test_df["order_vs_session_gap_min"] = (test_df["create_time"] - test_df["session_last_time"]).dt.total_seconds() / 60

# Feature engineering
test_df["order_hour"] = test_df["create_time"].dt.hour
test_df["order_dow"] = test_df["create_time"].dt.dayofweek
test_df["order_day"] = test_df["create_time"].dt.day
test_df["order_month"] = test_df["create_time"].dt.month
test_df["order_week"] = test_df["create_time"].dt.isocalendar().week.astype(int)
test_df["is_holiday"] = test_df["create_time"].apply(lambda dt: (dt.month, dt.day) in russian_holidays).astype(int)

test_df["gap_bucket"] = pd.cut(
    test_df["order_vs_session_gap_min"].fillna(9999),
    bins=[-1e6, 0, 5, 30, 60, 300, 1e6],
    labels=["<=0", "0-5", "5-30", "30-60", "60-300", "300+"]
)

test_df["combo_utm4_gap5_30"] = ((test_df["utm_medium"] == 4) & (test_df["gap_bucket"] == "5-30")).astype(int)
test_df["combo_utm5_gap5_30"] = ((test_df["utm_medium"] == 5) & (test_df["gap_bucket"] == "5-30")).astype(int)
test_df["combo_utm1_gap5_30"] = ((test_df["utm_medium"] == 1) & (test_df["gap_bucket"] == "5-30")).astype(int)

test_df["price_bin"] = pd.cut(test_df["price"], bins=[0, 500, 2000, 10000, 1e6], labels=["<500", "500-2k", "2k-10k", "10k+"])
test_df["qty_bin"] = pd.cut(test_df["goods_qty"], bins=[0, 1, 3, 10, 100], labels=["1", "2-3", "4-10", "10+"])
test_df["pages_bin"] = pd.cut(test_df["pages_after_order"], bins=[-1, 0, 5, 10, 30, 1000], labels=["0", "1-5", "6-10", "11-30", "30+"])
test_df["rating_bin"] = pd.cut(test_df["rating_count"], bins=[-1, 0, 5, 20, 1000], labels=["0", "1-5", "6-20", "20+"])

test_df["combo_root1504_no_after"] = ((test_df["root_id"] == 1504) & (test_df["pages_bin"] == "0")).astype(int)
test_df["combo_price10k_bulk"] = ((test_df["price_bin"] == "10k+") & (test_df["qty_bin"].isin(["2-3", "10+"]))).astype(int)
test_df["combo_no_reviews_no_pages"] = ((test_df["rating_bin"] == "0") & (test_df["pages_bin"] == "0")).astype(int)

# Merge session aggregates into test
test_df = test_df.merge(session_duration, on="sessionkey_id", how="left")
test_df = test_df.merge(session_pages, on="sessionkey_id", how="left")
test_df = test_df.merge(session_entropy, on="sessionkey_id", how="left")
test_df = test_df.merge(session_std, on="sessionkey_id", how="left")
test_df = test_df.merge(max_help_dwell, on="sessionkey_id", how="left")
test_df["session_speed_sec_per_page"] = test_df["session_duration_sec"] / test_df["session_total_pages"]
test_df["max_help_dwell_sec"] = test_df["max_help_dwell_sec"].clip(lower=0)

# Extract test features
X_test = test_df[features].copy()
for col in X_test.select_dtypes(include="object").columns:
    X_test[col] = X_test[col].fillna(X[col].mode()[0])
for col in X_test.select_dtypes(include="number").columns:
    X_test[col] = X_test[col].fillna(X[col].mean())
 
# Refit on full training data
print(X.shape)

pipe.fit(X, y)
test_preds = pipe.predict_proba(X_test)[:, 1]

# Save prediction
pd.DataFrame({
    "order_id": test_df["order_id"],
    "is_callcenter": test_preds
}).to_csv("submission.csv", index=False)
3