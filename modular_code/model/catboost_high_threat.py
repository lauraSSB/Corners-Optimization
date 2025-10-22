"""
CatBoost Classification: High-threat (xG > P75) vs Low-threat (<= P75)

- Computes the 75th percentile threshold and splits train/val on the TRAIN fold only (prevents data leakage).
- Creates target: 1 = high-threat, 0 = low-threat.
- Adds class weights to address the class imbalance.
- Light hyperparameter search via model.grid_search.
- Evaluates AUC, PR-AUC, F1 (with threshold tuned on validation for F1).
- Exports SHAP mean(|value|) per feature to CSV.
- Exports metrics, feature importance (gain), mean |SHAP|, and optionally:
    * shap_full_val.csv     -> ALL features per-sample SHAP on validation set
    * shap_compact_val.csv  -> Top-K features per-sample SHAP on validation set

USAGE (example):
    python catboost_high_threat.py \
  --data /path/to/your.csv \
  --xg_col xg_20s \
  --id_cols match_id,event_id \
  --val_size 0.2 \
  --export_shap_full \
  --export_shap_compact
"""

import argparse
import json
import math
import numpy as np
import pandas as pd

from typing import List, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, classification_report, confusion_matrix

from catboost import CatBoostClassifier, Pool

def compute_threshold(train_xg: pd.Series, quantile: float = 0.75) -> float:
    # robust to NaNs
    return float(np.nanquantile(train_xg.values, quantile))

def make_target(xg: pd.Series, threshold: float) -> pd.Series:
    # 1 = High threat, 0 = Low threat
    return (xg > threshold).astype(int)

def pick_best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    # Search thresholds on PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # skip the first point where threshold is undefined
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 37):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    preds = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    f1 = f1_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)
    rep = classification_report(y_true, preds, output_dict=True, zero_division=0)
    return {
        "auc": float(auc),
        "average_precision": float(ap),
        "f1": float(f1),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "classification_report": rep
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV")
    parser.add_argument("--xg_col", type=str, default="xg_20s", help="Column name with xG values")
    parser.add_argument("--id_cols", type=str, default="", help="Comma-separated ID cols to exclude (e.g., 'match_id,event_id')")
    parser.add_argument("--drop_cols", type=str, default="", help="Comma-separated extra cols to drop")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation size fraction")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--do_search", action="store_true", help="Run a small hyperparameter grid search")
    parser.add_argument("--iterations", type=int, default=2000, help="Max trees (with early stopping)")
    parser.add_argument("--es_rounds", type=int, default=200, help="Early stopping rounds")
    parser.add_argument("--export_shap_full", action="store_true", help="Export per-sample SHAP for ALL features (validation set)")
    parser.add_argument("--export_shap_compact", action="store_true", help="Export per-sample SHAP for top-K features (validation set)")
    parser.add_argument("--compact_top_k", type=int, default=15, help="K for compact SHAP export")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    
    # Drop ghost index columns if any
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    assert args.xg_col in df.columns, f"{args.xg_col} not found"
    xg = df[args.xg_col].astype(float)
    exclude_cols = [args.xg_col]
    if args.id_cols:
        exclude_cols += [c.strip() for c in args.id_cols.split(",") if c.strip()]
    if args.drop_cols:
        exclude_cols += [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    exclude_cols = list(dict.fromkeys(exclude_cols))

    # Split first so threshold is learned only from TRAIN xG
    # We'll compute target after we split.
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    assert len(feature_cols) > 0, "No feature columns found after exclusions"

    # Stratified split using rough target (median) to avoid leakage; true target computed on train later
    rough = (xg > np.nanmedian(xg)).astype(int)
    idx_train, idx_val = train_test_split(
        np.arange(len(df)), test_size=args.val_size, random_state=args.random_state, stratify=rough
    )

    train_df = df.iloc[idx_train].reset_index(drop=True)
    val_df   = df.iloc[idx_val].reset_index(drop=True)

    # Compute P75 threshold on TRAIN only
    train_threshold = compute_threshold(train_df[args.xg_col], 0.75)

    # Build target on each split
    train_df["target"] = make_target(train_df[args.xg_col], train_threshold)
    val_df["target"]   = make_target(val_df[args.xg_col], train_threshold)

    # Sanity: if class collapsed, fallback to median
    if train_df["target"].nunique() < 2:
        train_threshold = compute_threshold(train_df[args.xg_col], 0.50)
        train_df["target"] = make_target(train_df[args.xg_col], train_threshold)
        val_df["target"]   = make_target(val_df[args.xg_col], train_threshold)
        print(f"[WARN] P75 collapsed the class. Fallback to median: {train_threshold:.6f}")

    # Prepare features
    X_train = train_df[feature_cols].copy()
    y_train = train_df["target"].astype(int).copy()
    X_val   = val_df[feature_cols].copy()
    y_val   = val_df["target"].astype(int).copy()

    # Detect categoricals
    cat_idx = []
    for i, c in enumerate(feature_cols):
        if pd.api.types.is_object_dtype(X_train[c]) or pd.api.types.is_categorical_dtype(X_train[c]):
            cat_idx.append(i)  # indices for Pool
        # Optional: convert boolean to int for stability
        if pd.api.types.is_bool_dtype(X_train[c]):
            X_train[c] = X_train[c].astype(int)
            X_val[c]   = X_val[c].astype(int)

    # Pools
    train_pool = Pool(X_train, y_train, cat_features=cat_idx if len(cat_idx) else None)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_idx if len(cat_idx) else None)

    # Baseline model
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=args.random_state,
        iterations=args.iterations,
        verbose=200,
        od_type="Iter",
        od_wait=args.es_rounds,
        auto_class_weights='SqrtBalanced', 
        thread_count=-1
    )

    # Optional lightweight search
    if args.do_search:
        param_grid = {
            "depth": [5, 6, 7, 8],
            "learning_rate": [0.03, 0.05, 0.08],
            "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0],
            "border_count": [128, 254],
            "bagging_temperature": [0.0, 0.5, 1.0],
        }
        print("[INFO] Running CatBoost native grid_search (this may take a while)...")
        model.grid_search(param_grid, X=train_pool, y=None, cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=args.random_state), shuffle=True, verbose=False)

    # Fit with validation
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Evaluate
    val_prob = model.predict_proba(val_pool)[:, 1]
    best_t, best_f1 = pick_best_threshold_by_f1(y_val.values, val_prob)
    metrics = evaluate(y_val.values, val_prob, best_t)

    print("\n=== HOLDOUT METRICS ===")
    print(json.dumps(metrics, indent=2))

    # Export metrics
    with open("metrics_holdout.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance (gain)

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.get_feature_importance(type="FeatureImportance")
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv("feature_importance_gain.csv", index=False)

    # Mean |SHAP| per feature
    shap_values_val = model.get_feature_importance(type="ShapValues", data=val_pool)
    shap_vals_only = shap_values_val[:, :-1]
    mean_abs_shap = np.mean(np.abs(shap_vals_only), axis=0)
    shap_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap_val": mean_abs_shap}).sort_values("mean_abs_shap_val", ascending=False)
    shap_df.to_csv("shap_mean_abs_val.csv", index=False)

    print("\\nSaved: metrics_holdout.json, feature_importance_gain.csv, shap_mean_abs_val.csv")
    print(f"Train P75 threshold on '{args.xg_col}': {train_threshold:.6f}")
    print(f"Train positive rate: {float(y_train.mean()):.4f}")

    if len(cat_idx):
        print(f"Detected {len(cat_idx)} categorical features (by index): {cat_idx}")
    else:
        print("No categorical features detected.")

    # Optional full/compact per-sample SHAP export
    if args.export_shap_full or args.export_shap_compact:
        expected_values = shap_values_val[:, -1]
        shap_vals = shap_values_val[:, :-1]
        val_out = val_df[[args.xg_col, "target"]].copy()
        val_out["pred_proba"] = val_prob
        shap_cols = {f"shap::{fname}": shap_vals[:, i] for i, fname in enumerate(feature_cols)}
        shap_df_full = pd.concat([val_out.reset_index(drop=True), pd.DataFrame(shap_cols)], axis=1)

        # RAW feature values
        raw_feat_df = X_val.reset_index(drop=True).copy()
        shap_full_with_features = pd.concat([shap_df_full, raw_feat_df], axis=1)

        if args.export_shap_full:
            shap_df_full.to_csv("shap_full_val.csv", index=False)
            shap_full_with_features.to_csv("shap_full_with_features.csv", index=False)
            print(f"Saved: shap_full_val.csv (shape={shap_df_full.shape})")
            print(f"Saved: shap_full_with_features.csv (shape={shap_full_with_features.shape})")

        if args.export_shap_compact:
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            ordering = np.argsort(-mean_abs_shap)
            topk = max(1, int(args.compact_top_k))
            top_idx = ordering[:topk]
            keep_cols = ["pred_proba", args.xg_col, "target"] + [f"shap::{feature_cols[i]}" for i in top_idx]
            shap_compact = shap_df_full[keep_cols].copy()
            shap_compact_path = "shap_compact_val.csv"
            shap_compact.to_csv(shap_compact_path, index=False)
            print(f"Saved: {shap_compact_path} (shape={shap_compact.shape})")

if __name__ == "__main__":
    main()