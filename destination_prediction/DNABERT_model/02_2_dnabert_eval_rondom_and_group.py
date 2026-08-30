import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# ==========================================
# Settings (DNABERT: evaluation with fixed parameters: GroupHost + GroupPlasmid only)
# ==========================================
HOST_FILE = "data/host_vectors.csv"
PLASMID_FILE = "data/trainplasmid_vectors.csv"
PAIRS_FILE = "data/pairs_converted.csv"

# Read best_params_final_{method}.json saved in 02 (DNABERT nestedCV random-only)
PARAM_DIR = "results/results_dnabert_nestedcv_randomsplit"
PARAM_FILE_PATTERN = "best_params_final_{method}.json"

OUTPUT_DIR = "results/dnabert_eval_fixedparams_group"
METHODS = ["Diff", "AbsDiff", "Prod"]

MODEL_SEED = 42
N_SPLITS = 5

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42

PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
# ==========================================


def load_data():
    if not (os.path.exists(HOST_FILE) and os.path.exists(PLASMID_FILE) and os.path.exists(PAIRS_FILE)):
        return None, None, None, None

    hosts = pd.read_csv(HOST_FILE)
    plasmids = pd.read_csv(PLASMID_FILE)
    pairs = pd.read_csv(PAIRS_FILE)

    df = pd.merge(
        pairs, hosts,
        left_on="host_file", right_on="filename",
        suffixes=("", "_h_garbage"),
        how="left"
    )
    df = pd.merge(
        df, plasmids,
        left_on="plasmid_file", right_on="filename",
        suffixes=("_x", "_y"),
        how="left"
    )

    cols_x = [f"feature_{i}_x" for i in range(768)]
    cols_y = [f"feature_{i}_y" for i in range(768)]

    h_vecs = df[cols_x].fillna(0).values.astype(float)
    p_vecs = df[cols_y].fillna(0).values.astype(float)
    y = df["label"].values

    return pairs, h_vecs, p_vecs, y


def generate_feature_sets(h_vecs, p_vecs):
    return {
        "Diff": h_vecs - p_vecs,
        "AbsDiff": np.abs(h_vecs - p_vecs),
        "Prod": h_vecs * p_vecs,
    }


def load_fixed_params_for_method(method: str):
    path = os.path.join(PARAM_DIR, PARAM_FILE_PATTERN.format(method=method))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parameter file not found: {path}")

    with open(path, "r") as f:
        params = json.load(f)

    if not isinstance(params, dict):
        raise ValueError(f"Invalid params format in {path}: expected dict, got {type(params)}")

    return params


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, seed=42, alpha=0.95):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))

    if len(aucs) == 0:
        return (np.nan, np.nan)

    lo = np.quantile(aucs, (1 - alpha) / 2)
    hi = np.quantile(aucs, 1 - (1 - alpha) / 2)
    return (float(lo), float(hi))


def save_classification_report_csv(y_true, y_pred, out_csv):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(out_csv)


def save_calibration_plot(y_true, y_prob, out_png, title, n_bins=10):
    if len(np.unique(y_true)) < 2:
        return np.nan

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)

    plt.figure(figsize=(7, 7))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Calibration")
    plt.plot([0, 1], [0, 1], "k:", alpha=0.6, label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"{title}\nBrier={brier:.4f}")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return float(brier)


def eval_group_cv(X, y, groups, params, n_splits=5):
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError(f"Not enough unique groups for GroupKFold: unique_groups={len(np.unique(groups))}")

    cv = GroupKFold(n_splits=n_splits)

    oof_probs = np.full(len(y), np.nan)
    oof_preds = np.full(len(y), np.nan)
    fold_aucs, fold_accs = [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        model = RandomForestClassifier(**params, random_state=MODEL_SEED, n_jobs=-1)
        model.fit(X[train_idx], y[train_idx])

        probs = model.predict_proba(X[test_idx])[:, 1]
        preds = model.predict(X[test_idx])

        oof_probs[test_idx] = probs
        oof_preds[test_idx] = preds

        fold_aucs.append(safe_auc(y[test_idx], probs))
        fold_accs.append(accuracy_score(y[test_idx], preds))

    valid = ~np.isnan(oof_probs)
    yv = y[valid]
    pv = oof_probs[valid]

    return {
        "oof_probs": oof_probs,
        "oof_preds": oof_preds,
        "fold_aucs": fold_aucs,
        "fold_accs": fold_accs,
        "mean_fold_auc": float(np.nanmean(fold_aucs)),
        "std_fold_auc": float(np.nanstd(fold_aucs, ddof=1)) if np.sum(~np.isnan(fold_aucs)) >= 2 else np.nan,
        "overall_oof_auc": float(safe_auc(yv, pv)) if len(yv) else np.nan,
        "mean_fold_acc": float(np.mean(fold_accs)),
        "std_fold_acc": float(np.std(fold_accs, ddof=1)) if len(fold_accs) >= 2 else np.nan,
        "n_splits_used": int(n_splits),
        "y_valid": yv,
        "p_valid": pv,
        "pred_valid": oof_preds[valid].astype(int),
    }


def save_merged_roc_plot(y_true, prob_dict, out_png, title):
    """Overlay ROC curves for the methods (Diff/AbsDiff/Prod) in a single figure"""
    if len(np.unique(y_true)) < 2:
        return

    plt.figure(figsize=(8, 8))
    for name, probs in prob_dict.items():
        if probs is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k:", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    pairs, h_vecs, p_vecs, y = load_data()
    if pairs is None:
        print("Input files not found. Check HOST_FILE/PLASMID_FILE/PAIRS_FILE.")
        return

    groups_host = pairs["host_file"].astype(str).values
    groups_plasmid = pairs["plasmid_file"].astype(str).values

    base = pairs[["host_file", "plasmid_file", "label"]].copy()
    base.rename(columns={"label": "True_Label"}, inplace=True)

    df_oof_host = base.copy()
    df_oof_plasmid = base.copy()

    summary_host = []
    summary_plasmid = []

    feature_sets = generate_feature_sets(h_vecs, p_vecs)

    for method in METHODS:
        print(f"\n>>> Evaluating method={method}")

        X = feature_sets[method]

        try:
            params = load_fixed_params_for_method(method)
        except (FileNotFoundError, ValueError) as e:
            print(f"[SKIP] {e}")
            continue

        fixed_param_path = os.path.join(PARAM_DIR, PARAM_FILE_PATTERN.format(method=method))

        # -------------------------
        # GroupHost
        # -------------------------
        try:
            res_h = eval_group_cv(X, y, groups_host, params, n_splits=N_SPLITS)
        except ValueError as e:
            print(f"[SKIP GroupHost] {method}: {e}")
            res_h = None

        if res_h is not None:
            df_oof_host[f"Prob_{method}"] = res_h["oof_probs"]
            df_oof_host[f"Pred_{method}"] = res_h["oof_preds"]

            rep_path = os.path.join(REPORTS_DIR, f"classification_report_group_host_{method}.csv")
            save_classification_report_csv(res_h["y_valid"], res_h["pred_valid"], rep_path)

            calib_png = os.path.join(PLOTS_DIR, f"calibration_group_host_{method}.png")
            brier_h = save_calibration_plot(res_h["y_valid"], res_h["p_valid"], calib_png, f"Calibration: GroupHost {method}")

            ci_lo_h, ci_hi_h = bootstrap_auc_ci(res_h["y_valid"], res_h["p_valid"], n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)

            summary_host.append({
                "method": method,
                "split": "GroupHost",
                "n_splits_used": res_h["n_splits_used"],
                "mean_fold_auc": res_h["mean_fold_auc"],
                "std_fold_auc": res_h["std_fold_auc"],
                "overall_oof_auc": res_h["overall_oof_auc"],
                "oof_auc_ci95_low": ci_lo_h,
                "oof_auc_ci95_high": ci_hi_h,
                "mean_fold_acc": res_h["mean_fold_acc"],
                "std_fold_acc": res_h["std_fold_acc"],
                "brier_score_oof": brier_h,
                "fixed_params_file": fixed_param_path,
            })
            print(f"[GroupHost] {method}: overallOOF_AUC={res_h['overall_oof_auc']:.3f}")

        # -------------------------
        # GroupPlasmid
        # -------------------------
        try:
            res_p = eval_group_cv(X, y, groups_plasmid, params, n_splits=N_SPLITS)
        except ValueError as e:
            print(f"[SKIP GroupPlasmid] {method}: {e}")
            res_p = None

        if res_p is not None:
            df_oof_plasmid[f"Prob_{method}"] = res_p["oof_probs"]
            df_oof_plasmid[f"Pred_{method}"] = res_p["oof_preds"]

            rep_path = os.path.join(REPORTS_DIR, f"classification_report_group_plasmid_{method}.csv")
            save_classification_report_csv(res_p["y_valid"], res_p["pred_valid"], rep_path)

            calib_png = os.path.join(PLOTS_DIR, f"calibration_group_plasmid_{method}.png")
            brier_p = save_calibration_plot(res_p["y_valid"], res_p["p_valid"], calib_png, f"Calibration: GroupPlasmid {method}")

            ci_lo_p, ci_hi_p = bootstrap_auc_ci(res_p["y_valid"], res_p["p_valid"], n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)

            summary_plasmid.append({
                "method": method,
                "split": "GroupPlasmid",
                "n_splits_used": res_p["n_splits_used"],
                "mean_fold_auc": res_p["mean_fold_auc"],
                "std_fold_auc": res_p["std_fold_auc"],
                "overall_oof_auc": res_p["overall_oof_auc"],
                "oof_auc_ci95_low": ci_lo_p,
                "oof_auc_ci95_high": ci_hi_p,
                "mean_fold_acc": res_p["mean_fold_acc"],
                "std_fold_acc": res_p["std_fold_acc"],
                "brier_score_oof": brier_p,
                "fixed_params_file": fixed_param_path,
            })
            print(f"[GroupPlasmid] {method}: overallOOF_AUC={res_p['overall_oof_auc']:.3f}")

    # Save (GroupHost/GroupPlasmid only)
    df_oof_host.to_csv(os.path.join(OUTPUT_DIR, "oof_group_host.csv"), index=False)
    pd.DataFrame(summary_host).to_csv(os.path.join(OUTPUT_DIR, "summary_group_host.csv"), index=False)

    df_oof_plasmid.to_csv(os.path.join(OUTPUT_DIR, "oof_group_plasmid.csv"), index=False)
    pd.DataFrame(summary_plasmid).to_csv(os.path.join(OUTPUT_DIR, "summary_group_plasmid.csv"), index=False)

    # merged ROC (overlay the three methods: one figure per split)
    for split_name, oof_path in [
        ("GroupHost", os.path.join(OUTPUT_DIR, "oof_group_host.csv")),
        ("GroupPlasmid", os.path.join(OUTPUT_DIR, "oof_group_plasmid.csv")),
    ]:
        df = pd.read_csv(oof_path)
        label_col = "True_Label" if "True_Label" in df.columns else "label"

        prob_cols = [f"Prob_{m}" for m in METHODS if f"Prob_{m}" in df.columns]
        if not prob_cols:
            continue

        df_eval = df.dropna(subset=[label_col] + prob_cols).copy()
        if len(df_eval) == 0:
            continue

        y_eval = df_eval[label_col].values
        prob_dict = {m: (df_eval[f"Prob_{m}"].values if f"Prob_{m}" in df_eval.columns else None) for m in METHODS}

        out_png = os.path.join(PLOTS_DIR, f"merged_roc_{split_name}.png")
        save_merged_roc_plot(
            y_true=y_eval,
            prob_dict=prob_dict,
            out_png=out_png,
            title=f"ROC Curves ({'/'.join(METHODS)}) - {split_name} (OOF)"
        )
        print(f"✅ Saved merged ROC: {out_png}")

    print("\n✅ Saved evaluation results (Group-only).")
    print(f"  - {os.path.join(OUTPUT_DIR, 'oof_group_host.csv')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'oof_group_plasmid.csv')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'summary_group_host.csv')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'summary_group_plasmid.csv')}")
    print(f"  - {PLOTS_DIR}/merged_roc_*.png")
    print(f"  - {PLOTS_DIR}/calibration_*.png")
    print(f"  - {REPORTS_DIR}/classification_report_*.csv")


if __name__ == "__main__":
    main()
