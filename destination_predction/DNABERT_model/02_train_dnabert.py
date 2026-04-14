import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve, classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

# =================================================
# Settings area (aligned with k-mer 02)
# =================================================
HOST_FILE = "data/host_vectors.csv"
PLASMID_FILE = "data/trainplasmid_vectors.csv"
PAIRS_FILE = "data/pairs_converted.csv"

OUTPUT_DIR = "results/results_dnabert_nestedcv_randomsplit"  # Equivalent to OUTPUT_DIR in k-mer 02

METHODS = ["Diff", "AbsDiff", "Prod"]

RANDOM_SEED = 42
MODEL_SEED = 42

SCORING = "roc_auc"

PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5

# Missing rate check (equivalent to missing_rates.json in the k-mer version)
MISSING_WARN_THRESHOLD = 0.01  # 1%

# Uncertainty (bootstrap CI)
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
CI_ALPHA = 0.95  # 95% CI

# Output subfolders
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots_outer_oof")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports_outer_oof")
# =================================================


def load_data():
    """
    Merge the DNABERT training CSV files (host_vectors/trainplasmid_vectors)
    with pairs to obtain 768-dimensional host/plasmid vectors
    """
    if not (os.path.exists(HOST_FILE) and os.path.exists(PLASMID_FILE) and os.path.exists(PAIRS_FILE)):
        return None, None, None, None, None

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

    # Missing rate (record NaN proportion as an alternative to k-mer missing_rates)
    host_missing_rate = df[cols_x].isna().any(axis=1).mean()
    plasmid_missing_rate = df[cols_y].isna().any(axis=1).mean()

    if host_missing_rate > MISSING_WARN_THRESHOLD:
        print(f"[WARN] host vector missing in {host_missing_rate*100:.2f}% of pairs")
    if plasmid_missing_rate > MISSING_WARN_THRESHOLD:
        print(f"[WARN] plasmid vector missing in {plasmid_missing_rate*100:.2f}% of pairs")

    h_vecs = df[cols_x].fillna(0).values.astype(float)
    p_vecs = df[cols_y].fillna(0).values.astype(float)
    y = df["label"].values

    missing_info = {
        "host_missing_rate": float(host_missing_rate),
        "plasmid_missing_rate": float(plasmid_missing_rate),
    }

    return h_vecs, p_vecs, y, df, missing_info


def generate_feature_sets(h_vecs, p_vecs):
    return {
        "Diff": h_vecs - p_vecs,
        "AbsDiff": np.abs(h_vecs - p_vecs),
        "Prod": h_vecs * p_vecs,
    }


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, seed=42, alpha=0.95):
    """
    Bootstrap CI for AUC on OOF predictions (sampling with replacement)
    """
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


def save_roc_curve(y_true, y_prob, out_png, title):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k:", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_calibration_plot(y_true, y_prob, out_png, title, n_bins=10):
    """
    Reliability diagram + Brier score
    """
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


def save_classification_report_csv(y_true, y_pred, out_csv):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(out_csv)


def save_overlay_roc_methods(y_true, probs_by_method, out_png, title):
    """
    Save a single ROC figure with the three methods (Diff/AbsDiff/Prod) overlaid
    probs_by_method: dict[method] -> 1D array(prob)
    """
    if len(np.unique(y_true)) < 2:
        return

    plt.figure(figsize=(8, 8))

    for method, prob in probs_by_method.items():
        if prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        plt.plot(fpr, tpr, label=f"{method} (AUC={auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k:", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def nested_cv_oof_random_only(X, y, param_grid, scoring, n_outer, n_inner):
    """
    Outer loop: StratifiedKFold (for OOF estimation)
    Inner loop: StratifiedKFold + GridSearch (maximize AUC)
    """
    oof_probs = np.full(len(y), np.nan)
    oof_preds = np.full(len(y), np.nan)

    outer_fold_aucs = []
    outer_fold_accs = []
    outer_best_params = []

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=RANDOM_SEED)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=RANDOM_SEED)

        rf = RandomForestClassifier(random_state=MODEL_SEED, n_jobs=-1)
        grid = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True
        )
        grid.fit(X_train, y_train)

        outer_best_params.append(grid.best_params_)

        probs = grid.predict_proba(X_test)[:, 1]
        preds = grid.predict(X_test)  # equivalent to threshold=0.5

        oof_probs[test_idx] = probs
        oof_preds[test_idx] = preds

        fold_auc = safe_auc(y_test, probs)
        fold_acc = accuracy_score(y_test, preds)

        outer_fold_aucs.append(fold_auc)
        outer_fold_accs.append(fold_acc)

        print(
            f"   Fold {fold}: best_inner_CV_{scoring}={grid.best_score_:.3f}, "
            f"outer_AUC={fold_auc if not np.isnan(fold_auc) else 'NA'}, "
            f"outer_ACC={fold_acc:.3f}"
        )

    valid = ~np.isnan(oof_probs)
    overall_oof_auc = safe_auc(y[valid], oof_probs[valid])
    mean_outer_auc = float(np.nanmean(outer_fold_aucs))

    return {
        "oof_probs": oof_probs,
        "oof_preds": oof_preds,
        "mean_outer_auc": mean_outer_auc,
        "overall_oof_auc": float(overall_oof_auc) if not np.isnan(overall_oof_auc) else np.nan,
        "outer_fold_aucs": outer_fold_aucs,
        "mean_outer_acc": float(np.mean(outer_fold_accs)),
        "outer_fold_accs": outer_fold_accs,
        "outer_best_params": outer_best_params,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    h_vecs, p_vecs, y_all, df_meta, missing_info = load_data()
    if h_vecs is None:
        print("Input files not found. Check HOST_FILE/PLASMID_FILE/PAIRS_FILE.")
        return

    feature_sets = generate_feature_sets(h_vecs, p_vecs)

    # Output structure matching k-mer 02
    df_oof = df_meta[["host_file", "plasmid_file", "label"]].copy()
    df_oof.rename(columns={"label": "True_Label"}, inplace=True)

    summary_rows = []
    best_params_outer_by_fold = {}
    missing_rates = {}

    # Added: collect OOF probabilities for overlay ROC by method
    oof_probs_for_overlay = {}

    print("\nStarting Nested CV Evaluation (DNABERT, Random-only)...")
    print("=" * 60)

    for method_name in METHODS:
        print(f"\n>>> Method: {method_name}")
        X = feature_sets[method_name]

        # nested CV (random only)
        print("   [Random] Nested CV...")
        res = nested_cv_oof_random_only(
            X=X,
            y=y_all,
            param_grid=PARAM_GRID,
            scoring=SCORING,
            n_outer=N_OUTER_SPLITS,
            n_inner=N_INNER_SPLITS,
        )

        # Save OOF
        df_oof[f"Prob_{method_name}"] = res["oof_probs"]
        df_oof[f"Pred_{method_name}"] = res["oof_preds"]

        # For overlay ROC
        oof_probs_for_overlay[method_name] = res["oof_probs"].copy()

        # outer fold params
        best_params_outer_by_fold[method_name] = res["outer_best_params"]

        # missing rate
        missing_rates[method_name] = missing_info

        # ROC / report / calibration / CI using outer OOF
        valid = ~np.isnan(res["oof_probs"])
        y_valid = y_all[valid]
        prob_valid = res["oof_probs"][valid]
        pred_valid = res["oof_preds"][valid].astype(int)

        save_roc_curve(
            y_valid, prob_valid,
            out_png=os.path.join(PLOTS_DIR, f"outer_oof_roc_{method_name}.png"),
            title=f"Nested CV (Outer OOF) ROC: {method_name}"
        )

        save_classification_report_csv(
            y_valid, pred_valid,
            out_csv=os.path.join(REPORTS_DIR, f"outer_oof_classification_report_{method_name}.csv")
        )

        calib_png = os.path.join(PLOTS_DIR, f"outer_oof_calibration_{method_name}.png")
        brier = save_calibration_plot(
            y_valid, prob_valid,
            out_png=calib_png,
            title=f"Calibration (Outer OOF): {method_name}",
            n_bins=10
        )

        ci_lo, ci_hi = bootstrap_auc_ci(
            y_valid, prob_valid,
            n_boot=BOOTSTRAP_N,
            seed=BOOTSTRAP_SEED,
            alpha=CI_ALPHA
        )

        fold_aucs = np.asarray(res["outer_fold_aucs"], dtype=float)
        std_fold_auc = float(np.nanstd(fold_aucs, ddof=1)) if np.sum(~np.isnan(fold_aucs)) >= 2 else np.nan

        fold_accs = np.asarray(res["outer_fold_accs"], dtype=float)
        std_fold_acc = float(np.nanstd(fold_accs, ddof=1)) if len(fold_accs) >= 2 else np.nan

        summary_rows.append({
            "method": method_name,
            "mean_outer_fold_auc": res["mean_outer_auc"],
            "std_outer_fold_auc": std_fold_auc,
            "overall_oof_auc": res["overall_oof_auc"],
            "oof_auc_ci95_low": ci_lo,
            "oof_auc_ci95_high": ci_hi,
            "mean_outer_fold_acc": res["mean_outer_acc"],
            "std_outer_fold_acc": std_fold_acc,
            "brier_score_oof": brier,
            "host_missing_rate": missing_info["host_missing_rate"],
            "plasmid_missing_rate": missing_info["plasmid_missing_rate"],
        })

        print(
            f"   -> Mean Outer AUC={res['mean_outer_auc']:.3f} (std={std_fold_auc:.3f}), "
            f"OOF AUC={res['overall_oof_auc']:.3f} (95% CI: {ci_lo:.3f}-{ci_hi:.3f}), "
            f"Brier={brier if not np.isnan(brier) else 'NA'}"
        )

        # For deployment: GridSearch on full data -> save
        print("   Training final model on FULL data...")
        final_cv = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        rf_final = RandomForestClassifier(random_state=MODEL_SEED, n_jobs=-1)
        final_grid = GridSearchCV(
            estimator=rf_final,
            param_grid=PARAM_GRID,
            cv=final_cv,
            scoring=SCORING,
            n_jobs=-1,
            refit=True
        )
        final_grid.fit(X, y_all)

        joblib.dump(final_grid.best_estimator_, os.path.join(OUTPUT_DIR, f"model_{method_name}.pkl"))
        with open(os.path.join(OUTPUT_DIR, f"best_params_final_{method_name}.json"), "w") as f:
            json.dump(final_grid.best_params_, f, indent=4)

    print("=" * 60)

    # ===== ROC with the three methods overlaid in one figure (based on outer OOF) =====
    # Restrict to rows with no NaN for all three methods and with at least two label classes
    try:
        cols = [f"Prob_{m}" for m in METHODS]
        df_tmp = df_oof[["True_Label"] + cols].copy()
        df_tmp = df_tmp.dropna(subset=["True_Label"] + cols)
        y_overlay = df_tmp["True_Label"].values

        probs_overlay = {}
        for m in METHODS:
            probs_overlay[m] = df_tmp[f"Prob_{m}"].values

        overlay_png = os.path.join(PLOTS_DIR, "outer_oof_roc_overlay_methods.png")
        save_overlay_roc_methods(
            y_true=y_overlay,
            probs_by_method=probs_overlay,
            out_png=overlay_png,
            title="Nested CV (Outer OOF) ROC: Diff vs AbsDiff vs Prod"
        )
        print(f"✅ Saved overlay ROC: {overlay_png}")
    except Exception as e:
        print(f"[WARN] Failed to create overlay ROC: {e}")

    # ===== Save using the same filenames as k-mer 02 =====
    df_oof.to_csv(os.path.join(OUTPUT_DIR, "cv_predictions_oof.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "best_params_outer_by_fold.json"), "w") as f:
        json.dump(best_params_outer_by_fold, f, indent=4)

    with open(os.path.join(OUTPUT_DIR, "missing_rates.json"), "w") as f:
        json.dump(missing_rates, f, indent=4)

    print("✅ Saved:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'cv_predictions_oof.csv')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'summary.csv')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'best_params_outer_by_fold.json')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'missing_rates.json')}")
    print(f"  - {PLOTS_DIR}/outer_oof_roc_*.png")
    print(f"  - {PLOTS_DIR}/outer_oof_calibration_*.png")
    print(f"  - {PLOTS_DIR}/outer_oof_roc_overlay_methods.png")
    print(f"  - {REPORTS_DIR}/outer_oof_classification_report_*.csv")
    print(f"  - {OUTPUT_DIR}/model_*.pkl")
    print(f"  - {OUTPUT_DIR}/best_params_final_*.json")


if __name__ == "__main__":
    main()
