import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# ==========================================
# Settings (random split)
# ==========================================
KMER_DIR = "results/kmer/kmer_features"
PAIRS_FILE = "data/pairs_converted.csv"
OUTPUT_DIR = "results/kmer/results_nestedcv_randomsplit_cano"
K_LIST = [2, 3, 4, 5, 6, 7]

RANDOM_SEED = 42
MODEL_SEED = 42

PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5

MISSING_WARN_THRESHOLD = 0.01  # 1%

# Added: uncertainty (bootstrap CI)
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42

# Added: output subfolders
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots_outer_oof")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports_outer_oof")
# ==========================================


def load_data(k, pairs, verbose=True):
    h_file = os.path.join(KMER_DIR, f"host_{k}mer_canonical.csv")
    p_file = os.path.join(KMER_DIR, f"plasmid_{k}mer_canonical.csv")
    if not os.path.exists(h_file) or not os.path.exists(p_file):
        return None, None, None

    hosts = pd.read_csv(h_file)
    plasmids = pd.read_csv(p_file)

    df = pd.merge(
        pairs, hosts, left_on="host_file", right_on="filename",
        how="left", suffixes=("", "_h")
    )
    host_missing_rate = df["filename"].isna().mean()

    df = pd.merge(
        df, plasmids, left_on="plasmid_file", right_on="filename",
        how="left", suffixes=("_x", "_y")
    )
    plasmid_filename_col = "filename_y" if "filename_y" in df.columns else "filename"
    plasmid_missing_rate = df[plasmid_filename_col].isna().mean()

    if verbose:
        if host_missing_rate > MISSING_WARN_THRESHOLD:
            print(f"   [WARN] k={k}: host kmer not found for {host_missing_rate*100:.2f}% of pairs")
        if plasmid_missing_rate > MISSING_WARN_THRESHOLD:
            print(f"   [WARN] k={k}: plasmid kmer not found for {plasmid_missing_rate*100:.2f}% of pairs")

    cols_x = [c for c in df.columns if f"{k}mer_" in c and c.endswith("_x")]
    cols_y = [c for c in df.columns if f"{k}mer_" in c and c.endswith("_y")]

    if len(cols_x) == 0 or len(cols_y) == 0:
        print(f"   [ERROR] k={k}: feature columns not found. cols_x={len(cols_x)}, cols_y={len(cols_y)}")
        return None, None, None

    X = df[cols_x].fillna(0).values - df[cols_y].fillna(0).values
    y = df["label"].values

    missing_info = {
        "host_missing_rate": float(host_missing_rate),
        "plasmid_missing_rate": float(plasmid_missing_rate),
    }
    return X, y, missing_info


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, seed=42, alpha=0.95):
    """Bootstrap CI for AUC on OOF predictions (sampling with replacement)"""
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
    """Reliability diagram + Brier score"""
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    pairs = pd.read_csv(PAIRS_FILE)

    df_oof = pairs[["host_file", "plasmid_file", "label"]].copy()
    summary_list = []
    missing_record = {}
    best_params_outer_record = {}

    outer_cv = StratifiedKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    for k in K_LIST:
        print(f"\n>>> Processing k={k} (Nested CV: random/stratified) ...")
        X, y, missing_info = load_data(k, pairs, verbose=True)
        if X is None:
            continue

        missing_record[f"k{k}"] = missing_info

        oof_probs = np.full(len(y), np.nan)
        oof_preds = np.full(len(y), np.nan)

        fold_aucs, fold_accs = [], []
        fold_best_params = []

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            inner_cv = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_SEED)

            rf = RandomForestClassifier(random_state=MODEL_SEED, n_jobs=-1)
            grid = GridSearchCV(
                estimator=rf,
                param_grid=PARAM_GRID,
                cv=inner_cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True
            )
            grid.fit(X_train, y_train)

            fold_best_params.append(grid.best_params_)

            probs = grid.predict_proba(X_test)[:, 1]
            preds = grid.predict(X_test)  # equivalent to threshold=0.5

            oof_probs[test_idx] = probs
            oof_preds[test_idx] = preds

            fold_auc = safe_auc(y_test, probs)
            fold_acc = accuracy_score(y_test, preds)

            fold_aucs.append(fold_auc)
            fold_accs.append(fold_acc)

            print(
                f"   Fold {fold}: best_inner_CV_AUC={grid.best_score_:.3f}, "
                f"outer_AUC={fold_auc if not np.isnan(fold_auc) else 'NA'}, "
                f"outer_ACC={fold_acc:.3f}"
            )

        valid = ~np.isnan(oof_probs)
        y_valid = y[valid]
        prob_valid = oof_probs[valid]
        pred_valid = oof_preds[valid].astype(int)

        overall_oof_auc = safe_auc(y_valid, prob_valid)
        mean_fold_auc = float(np.nanmean(fold_aucs))
        std_fold_auc = float(np.nanstd(fold_aucs, ddof=1)) if np.sum(~np.isnan(fold_aucs)) >= 2 else np.nan

        mean_fold_acc = float(np.mean(fold_accs))
        std_fold_acc = float(np.std(fold_accs, ddof=1)) if len(fold_accs) >= 2 else np.nan

        ci_lo, ci_hi = bootstrap_auc_ci(
            y_valid, prob_valid, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, alpha=0.95
        )

        print(f"   -> Mean Outer-Fold AUC: {mean_fold_auc:.3f} (std={std_fold_auc:.3f})")
        print(f"   -> Overall OOF AUC: {overall_oof_auc:.3f} (95% CI: {ci_lo:.3f}-{ci_hi:.3f})")

        # Save OOF
        df_oof[f"Prob_k{k}"] = oof_probs
        df_oof[f"Pred_k{k}"] = oof_preds
        best_params_outer_record[f"k{k}"] = fold_best_params

        # Added: ROC / report / calibration based on outer-CV OOF
        roc_png = os.path.join(PLOTS_DIR, f"outer_oof_roc_k{k}.png")
        save_roc_curve(y_valid, prob_valid, roc_png, title=f"Nested CV (Outer OOF) ROC: k={k}")

        rep_csv = os.path.join(REPORTS_DIR, f"outer_oof_classification_report_k{k}.csv")
        save_classification_report_csv(y_valid, pred_valid, rep_csv)

        calib_png = os.path.join(PLOTS_DIR, f"outer_oof_calibration_k{k}.png")
        brier = save_calibration_plot(y_valid, prob_valid, calib_png, title=f"Calibration (Outer OOF): k={k}")

        summary_list.append({
            "k": k,
            "mean_outer_fold_auc": mean_fold_auc,
            "std_outer_fold_auc": std_fold_auc,
            "overall_oof_auc": float(overall_oof_auc) if not np.isnan(overall_oof_auc) else np.nan,
            "oof_auc_ci95_low": ci_lo,
            "oof_auc_ci95_high": ci_hi,
            "mean_outer_fold_acc": mean_fold_acc,
            "std_outer_fold_acc": std_fold_acc,
            "brier_score_oof": brier,
            "host_missing_rate": missing_info["host_missing_rate"],
            "plasmid_missing_rate": missing_info["plasmid_missing_rate"],
        })

        # =========================================================
        # Final model (for deployment): retrain on FULL data for each k -> save
        # =========================================================
        print(f"   Training final model on FULL data for k={k} ...")

        final_cv = StratifiedKFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED
        )

        rf_final = RandomForestClassifier(
            random_state=MODEL_SEED,
            n_jobs=-1
        )

        final_grid = GridSearchCV(
            estimator=rf_final,
            param_grid=PARAM_GRID,
            cv=final_cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True
        )

        final_grid.fit(X, y)

        joblib.dump(
            final_grid.best_estimator_,
            os.path.join(OUTPUT_DIR, f"model_k{k}.pkl")
        )

        with open(
            os.path.join(OUTPUT_DIR, f"best_params_final_k{k}.json"),
            "w"
        ) as f:
            json.dump(final_grid.best_params_, f, indent=4)


    # =========================================================
    # Added: save the Outer-OOF ROC curves for k=2..7 overlaid in a single figure
    #  - To compare OOF probabilities (Prob_k{k}) for each k on the same population,
    #    plot only rows without NaN for all k (common mask).
    #  - This can be pasted near the end of script 02 (immediately before or after the save step).
    # =========================================================
    try:
        prob_cols = [f"Prob_k{k}" for k in K_LIST]

        # Common mask (rows where Prob is available for all k)
        df_eval_allk = df_oof.dropna(subset=["label"] + prob_cols).copy()
        if len(df_eval_allk) == 0:
            print("[WARN] Overlay ROC skipped: no rows with complete OOF probabilities for all k.")
        else:
            y_allk = df_eval_allk["label"].values.astype(int)

            if len(np.unique(y_allk)) < 2:
                print("[WARN] Overlay ROC skipped: y contains only one class in the common subset.")
            else:
                plt.figure(figsize=(8, 8))

                for k in K_LIST:
                    probs_k = df_eval_allk[f"Prob_k{k}"].values.astype(float)
                    fpr, tpr, _ = roc_curve(y_allk, probs_k)
                    auc_k = roc_auc_score(y_allk, probs_k)
                    plt.plot(fpr, tpr, label=f"k={k} (AUC={auc_k:.3f})", linewidth=2)

                plt.plot([0, 1], [0, 1], "k:", alpha=0.6)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Nested CV (Outer OOF) ROC: k=2–7 (common subset)")
                plt.legend(loc="lower right")
                plt.grid(alpha=0.3)
                plt.tight_layout()

                out_png = os.path.join(PLOTS_DIR, "outer_oof_roc_overlay_k2to7.png")
                plt.savefig(out_png)
                plt.close()

                print(f"✅ Saved overlay ROC: {out_png} (n={len(df_eval_allk)})")

    except Exception as e:
        print(f"[WARN] Overlay ROC block failed: {e}")

    df_oof.to_csv(os.path.join(OUTPUT_DIR, "cv_predictions_oof.csv"), index=False)
    pd.DataFrame(summary_list).to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "best_params_outer_by_fold.json"), "w") as f:
        json.dump(best_params_outer_record, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "missing_rates.json"), "w") as f:
        json.dump(missing_record, f, indent=4)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
