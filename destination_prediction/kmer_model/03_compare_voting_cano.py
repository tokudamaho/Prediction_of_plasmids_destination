import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, classification_report,
    f1_score, matthews_corrcoef, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# =================================================
# Settings area
# =================================================
INPUT_FILE = "results/kmer/results_nestedcv_randomsplit_cano/cv_predictions_oof.csv"
#results_nestedcv_randomsplit/cv_predictions_oof.csv//eval_fixedparams_group/oof_group_plasmid//eval_fixedparams_group/oof_group_host
OUTPUT_DIR = "results/kmer/results_evaluation_cano"

K_LIST = [2, 3, 4, 5, 6, 7]
SINGLE_THRESH = 0.5  # Fixed binarization threshold for each model (used to create votes)

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
# =================================================


def get_prefix_from_input_path(input_path: str) -> str:
    base = os.path.basename(input_path)
    prefix = os.path.splitext(base)[0]
    return prefix


def get_label_column(df: pd.DataFrame) -> str:
    if "True_Label" in df.columns:
        return "True_Label"
    if "label" in df.columns:
        return "label"
    raise KeyError("Label column not found. Expected 'True_Label' or 'label'.")


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


def get_metrics(y_true, y_pred, y_prob=None):
    res = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    if y_prob is not None:
        res["AUC"] = safe_auc(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob) if len(np.unique(y_true)) >= 2 else np.nan
        res["Brier"] = float(brier) if not np.isnan(brier) else np.nan
        lo, hi = bootstrap_auc_ci(y_true, y_prob, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        res["AUC_CI95_low"] = lo
        res["AUC_CI95_high"] = hi
    else:
        res["AUC"] = np.nan
        res["Brier"] = np.nan
        res["AUC_CI95_low"] = np.nan
        res["AUC_CI95_high"] = np.nan

    return res


def save_classification_report(y_true, y_pred, filename):
    report_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    df_rep = pd.DataFrame(report_dict).transpose()
    df_rep.to_csv(filename)


def save_calibration_plot(y_true, y_score, out_png, title, n_bins=10):
    if len(np.unique(y_true)) < 2:
        return np.nan

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_score)

    plt.figure(figsize=(7, 7))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Calibration")
    plt.plot([0, 1], [0, 1], "k:", alpha=0.6, label="Perfectly calibrated")
    plt.xlabel("Mean predicted score")
    plt.ylabel("Fraction of positives")
    plt.title(f"{title}\nBrier={brier:.4f}")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return float(brier)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prefix = get_prefix_from_input_path(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)
    label_col = get_label_column(df)
    prob_cols = [f"Prob_k{k}" for k in K_LIST]
    missing_cols = [c for c in prob_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing probability columns: {missing_cols}")

    df_eval = df.dropna(subset=[label_col] + prob_cols).copy()
    if len(df_eval) != len(df):
        print(f"[WARN] Dropped {len(df) - len(df_eval)} rows due to NaN in label/prob columns.")

    y_true = df_eval[label_col].values

    # Votes: fixed 0.5 threshold within each model
    vote_matrix = pd.DataFrame(index=df_eval.index)
    for k in K_LIST:
        vote_matrix[f'k{k}'] = (df_eval[f'Prob_k{k}'] > SINGLE_THRESH).astype(int)

    vote_counts = vote_matrix.sum(axis=1).values
    vote_score = vote_counts / len(K_LIST)  # Continuous score for ROC (vote fraction)

    results = []

    print("\n--- Evaluation Started ---")
    print(f"INPUT : {INPUT_FILE}")
    print(f"OUTPUT: {OUTPUT_DIR}/ (prefix={prefix})")

    # ---------------------------------------------------------
    # 1. Single model evaluation (k=2~7)
    # ---------------------------------------------------------
    for k in K_LIST:
        pred = vote_matrix[f'k{k}'].values  # Binarized with fixed threshold 0.5
        prob = df_eval[f'Prob_k{k}'].values  # Continuous score

        met = get_metrics(y_true, pred, prob)
        met['Model'] = f"{k}-mer"
        met['Type'] = "Single"
        met['Threshold_Votes'] = np.nan
        results.append(met)

        save_path = os.path.join(OUTPUT_DIR, f"{prefix}_report_single_k{k}.csv")
        save_classification_report(y_true, pred, save_path)

    # ---------------------------------------------------------
    # 2. Ensemble evaluation (Vote >= t)
    #    AUC is calculated using vote_score (vote fraction)
    #    (it is the same for all t, but kept by design)
    # ---------------------------------------------------------
    for t in range(1, len(K_LIST) + 1):
        pred = (vote_counts >= t).astype(int)

        met = get_metrics(y_true, pred, vote_score)
        met['Model'] = f"Vote >={t}"
        met['Type'] = "Ensemble"
        met['Threshold_Votes'] = t
        results.append(met)

        save_path = os.path.join(OUTPUT_DIR, f"{prefix}_report_ensemble_vote_ge_{t}.csv")
        save_classification_report(y_true, pred, save_path)

    # ---------------------------------------------------------
    # 3. Save summary
    # ---------------------------------------------------------
    res_df = pd.DataFrame(results)
    cols = [
        'Model', 'Type', 'Threshold_Votes',
        'AUC', 'AUC_CI95_low', 'AUC_CI95_high',
        'Accuracy', 'F1', 'Precision', 'Recall', 'MCC',
        'Brier'
    ]
    res_df = res_df[cols]

    out_csv = os.path.join(OUTPUT_DIR, f"{prefix}_comparison_summary_detailed.csv")
    res_df.to_csv(out_csv, index=False)

    print(f"\n✅ Summary saved to: {out_csv}")

    # ---------------------------------------------------------
    # 4. Plot ROC curve: only one hard voting curve (vote_score)
    #    + annotate thresholds t=1..6 as points on the ROC
    # ---------------------------------------------------------
    if len(np.unique(y_true)) >= 2:
        plt.figure(figsize=(10, 8))

        # (A) ROC using vote_score as a continuous score
        fpr_v, tpr_v, thr_v = roc_curve(y_true, vote_score)
        auc_v = roc_auc_score(y_true, vote_score)
        plt.plot(fpr_v, tpr_v, label=f'Hard voting score (AUC={auc_v:.3f})', linewidth=3)

        # (B) Plot classification points for t=1..6 on the ROC with labels
        #     corresponding to pred = (vote_counts >= t)
        for t in range(1, len(K_LIST) + 1):
            pred_t = (vote_counts >= t).astype(int)

            # Confusion matrix components (with care for zero denominators)
            tp = np.sum((pred_t == 1) & (y_true == 1))
            fp = np.sum((pred_t == 1) & (y_true == 0))
            tn = np.sum((pred_t == 0) & (y_true == 0))
            fn = np.sum((pred_t == 0) & (y_true == 1))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # sensitivity
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan  # 1-specificity

            if np.isnan(tpr) or np.isnan(fpr):
                continue

            plt.scatter([fpr], [tpr], s=60)
            # Display t near the point (slightly shifted)
            plt.text(fpr + 0.01, tpr + 0.01, f"t={t}", fontsize=10)

        plt.plot([0, 1], [0, 1], 'k:', alpha=0.6)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: Hard voting only ({prefix})')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        roc_path = os.path.join(OUTPUT_DIR, f"{prefix}_roc_curve_hard_voting_only.png")
        plt.savefig(roc_path)
        plt.close()
        print(f"✅ ROC Plot saved to: {roc_path}")
    else:
        print("[WARN] ROC skipped: y_true contains only one class.")


if __name__ == "__main__":
    main()
