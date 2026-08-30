import pandas as pd
import numpy as np
import os
import joblib
import itertools
import argparse
from Bio import SeqIO
from tqdm import tqdm

# =================================================
# ★ Settings area
# =================================================
MODEL_DIR = "results/kmer/results_nestedcv_randomsplit_cano"
HOST_KMER_DIR = "results/kmer/kmer_features"
OUTPUT_ROOT = "results/kmer/predictions_output_cano"

K_LIST = [2, 3, 4, 5, 6, 7]

# Voting criterion for single models
SINGLE_THRESH = 0.5

# Backward compatibility: representative threshold (Prediction column)
VOTE_THRESH = 3
# =================================================


COMP = str.maketrans("ACGT", "TGCA")


def get_reverse_complement(seq: str) -> str:
    return seq.translate(COMP)[::-1]


def all_kmers_order(k: int):
    """Return all 4^k k-mers in lexicographic order (the order of itertools.product)."""
    bases = ["A", "C", "G", "T"]
    return ["".join(p) for p in itertools.product(bases, repeat=k)]


def canonical_kmer(kmer: str) -> str:
    """Use the lexicographically smaller one as the canonical k-mer."""
    rc = get_reverse_complement(kmer)
    return min(kmer, rc)


def canonical_kmers_order(k: int):
    """
    Determine the order of canonical representative k-mers
    (important: fixed to match the column order in the host CSV).

    Policy:
    Scan all 4^k k-mers in lexicographic order, convert each to canonical form,
    and keep unique values in order of first appearance.
    """
    kmers = all_kmers_order(k)
    seen = set()
    out = []
    for km in kmers:
        can = canonical_kmer(km)
        if can not in seen:
            seen.add(can)
            out.append(can)
    return out


def count_nucleotides_canonical_like_csv(sequence: str, k: int) -> np.ndarray:
    """
    ★ Important ★
    Assuming host_{k}mer_canonical.csv was created by:
    "counting both strands -> normalization -> removing duplicated columns",
    construct the canonical vector for the plasmid side using the same definition.

    Procedure:
      1) Count all 4^k k-mers on both strands (sequence and revcomp(sequence))
      2) Divide by total_valid (sum over both strands) to obtain relative frequencies
      3) Extract only canonical representative k-mers as a vector
         (the value is identical to one of the duplicated columns)
    """
    all_kmers = all_kmers_order(k)
    idx = {km: i for i, km in enumerate(all_kmers)}

    counts = np.zeros(len(all_kmers), dtype=np.int64)

    sequence = "".join(b for b in sequence.upper() if b in "ACGT")
    if len(sequence) < k:
        # Return in canonical dimensions
        return np.zeros(len(canonical_kmers_order(k)), dtype=float)

    targets = [sequence, get_reverse_complement(sequence)]
    total_valid = 0

    for seq in targets:
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            counts[idx[kmer]] += 1
            total_valid += 1

    if total_valid == 0:
        return np.zeros(len(canonical_kmers_order(k)), dtype=float)

    freqs_all = counts.astype(float) / float(total_valid)

    # Extract only canonical columns (representative = min(kmer, rc))
    can_kmers = canonical_kmers_order(k)
    can_vec = np.array([freqs_all[idx[can]] for can in can_kmers], dtype=float)
    return can_vec


def detect_feature_columns(df_h: pd.DataFrame, k: int):
    """
    Detect the canonical columns in the host-side DataFrame and return their column names.
    The column order must match canonical_kmers_order(k).

    Accepted formats:
      1) Bare: 'ACGT', etc. (canonical set only)
      2) With prefix: '{k}mer_ACGT' or 'kmer_ACGT'
    """
    can_kmers = canonical_kmers_order(k)

    # 1) Bare canonical k-mers
    if all(km in df_h.columns for km in can_kmers):
        return can_kmers

    # 2) With prefix
    prefixes = [f"{k}mer_", "kmer_"]
    for pref in prefixes:
        cols = [pref + km for km in can_kmers]
        if all(c in df_h.columns for c in cols):
            return cols

    raise ValueError(
        f"Cannot detect canonical k-mer columns for k={k}. "
        f"Expected canonical set columns like '{k}mer_<CAN>' (or 'kmer_<CAN>' or bare)."
    )


def load_plasmid_sequence(plasmid_path: str) -> str:
    seq_parts = []
    for record in SeqIO.parse(plasmid_path, "fasta"):
        seq_parts.append(str(record.seq).upper())
    return "".join(seq_parts)


def summarize_votes(votes: np.ndarray):
    """
    votes: total number of votes for each host (0..6)
    Returns: vote distribution (count_vote0..count_vote6)
             and number of positive hosts for each threshold v (pos_hosts_v1..v6)
    """
    res = {}
    for m in range(0, 7):
        res[f"count_vote{m}"] = int(np.sum(votes == m))
    for v in range(1, 7):
        res[f"pos_hosts_v{v}"] = int(np.sum(votes >= v))
    return res


def predict_one_plasmid(plasmid_path: str, host_names: np.ndarray):
    plasmid_name = os.path.splitext(os.path.basename(plasmid_path))[0]

    try:
        plasmid_seq = load_plasmid_sequence(plasmid_path)
    except Exception as e:
        raise RuntimeError(f"Error reading FASTA: {e}")

    if not plasmid_seq:
        raise RuntimeError("Empty sequence in FASTA.")

    results = pd.DataFrame({"Host": host_names})
    vote_counts = np.zeros(len(results), dtype=int)

    for k in K_LIST:
        # ★ Construct using the canonical definition aligned with the host-side CSV
        p_vec = count_nucleotides_canonical_like_csv(plasmid_seq, k)

        h_file = os.path.join(HOST_KMER_DIR, f"host_{k}mer_canonical.csv")
        if not os.path.exists(h_file):
            raise FileNotFoundError(f"Missing host feature file: {h_file}")

        df_h = pd.read_csv(h_file)
        if "filename" not in df_h.columns:
            raise ValueError(f"{h_file} must contain 'filename' column.")

        # Align based on host order
        df_h = df_h.copy()
        df_h["filename"] = df_h["filename"].astype(str)
        df_h = df_h.set_index("filename").reindex(host_names).reset_index()

        if df_h["filename"].isna().any():
            missing = int(df_h["filename"].isna().sum())
            raise ValueError(
                f"Host alignment failed for k={k}: {missing} missing hosts in host_{k}mer_canonical.csv. "
                f"Check filename consistency."
            )

        feature_cols = detect_feature_columns(df_h, k)
        h_vecs = df_h[feature_cols].values.astype(float)

        # Check dimensional consistency (to avoid accidents)
        if h_vecs.shape[1] != p_vec.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch for k={k}: "
                f"host has {h_vecs.shape[1]} cols, plasmid vec has {p_vec.shape[0]}."
            )

        X = h_vecs - p_vec  # host - plasmid

        model_path = os.path.join(MODEL_DIR, f"model_k{k}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}")

        model = joblib.load(model_path)

        prob = model.predict_proba(X)[:, 1]
        vote = (prob >= SINGLE_THRESH).astype(int)

        vote_counts += vote

        results[f"Prob_k{k}"] = prob
        results[f"Vote_k{k}"] = vote

    results["Votes"] = vote_counts
    results["VoteFraction"] = vote_counts / len(K_LIST)

    for v in range(1, len(K_LIST) + 1):
        results[f"Prediction_v{v}"] = (vote_counts >= v).astype(int)

    results["Prediction"] = (vote_counts >= VOTE_THRESH).astype(int)

    summary = {
        "Plasmid": plasmid_name,
        "Input_File": plasmid_path,
        "n_hosts": int(len(host_names)),
        "SINGLE_THRESH": float(SINGLE_THRESH),
        "K_LIST": ",".join(map(str, K_LIST)),
        "Representative_VOTE_THRESH": int(VOTE_THRESH),
        "pos_hosts_rep_thresh": int(np.sum(vote_counts >= VOTE_THRESH)),
        "mean_votes": float(np.mean(vote_counts)),
        "median_votes": float(np.median(vote_counts)),
        "max_votes": int(np.max(vote_counts)),
    }
    summary.update(summarize_votes(vote_counts))

    return results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch predict for all FASTA files in a directory (k-mer ensemble) and output (1) per-plasmid detail CSV + (2) combined vote summary CSV."
    )
    parser.add_argument("input_dir", help="Directory containing plasmid FASTA files (.fasta/.fna)")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' not found.")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith((".fasta", ".fna"))])
    print(f"Found {len(files)} plasmids in {args.input_dir}")

    # ★ Obtain host names from the canonical 7-mer file
    # (the current code used to look for host_7mer.csv)
    host7_path = os.path.join(HOST_KMER_DIR, "host_7mer_canonical.csv")
    if not os.path.exists(host7_path):
        print(f"Error: Host k-mer data not found: {host7_path}")
        print("       Create canonical host feature CSVs first (host_{k}mer_canonical.csv).")
        return

    df_h7 = pd.read_csv(host7_path)
    if "filename" not in df_h7.columns:
        print(f"Error: {host7_path} must contain 'filename' column.")
        return

    host_names = df_h7["filename"].astype(str).values

    all_summaries = []

    print(f"Starting Batch Prediction (Representative threshold: Vote >= {VOTE_THRESH})...")
    print("Per-plasmid detail: prediction_<plasmid>.csv")
    print("Combined summary   : batch_votes_summary.csv")

    for fname in tqdm(files, desc="Processing"):
        plasmid_path = os.path.join(args.input_dir, fname)
        plasmid_name = os.path.splitext(fname)[0]

        try:
            df_res, summary = predict_one_plasmid(plasmid_path, host_names)
        except Exception as e:
            print(f"\n[SKIP] {plasmid_name}: {e}")
            continue

        out_detail = os.path.join(OUTPUT_ROOT, f"prediction_{plasmid_name}.csv")
        df_res.to_csv(out_detail, index=False)

        summary["Detail_File"] = out_detail
        all_summaries.append(summary)

    if all_summaries:
        df_sum = pd.DataFrame(all_summaries)

        front_cols = [
            "Plasmid", "Input_File", "Detail_File", "n_hosts",
            "SINGLE_THRESH", "K_LIST",
            "Representative_VOTE_THRESH", "pos_hosts_rep_thresh",
            "mean_votes", "median_votes", "max_votes",
        ]
        vote_dist_cols = [f"count_vote{i}" for i in range(0, 7)]
        pos_cols = [f"pos_hosts_v{i}" for i in range(1, 7)]
        ordered_cols = [c for c in front_cols + vote_dist_cols + pos_cols if c in df_sum.columns]
        df_sum = df_sum[ordered_cols]

        out_sum = os.path.join(OUTPUT_ROOT, "batch_votes_summary.csv")
        df_sum.to_csv(out_sum, index=False)

        print("\n✅ Batch processing complete!")
        print(f"   - Individual results: {OUTPUT_ROOT}/prediction_*.csv")
        print(f"   - Combined summary :  {out_sum}")
        print("\n[Summary Preview]")
        print(df_sum.head())
    else:
        print("No valid results generated.")


if __name__ == "__main__":
    main()
