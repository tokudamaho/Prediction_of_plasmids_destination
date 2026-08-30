import os
import sys
import argparse
import joblib
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO

# =================================================
# DNABERT Plasmid -> Host prediction (Folder-enabled)
# - Embedding generation matches TRAINING vector script:
#   CHUNK_SIZE=510, STRIDE=256, 6-mer tokenize, mean-pool over tokens,
#   average over chunks, CPU + dynamic quantization.
# - Uses fixed, trained RF models: model_{Diff,AbsDiff,Prod}.pkl
# - NEW: input can be a FASTA file OR a directory containing FASTA files
#        output is one CSV per plasmid (file)
# =================================================

# -------------------------------------------------
# Config (edit if needed)
# -------------------------------------------------
MODEL_DIR = "results/results_dnabert_nestedcv_randomsplit"
HOST_VEC_FILE = "data/host_vectors.csv"
DNABERT_PATH = "zhihan1996/DNA_bert_6"
THRESHOLD = 0.5

CHUNK_SIZE = 510
STRIDE = 256

METHODS = ["Diff", "AbsDiff", "Prod"]


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def read_fasta_sequence(filepath: str) -> str:
    """Concatenate all records, uppercase, keep only ATGC (matches training)."""
    seq_parts = []
    for record in SeqIO.parse(filepath, "fasta"):
        s = str(record.seq).upper()
        s = "".join([b for b in s if b in "ATGC"])
        seq_parts.append(s)
    return "".join(seq_parts)


def load_host_vectors(host_vec_file: str):
    """
    Load host vectors. Supports:
      (A) columns: filename, feature_0..feature_767
      (B) columns: [first col is filename], then 768 features
    """
    if not os.path.exists(host_vec_file):
        print(f"Error: Host vector file '{host_vec_file}' not found.")
        sys.exit(1)

    df = pd.read_csv(host_vec_file)

    feature_cols = [f"feature_{i}" for i in range(768)]
    if feature_cols[0] in df.columns:
        host_names = df["filename"].astype(str).values
        host_vecs = df[feature_cols].values.astype(float)
    else:
        host_names = df.iloc[:, 0].astype(str).values
        host_vecs = df.iloc[:, 1:].values.astype(float)

    if host_vecs.shape[1] != 768:
        raise ValueError(f"Host vectors dim mismatch: {host_vecs.shape}. Expected (*, 768).")

    return host_names, host_vecs


def kmer_tokenize(sequence: str, k: int = 6):
    if len(sequence) < k:
        return []
    return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]


def load_dnabert_quantized_cpu(model_name: str):
    """
    Match training vector script:
      - CPU device
      - dynamic quantization on Linear layers
    """
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    base_model = AutoModel.from_pretrained(model_name)
    base_model.eval()

    model = torch.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device


def get_dnabert_embedding(sequence: str, tokenizer, model, device) -> np.ndarray:
    """
    EXACTLY matches training plasmid vector script:
      tokens_all = 6-mer tokenize
      chunks: for i in range(0, len(tokens_all), STRIDE)
              chunk = tokens_all[i:i+CHUNK_SIZE]
              keep if len(chunk) > 10
      embedding per chunk: mean over sequence length (dim=1) of last_hidden_state
      final: average over chunks
    """
    tokens_all = kmer_tokenize(sequence, k=6)
    if not tokens_all:
        return np.zeros(768, dtype=float)

    chunks = []
    for i in range(0, len(tokens_all), STRIDE):
        chunk = tokens_all[i : i + CHUNK_SIZE]
        if len(chunk) > 10:
            chunks.append(chunk)

    if not chunks:
        return np.zeros(768, dtype=float)

    embedding_sum = np.zeros(768, dtype=float)
    count = 0

    with torch.no_grad():
        for chunk in chunks:
            inputs = tokenizer(
                [chunk],
                return_tensors="pt",
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embedding_sum += emb
            count += 1

    return embedding_sum / max(count, 1)


def generate_feature_X(method_name: str, h_vecs: np.ndarray, p_vec: np.ndarray) -> np.ndarray:
    """
    Create X for all hosts against one plasmid vector.
    h_vecs: (n_hosts, 768)
    p_vec:  (768,)
    returns: (n_hosts, 768)
    """
    if p_vec.shape != (768,):
        raise ValueError(f"Plasmid vector dim mismatch: {p_vec.shape}. Expected (768,).")

    if method_name == "Diff":
        return h_vecs - p_vec
    if method_name == "AbsDiff":
        return np.abs(h_vecs - p_vec)
    if method_name == "Prod":
        return h_vecs * p_vec

    raise ValueError(f"Unknown method: {method_name}")


def ensure_models_exist(model_dir: str, methods):
    missing = []
    for m in methods:
        path = os.path.join(model_dir, f"model_{m}.pkl")
        if not os.path.exists(path):
            missing.append(path)
    return missing


def collect_fasta_files(path: str):
    """
    If path is a FASTA file -> return [path]
    If path is a directory -> return sorted list of FASTA files inside
    """
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        fasta_ext = (".fa", ".fasta", ".fna")
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(fasta_ext)
        ]
        if not files:
            raise ValueError(f"No FASTA files found in directory: {path}")
        return sorted(files)

    raise ValueError(f"Input path is neither file nor directory: {path}")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict plasmid destination using DNABERT embeddings + fixed RF models "
            "(Diff/AbsDiff/Prod). Input can be a FASTA file or a directory of FASTA files. "
            "Embedding logic matches training vector script."
        )
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input plasmid FASTA file OR directory containing FASTA files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output CSV filename. If input is a directory, this is treated as a prefix "
            "(e.g., results.csv -> results_<plasmid>.csv). If omitted, outputs "
            "prediction_dnabert_<plasmid>.csv per plasmid."
        ),
    )
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR, help="Directory containing model_*.pkl")
    parser.add_argument("--host_vec", type=str, default=HOST_VEC_FILE, help="Host vectors CSV file")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Decision threshold for Pred_*")
    parser.add_argument("--dnabert", type=str, default=DNABERT_PATH, help="DNABERT model name/path")

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' not found.")
        sys.exit(1)

    missing_models = ensure_models_exist(args.model_dir, METHODS)
    if missing_models:
        print("Error: Missing model files:")
        for p in missing_models:
            print(f"  - {p}")
        sys.exit(1)

    # Collect FASTA inputs (file or directory)
    try:
        fasta_files = collect_fasta_files(args.input_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load DNABERT once (CPU + quantization, matching training)
    print("Loading DNABERT model (CPU + dynamic quantization)...")
    try:
        tokenizer, model, device = load_dnabert_quantized_cpu(args.dnabert)
    except Exception as e:
        print(f"Failed to load DNABERT: {e}")
        sys.exit(1)

    # Load hosts once
    print(f"Loading host vectors from: {args.host_vec}")
    try:
        host_names, h_vecs = load_host_vectors(args.host_vec)
    except Exception as e:
        print(f"Error loading host vectors: {e}")
        sys.exit(1)

    # Pre-load RF models once (optional but faster)
    rf_models = {}
    for method in METHODS:
        model_path = os.path.join(args.model_dir, f"model_{method}.pkl")
        try:
            rf_models[method] = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            sys.exit(1)

    print(f"\nFound {len(fasta_files)} FASTA input(s). Starting predictions...")
    print("=" * 60)

    for fasta_path in fasta_files:
        plasmid_name = os.path.splitext(os.path.basename(fasta_path))[0]

        # Decide output filename for this plasmid
        if args.output is None:
            output_csv = f"prediction_dnabert_{plasmid_name}.csv"
        else:
            base, ext = os.path.splitext(args.output)
            if ext == "":
                ext = ".csv"
            output_csv = f"{base}_{plasmid_name}{ext}"

        print(f"\nProcessing plasmid: {plasmid_name}")
        print(f"Input FASTA: {fasta_path}")
        print("-" * 60)

        # Read plasmid
        try:
            plasmid_seq = read_fasta_sequence(fasta_path)
        except Exception as e:
            print(f"  Error reading FASTA: {e}")
            continue

        if len(plasmid_seq) == 0:
            print("  Skipped: plasmid sequence is empty after filtering to ATGC.")
            continue

        print(f"  Calculating plasmid embedding (len={len(plasmid_seq)} bp)...")
        p_vec = get_dnabert_embedding(plasmid_seq, tokenizer, model, device)

        df_results = pd.DataFrame({"Host": host_names})

        for method in METHODS:
            clf = rf_models[method]
            try:
                X = generate_feature_X(method, h_vecs, p_vec)
                probs = clf.predict_proba(X)[:, 1]
                preds = (probs > args.threshold).astype(int)

                df_results[f"Prob_{method}"] = probs
                df_results[f"Pred_{method}"] = preds
            except Exception as e:
                print(f"  Error during prediction ({method}): {e}")
                # keep going to next method
                continue

        df_results.to_csv(output_csv, index=False)
        print(f"  ✅ Saved: {output_csv}")

    print("\n" + "=" * 60)
    print("✅ All predictions completed.")


if __name__ == "__main__":
    main()
