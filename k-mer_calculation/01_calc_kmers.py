import os
import csv
import sys
import itertools
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# =================================================
# 設定
# =================================================
HOST_DIR = "data/fna_127"          # ホストのフォルダ
PLASMID_DIR = "data/allplasmids"  # プラスミドのフォルダ
OUTPUT_DIR = "results/kmer_features"  # 出力先フォルダ

# kの範囲 (2-7)
K_RANGE = range(2, 8)
# =================================================

def get_reverse_complement(seq):
    """逆相補鎖を生成"""
    complement = str.maketrans('ACGT', 'TGCA')
    return seq.translate(complement)[::-1]

def count_nucleotides(sequence, k):
    """
    指定されたkの頻度を計算 (両鎖考慮、正規化済み)
    """
    # 全パターン生成 (例: AA, AC...)
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    counts = {kmer: 0 for kmer in all_kmers}
    
    # ACGTのみ抽出
    sequence = ''.join(b for b in sequence.upper() if b in "ACGT")
    if len(sequence) < k:
        return [0.0] * len(all_kmers), all_kmers

    # 正鎖と相補鎖
    targets = [sequence, get_reverse_complement(sequence)]
    total_valid = 0
    
    for seq in targets:
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i+k]
            if kmer in counts:
                counts[kmer] += 1
                total_valid += 1
    
    # 正規化 (Relative Frequency)
    if total_valid > 0:
        freqs = [counts[kmer] / total_valid for kmer in all_kmers]
    else:
        freqs = [0.0] * len(all_kmers)
        
    # カラム名 (例: 2mer_AA)
    col_names = [f"{k}mer_{kmer}" for kmer in all_kmers]
    
    return freqs, col_names

def process_directory(input_dir, output_prefix):
    """ディレクトリ内の全ファイルを処理してkごとにCSV保存"""
    files = sorted([f for f in os.listdir(input_dir) if f.endswith((".fna", ".fasta"))])
    print(f"Processing {len(files)} files in {input_dir}...")
    
    # ファイル内容をメモリにロード（高速化のため）
    sequences = {}
    for fname in tqdm(files, desc="Reading files"):
        path = os.path.join(input_dir, fname)
        try:
            # 全コンティグを結合
            seq_parts = []
            for record in SeqIO.parse(path, "fasta"):
                seq_parts.append(str(record.seq).upper())
            sequences[fname] = "".join(seq_parts)
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    # kごとに計算して保存
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for k in K_RANGE:
        print(f"  Computing {k}-mer frequencies...")
        data_rows = []
        header = ["filename"]
        header_set = False
        
        for fname, seq in sequences.items():
            freqs, names = count_nucleotides(seq, k)
            
            if not header_set:
                header.extend(names)
                header_set = True
            
            data_rows.append([fname] + freqs)
            
        # CSV保存
        out_file = os.path.join(OUTPUT_DIR, f"{output_prefix}_{k}mer.csv")
        df = pd.DataFrame(data_rows, columns=header)
        df.to_csv(out_file, index=False)
        print(f"    -> Saved: {out_file}")

if __name__ == "__main__":
    print("--- Generating Host k-mers ---")
    process_directory(HOST_DIR, "host")
    
    print("\n--- Generating Plasmid k-mers ---")
    process_directory(PLASMID_DIR, "plasmid")