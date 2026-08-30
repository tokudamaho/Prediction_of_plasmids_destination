#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
現存の k-mer CSV（あなたの現行スクリプト出力）を、再計算なしで canonical k-mer に変換します。

前提（重要）:
- 入力CSVは「forward + reverse complement を両方カウントして relative frequency」で作られている
  → その場合、k-mer列とそのrevcomp列は理論上“完全に同じ値”になります。
- よって canonical化は「ペア列の片方を残して、もう片方を削除」(集約/加算はしない) でOKです。

出力:
- 例: host_4mer.csv -> host_4mer_canonical.csv
- 例: plasmid_6mer.csv -> plasmid_6mer_canonical.csv
"""

import os
import re
import glob
import argparse
import pandas as pd

COMP = str.maketrans("ACGT", "TGCA")

def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]

def canonical_kmer(s: str) -> str:
    rc = revcomp(s)
    return min(s, rc)

def parse_feature_col(col: str):
    """
    "4mer_ACGT" -> (4, "ACGT")
    戻り値: (k:int, kmer:str) or (None, None) if not match
    """
    m = re.match(r"^(\d+)mer_([ACGT]+)$", col)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)

def canonicalize_one_csv(in_csv: str, out_csv: str, *, check_identical: bool = True, tol: float = 1e-12):
    df = pd.read_csv(in_csv)

    if "filename" not in df.columns:
        raise ValueError(f"'filename' column not found in {in_csv}")

    feat_cols = [c for c in df.columns if c != "filename"]

    # canonical列名 -> 元列名リスト
    groups = {}
    # canonical列名の出力順（入力順をできるだけ保持）
    out_order = []

    for c in feat_cols:
        k, kmer = parse_feature_col(c)
        if k is None:
            # 想定外の列はそのまま残す（必要ならここでraiseに変更）
            can_col = c
        else:
            can = canonical_kmer(kmer)
            can_col = f"{k}mer_{can}"

        if can_col not in groups:
            groups[can_col] = []
            out_order.append(can_col)
        groups[can_col].append(c)

    out = pd.DataFrame({"filename": df["filename"]})

    # canonical列を作成：現行CSVなら同一値なので代表1列をコピー
    for can_col in out_order:
        cols = groups[can_col]
        out[can_col] = df[cols[0]]

        # 念のため同一性チェック（理論上は差分0）
        if check_identical and len(cols) > 1:
            # max absolute difference across all rows & columns in this group
            base = df[cols[0]].to_numpy()
            diffs = (df[cols].to_numpy() - base.reshape(-1, 1))
            max_abs_diff = abs(diffs).max()
            if max_abs_diff > tol:
                print(f"[WARNING] {os.path.basename(in_csv)}: group '{can_col}' not identical "
                      f"(max_abs_diff={max_abs_diff:.3e}). columns={cols}")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out.shape[1] - 1, len(feat_cols)  # (n_out_features, n_in_features)

def main():
    ap = argparse.ArgumentParser(
        description="現存k-mer CSVを再計算なしでcanonical k-mer化して保存します（*_canonical.csv）。"
    )
    ap.add_argument("--input_dir", default="results/kmer/kmer_features", help="入力CSVがあるフォルダ")
    ap.add_argument("--pattern", default="*.csv", help="対象CSVのglobパターン（例: 'host_*mer.csv'）")
    ap.add_argument("--suffix", default="_canonical", help="出力ファイル名に付けるサフィックス")
    ap.add_argument("--no_check", action="store_true", help="同一列チェックを無効化（高速化）")
    ap.add_argument("--tol", type=float, default=1e-12, help="同一性チェックの許容誤差")
    args = ap.parse_args()

    in_glob = os.path.join(args.input_dir, args.pattern)
    files = sorted(glob.glob(in_glob))

    if not files:
        raise SystemExit(f"No files matched: {in_glob}")

    # 既にcanonical出力のファイルは除外
    files = [f for f in files if args.suffix not in os.path.splitext(os.path.basename(f))[0]]

    print(f"Found {len(files)} input CSV(s).")
    for in_csv in files:
        base, ext = os.path.splitext(os.path.basename(in_csv))
        out_csv = os.path.join(args.input_dir, f"{base}{args.suffix}{ext}")

        n_out, n_in = canonicalize_one_csv(
            in_csv, out_csv, check_identical=(not args.no_check), tol=args.tol
        )
        print(f"OK: {os.path.basename(in_csv)} -> {os.path.basename(out_csv)} "
              f"(features: {n_in} -> {n_out})")

if __name__ == "__main__":
    main()
