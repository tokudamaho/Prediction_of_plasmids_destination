import os
import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm

# 設定
CHUNK_SIZE = 510
STRIDE = 256
INPUT_DIR = "allplasmids2"   # プラスミドのフォルダ
OUTPUT_FILE = "plasmid_vectors.csv"

# モデル準備
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
base_model = AutoModel.from_pretrained("zhihan1996/DNA_bert_6")
base_model.eval()
device = torch.device("cpu")
model = torch.quantization.quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
model.to(device)

# 関数定義
def read_fasta_sequence(filepath):
    # プラスミドも念のため全レコード結合
    seqs = []
    for record in SeqIO.parse(filepath, "fasta"):
        s = str(record.seq).upper()
        s = ''.join([base for base in s if base in "ATGC"])
        seqs.append(s)
    return "".join(seqs)

def kmer_tokenize(sequence, k=6):
    if len(sequence) < k: return []
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def get_full_embedding(sequence):
    tokens_all = kmer_tokenize(sequence)
    if not tokens_all: return np.zeros(768)
    
    chunks = []
    for i in range(0, len(tokens_all), STRIDE):
        chunk = tokens_all[i : i + CHUNK_SIZE]
        if len(chunk) > 10: chunks.append(chunk)
    if not chunks: return np.zeros(768)

    embedding_sum = np.zeros(768)
    count = 0
    with torch.no_grad():
        for chunk in chunks:
            inputs = tokenizer([chunk], return_tensors="pt", is_split_into_words=True, 
                               padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embedding_sum += emb
            count += 1
    return embedding_sum / count

# メイン処理
target_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".fasta")])
print(f"Target Plasmids: {len(target_files)}")

processed_files = set()
if os.path.exists(OUTPUT_FILE):
    try:
        df_exist = pd.read_csv(OUTPUT_FILE, usecols=['filename'])
        processed_files = set(df_exist['filename'].values)
        print(f"Resuming... ({len(processed_files)} done)")
    except: pass

files_to_process = [f for f in target_files if f not in processed_files]

if not os.path.exists(OUTPUT_FILE):
    header = ['filename'] + [f'feature_{i}' for i in range(768)]
    pd.DataFrame(columns=header).to_csv(OUTPUT_FILE, index=False)

for filename in tqdm(files_to_process):
    filepath = os.path.join(INPUT_DIR, filename)
    try:
        seq = read_fasta_sequence(filepath)
        vec = get_full_embedding(seq)
        df_row = pd.DataFrame([[filename] + vec.tolist()])
        df_row.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    except Exception as e:
        print(f"Error: {filename} -> {e}")

print(f"✅ Plasmid vectors saved to: {OUTPUT_FILE}")