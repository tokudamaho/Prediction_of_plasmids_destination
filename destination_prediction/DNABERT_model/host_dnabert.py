import os
import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm

# -------------------------------------------------------
# 設定：精度最優先（全読込・スライディングウィンドウ）
# -------------------------------------------------------
CHUNK_SIZE = 510
STRIDE = 256
BATCH_SIZE = 8   # CPU用

# 入出力設定
INPUT_DIR = "data/fna_127"        # ホストのフォルダ
OUTPUT_FILE = "data/host_vectors.csv"

# -------------------------------------------------------
# モデル準備（CPU最適化）
# -------------------------------------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
base_model = AutoModel.from_pretrained("zhihan1996/DNA_bert_6")
base_model.eval()

device = torch.device("cpu")
print(f"Using device: {device}")

print("Applying dynamic quantization...")
model = torch.quantization.quantize_dynamic(
    base_model, {torch.nn.Linear}, dtype=torch.qint8
)
model.to(device)

# -------------------------------------------------------
# 関数定義
# -------------------------------------------------------
def read_fasta_sequence(filepath):
    # 念のため、ファイル内の全レコード（コンティグ等）を結合して1本の配列として扱う
    # ※不要な場合は以前の next(...) のみに戻してもOK
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
    """
    手加減なしの全読込（Full Sliding Window）
    """
    tokens_all = kmer_tokenize(sequence)
    if not tokens_all: return np.zeros(768)
    
    # 全チャンク作成
    chunks = []
    for i in range(0, len(tokens_all), STRIDE):
        chunk = tokens_all[i : i + CHUNK_SIZE]
        if len(chunk) > 10:
            chunks.append(chunk)
            
    if not chunks: return np.zeros(768)

    # 推論実行
    embedding_sum = np.zeros(768)
    count = 0

    with torch.no_grad():
        # バッチ処理なしで1つずつ着実に進める（メモリ安全重視）
        for chunk in chunks:
            inputs = tokenizer([chunk], return_tensors="pt", 
                               is_split_into_words=True, 
                               padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embedding_sum += emb
            count += 1

    return embedding_sum / count

# -------------------------------------------------------
# メイン処理（途中再開機能付き）
# -------------------------------------------------------
target_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".fna")])
print(f"Target Hosts: {len(target_files)}")

# 既に処理済みのファイルを確認
processed_files = set()
if os.path.exists(OUTPUT_FILE):
    # 既存のCSVからファイル名だけ読み取る
    try:
        df_exist = pd.read_csv(OUTPUT_FILE, usecols=['filename'])
        processed_files = set(df_exist['filename'].values)
        print(f"Already processed: {len(processed_files)} files. Resuming...")
    except:
        print("Output file exists but seems empty or broken. Starting fresh.")

# まだ処理していないファイルだけ抽出
files_to_process = [f for f in target_files if f not in processed_files]

if not files_to_process:
    print("All files are already processed!")
    exit()

# ヘッダー作成（初回のみ書き込む）
if not os.path.exists(OUTPUT_FILE):
    header = ['filename'] + [f'feature_{i}' for i in range(768)]
    pd.DataFrame(columns=header).to_csv(OUTPUT_FILE, index=False)

# 1ファイルずつ処理して追記保存
for filename in tqdm(files_to_process):
    filepath = os.path.join(INPUT_DIR, filename)
    
    try:
        # ベクトル計算（ここが時間かかる）
        seq = read_fasta_sequence(filepath)
        vec = get_full_embedding(seq)
        
        # 1行だけDataFrameにする
        df_row = pd.DataFrame([[filename] + vec.tolist()])
        
        # 追記モード(mode='a')で保存。header=Falseにする（既にヘッダーはあるため）
        df_row.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        
    except Exception as e:
        print(f"\nError processing {filename}: {e}")
        continue

print(f"✅ Host vectors saved to: {OUTPUT_FILE}")
