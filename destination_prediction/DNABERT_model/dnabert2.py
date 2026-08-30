import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np

# Prepare the model
MODEL_NAME = "zhihan1996/DNABERT-2-117M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()

# Function to read a nucleotide sequence from FASTA
def read_fasta(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    seq = "".join([line.strip() for line in lines if not line.startswith(">")])
    return seq.upper()

# Vectorize the sequence with DNABERT-2
def get_embedding(sequence):
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    vec = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return vec

# Settings
host_dir = "fna_127"
plasmid_dir = "allplasmids"
pair_csv = "pairs_converted.csv"
output_csv = "output_embeddings.csv"

# Load data
df = pd.read_csv(pair_csv)
embeddings = []
labels = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    host_file = os.path.join(host_dir, row['host_file'])
    plasmid_file = os.path.join(plasmid_dir, row['plasmid_file'])
    
    try:
        host_seq = read_fasta(host_file)
        plasmid_seq = read_fasta(plasmid_file)

        host_vec = get_embedding(host_seq)
        plasmid_vec = get_embedding(plasmid_seq)

        combined_vec = np.concatenate([host_vec, plasmid_vec])
        embeddings.append(combined_vec)
        labels.append(row['label'])

    except Exception as e:
        print(f"⚠️ Error: {row['host_file']} × {row['plasmid_file']} - {e}")

# Save as CSV
embedding_df = pd.DataFrame(embeddings)
embedding_df["label"] = labels
embedding_df.to_csv(output_csv, index=False)

print(f"\n✅ Saved vectors to: {output_csv}")
