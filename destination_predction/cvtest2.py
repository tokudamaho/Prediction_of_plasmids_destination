import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import csv
import sys

def run_random_split(X, y, test_size, random_state):
    # 1回のランダム分割＋学習評価
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

def main(input_file, output_dir):
    data = pd.read_csv(input_file)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    random_states = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    all_results = []

    for i, rs in enumerate(random_states):
        print(f"Running trial {i+1}/10 with random_state={rs}...")
        results = run_random_split(X, y, test_size=0.2, random_state=rs)
        all_results.append(results)

    results_df = pd.DataFrame(all_results)

    # 平均・標準偏差
    mean_series = results_df.mean()
    std_series = results_df.std()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_df.to_csv(os.path.join(output_dir, "RandomSplit_10trials_Scores.csv"), index_label="Trial")

    summary_df = pd.DataFrame({
        "Metric": mean_series.index,
        "Mean": mean_series.values,
        "StdDev": std_series.values
    })
    summary_df.to_csv(os.path.join(output_dir, "RandomSplit_10trials_Statistics.csv"), index=False)

    print("Evaluation complete. Results and summary saved.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python random_split_test.py input_file output_directory")
        sys.exit(1)
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_file, output_dir)
