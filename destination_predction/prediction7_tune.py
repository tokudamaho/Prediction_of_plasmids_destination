# -*- coding: utf-8 -*-
import os
import csv
import sys
import itertools
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Generate all possible heptanucleotides (7-mers) from 'ACGT'
def generate_all_heptanucleotides():
    """Generate all possible heptanucleotides (7-mers) from 'ACGT'."""
    return [''.join(p) for p in itertools.product('ACGT', repeat=7)]

# Generate the reverse complement of a DNA sequence
def reverse_complement(sequence):
    """Generate the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence))

# Count occurrences of heptanucleotides in a genome sequence (including reverse complement)
def count_heptanucleotides(genome_sequence):
    """
    Count heptanucleotide frequencies in the given sequence and its reverse complement.

    Parameters:
        genome_sequence: DNA sequence as a string.
    Returns:
        Dictionary with heptanucleotide relative frequencies.
    """
    heptanucleotide_counts = {}
    total_heptanucleotides = 0
    sequences = [genome_sequence, reverse_complement(genome_sequence)]  # Include both strands
    for seq in sequences:
        for i in range(len(seq) - 6):  # Slide a window of size 7 along the sequence
            heptanucleotide = seq[i:i+7]
            if set(heptanucleotide) <= set("ACGT"):  # Only count valid ACGT heptanucleotides
                total_heptanucleotides += 1
                heptanucleotide_counts[heptanucleotide] = heptanucleotide_counts.get(heptanucleotide, 0) + 1
    # Normalize counts to relative frequencies
    for heptanucleotide in heptanucleotide_counts:
        heptanucleotide_counts[heptanucleotide] /= total_heptanucleotides
    return heptanucleotide_counts

# Process a .fna file to count heptanucleotides
def process_fna_file(file_path):
    """
    Extract DNA sequence from a .fna file and count heptanucleotides.

    Parameters:
        file_path: Path to the input .fna file.
    Returns:
        Dictionary with heptanucleotide frequencies.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Concatenate sequence lines, ignoring header lines starting with '>'
            genome_sequence = ''.join(line.strip().upper() for line in lines if not line.startswith('>'))
            return count_heptanucleotides(genome_sequence)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)

# Transpose a CSV file
def transpose_csv(input_file, output_file):
    """
    Transpose rows and columns of a CSV file.

    Parameters:
        input_file: Path to the input CSV file.
        output_file: Path to save the transposed CSV file.
    """
    df = pd.read_csv(input_file, header=None)  # Read the input CSV file
    df_transposed = df.T  # Transpose the DataFrame
    df_transposed.to_csv(output_file, index=False, header=False)  # Save the transposed file

# Prepare data by calculating differences between rows and repeated columns
def prepare_data(cols_file):
    """
    Prepare data for training by calculating differences between rows and repeated columns.

    Parameters:
        cols_file: Path to the columns data file.
    Returns:
        Path to the temporary output file with prepared data.
    """
    rows_file = 'data/chr127_7_dra_both.csv'  # File containing row data
    rows_df = pd.read_csv(rows_file, header=0, index_col=0)  # Load row data
    cols_df = pd.read_csv(cols_file, header=0, index_col=0)  # Load column data

    rows_header = rows_df.columns.values  # Extract headers from rows data
    rows_sequences = rows_df.values  # Convert rows data to numpy array
    cols_sequences = cols_df.values  # Convert columns data to numpy array

    # Repeat columns data to match the number of rows
    cols_repeated = np.repeat(cols_sequences, 127, axis=0)

    # Calculate differences between rows and repeated columns
    data = rows_sequences - cols_repeated

    # Save the prepared data to a temporary file
    temp_output_file = os.path.join(output_directory, 'temp_testdata7.csv')
    pd.DataFrame(data).to_csv(temp_output_file, index=False, header=rows_header)
    return temp_output_file

# Train a Random Forest model and predict labels for test data
def train_and_predict(input_file, output_directory):
    """
    Train a Random Forest model on labeled data and predict test data.

    Parameters:
        input_file: Path to the prepared test data.
        output_directory: Directory to save results.
    """
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load training data
    train_data = pd.read_csv("data/data_with_labels72.csv", delimiter=",", usecols=range(0, 16385))
    X_train = train_data.iloc[:, :-1]  # Features
    y_train = train_data.iloc[:, -1]   # Labels

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)

    # Load test data
    test_data = pd.read_csv(input_file, delimiter=",", usecols=range(0, 16384))
    predictions = model.predict(test_data)

    # Save predictions to a CSV file
    predictions_file = os.path.join(output_directory, "Predictions7.csv")
    pd.DataFrame(predictions, columns=["Prediction"]).to_csv(predictions_file, index=False)
    print(f"Predictions written to {predictions_file}")

    # Save feature importances to a CSV file
    feature_importances = model.feature_importances_
    importance_file = os.path.join(output_directory, "Feature_Importances7.csv")
    pd.DataFrame({'Feature': test_data.columns, 'Importance': feature_importances}).to_csv(importance_file, index=False)
    print(f"Feature importances written to {importance_file}")

# Main script
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_fasta_file> <output_directory>")
        sys.exit(1)

    input_fasta_file = sys.argv[1]  # Path to the input FASTA file
    output_directory = sys.argv[2]  # Directory to save results
    transposed_output_file = "transposed7.csv"  # Fixed file name for transposed output

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Step 1: Count heptanucleotides
    heptanucleotide_counts = process_fna_file(input_fasta_file)

    # Save heptanucleotide counts to a CSV file
    counts_file = os.path.join(output_directory, 'heptanucleotide_counts.csv')
    pd.DataFrame.from_dict(heptanucleotide_counts, orient='index', columns=['Frequency']).reset_index() \
        .rename(columns={'index': 'Heptanucleotide'}).to_csv(counts_file, index=False)
    print(f"Heptanucleotide counts written to {counts_file}")

    # Step 2: Transpose the CSV file
    transposed_path = os.path.join(output_directory, transposed_output_file)
    transpose_csv(counts_file, transposed_path)

    # Step 3: Prepare data for prediction
    prepared_file = prepare_data(transposed_path)

    # Step 4: Train and predict
    train_and_predict(prepared_file, output_directory)
