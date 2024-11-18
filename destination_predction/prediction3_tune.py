# -*- coding: utf-8 -*-
import os
import csv
import sys
import itertools
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Generate all possible n-nucleotides (e.g., triplets) from 'ACGT'
def generate_all_nucleotides(n):
    """Generate all possible n-nucleotides from 'ACGT'."""
    return [''.join(p) for p in itertools.product('ACGT', repeat=n)]

# Generate the reverse complement of a DNA sequence
def reverse_complement(sequence):
    """Generate the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence))

# Count occurrences of n-nucleotides in a genome sequence (including reverse complement)
def count_nucleotides(genome_sequence, n):
    """
    Count n-nucleotide frequencies in the given sequence and its reverse complement.

    Parameters:
        genome_sequence: DNA sequence as a string.
        n: Length of n-nucleotide (e.g., 3 for triplets).
    Returns:
        nucleotide_counts: Dictionary with n-nucleotide counts.
        total_nucleotides: Total number of valid n-nucleotides counted.
    """
    nucleotide_counts = {}
    total_nucleotides = 0
    sequences = [genome_sequence, reverse_complement(genome_sequence)]  # Include both strands
    for seq in sequences:
        for i in range(len(seq) - n + 1):  # Slide a window of size n along the sequence
            nucleotide = seq[i:i+n]
            if set(nucleotide) <= set("ACGT"):  # Ensure only valid n-nucleotides are counted
                total_nucleotides += 1
                if nucleotide in nucleotide_counts:
                    nucleotide_counts[nucleotide] += 1
                else:
                    nucleotide_counts[nucleotide] = 1
    return nucleotide_counts, total_nucleotides

# Process a .fna file to calculate n-nucleotide counts
def process_fna_file(file_path, n):
    """
    Extract the DNA sequence from a .fna file and calculate n-nucleotide counts.

    Parameters:
        file_path: Path to the input .fna file.
        n: Length of n-nucleotide (e.g., 3 for triplets).
    Returns:
        Dictionary of relative frequencies of each n-nucleotide.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            current_sequence = []
            nucleotide_counts = {}
            total_nucleotides = 0

            # Process each line in the .fna file
            for line in lines:
                if line.startswith('>'):  # Header line
                    if current_sequence:  # Process the sequence collected so far
                        sequence = ''.join(current_sequence)
                        counts, count_total = count_nucleotides(sequence, n)
                        for nucleotide, count in counts.items():
                            nucleotide_counts[nucleotide] = nucleotide_counts.get(nucleotide, 0) + count
                        total_nucleotides += count_total
                        current_sequence = []
                else:
                    current_sequence.append(line.strip().upper())  # Append sequence line

            # Process the final sequence block
            if current_sequence:
                sequence = ''.join(current_sequence)
                counts, count_total = count_nucleotides(sequence, n)
                for nucleotide, count in counts.items():
                    nucleotide_counts[nucleotide] = nucleotide_counts.get(nucleotide, 0) + count
                total_nucleotides += count_total

            # Normalize counts to relative frequencies
            for nucleotide in nucleotide_counts:
                nucleotide_counts[nucleotide] /= total_nucleotides

            return nucleotide_counts
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
    df_transposed.to_csv(output_file, index=False, header=False)  # Save the transposed CSV file

# Prepare data by calculating differences between rows and repeated columns
def prepare_data(cols_file):
    """
    Prepare data for training by calculating differences between rows and repeated columns.

    Parameters:
        cols_file: Path to the columns data file.
    Returns:
        temp_output_file: Path to the temporary output file.
    """
    rows_file = 'chr126_3_dra_both.csv'  # File containing row data
    rows_df = pd.read_csv(rows_file, header=0, index_col=0)  # Load row data
    cols_df = pd.read_csv(cols_file, header=0, index_col=0)  # Load column data

    rows_header = rows_df.columns.values  # Get header from rows data
    rows_256_sequences = rows_df.values  # Convert rows data to numpy array
    cols_256_sequences = cols_df.values  # Convert columns data to numpy array

    # Repeat columns data to match the number of rows
    cols_repeated = np.repeat(cols_256_sequences, 126, axis=0)

    # Calculate the difference between rows and repeated columns
    data = rows_256_sequences - cols_repeated

    # Save the processed data to a temporary CSV file
    df = pd.DataFrame(data)
    temp_output_file = os.path.join(output_directory, 'temp_testdata.csv')
    df.to_csv(temp_output_file, index=False, header=rows_header)
    return temp_output_file

# Train a Random Forest model and predict labels for test data
def train_and_predict(input_file, output_directory):
    """
    Train a Random Forest model on labeled data and predict labels for test data.

    Parameters:
        input_file: Path to the prepared test data.
        output_directory: Directory to save results.
    """
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load training data
    train_data = pd.read_csv("data_with_labels.csv", delimiter=",", skiprows=0, usecols=range(0, 65))
    X_train = train_data.iloc[:, :-1]  # Features
    y_train = train_data.iloc[:, -1]   # Labels

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_leaf=2, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)

    # Load test data
    test_data = pd.read_csv(input_file, delimiter=",", skiprows=0, usecols=range(0, 64))
    predictions = model.predict(test_data)

    # Save predictions to a CSV file
    output_file_path = os.path.join(output_directory, "Predictions.csv")
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Prediction"])
        writer.writerows(predictions.reshape(-1, 1))
    print(f"Predictions written to {output_file_path}")

    # Save feature importances to a CSV file
    feature_importances = model.feature_importances_
    feature_importances_file_path = os.path.join(output_directory, "Feature_Importances.csv")
    with open(feature_importances_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Feature", "Importance"])
        for i, feature_name in enumerate(test_data.columns):
            writer.writerow([feature_name, feature_importances[i]])
    print(f"Feature importances written to {feature_importances_file_path}")

# Main script
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_fasta_file> <transposed_output_file> <output_directory>")
        sys.exit(1)

    input_fasta_file = sys.argv[1]  # Path to the input FASTA file
    transposed_output_file = sys.argv[2]  # Path to save the transposed output file
    output_directory = sys.argv[3]  # Directory to save results

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Step 1: Count n-nucleotides (e.g., triplets)
    n = 3  # Length of the nucleotide sequences
    all_nucleotides = generate_all_nucleotides(n)  # Generate all possible triplets
    nucleotide_counts = process_fna_file(input_fasta_file, n)  # Count triplets in the input file

    # Save the n-nucleotide counts to a CSV file
    output_file = os.path.join(output_directory, 'triplet_counts.csv')
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'{n}-Nucleotide', 'Count'])
        for nucleotide in all_nucleotides:
            writer.writerow([nucleotide, nucleotide_counts.get(nucleotide, 0)])

    # Step 2: Transpose the CSV file
    transposed_output_file = os.path.join(output_directory, transposed_output_file)
    transpose_csv(output_file, transposed_output
