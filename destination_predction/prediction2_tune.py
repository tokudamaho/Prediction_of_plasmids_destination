# -*- coding: utf-8 -*-
import os
import csv
import sys
import itertools
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Generate all possible dinucleotides (2-mers) from 'ACGT'
def generate_all_dinucleotides():
    """Generate all possible dinucleotides from 'ACGT'."""
    return [''.join(p) for p in itertools.product('ACGT', repeat=2)]

# Generate the reverse complement of a DNA sequence
def reverse_complement(sequence):
    """Generate the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence))

# Count occurrences of dinucleotides in a genome sequence (including reverse complement)
def count_dinucleotides(genome_sequence):
    """
    Count dinucleotide frequencies in the given sequence and its reverse complement.

    Parameters:
        genome_sequence: DNA sequence as a string.
    Returns:
        dinucleotide_counts: A dictionary with dinucleotide relative frequencies.
    """
    dinucleotide_counts = {}
    total_dinucleotides = 0
    sequences = [genome_sequence, reverse_complement(genome_sequence)]  # Include both strands
    for seq in sequences:
        for i in range(len(seq) - 1):  # Slide over the sequence to extract dinucleotides
            dinucleotide = seq[i:i+2]
            if set(dinucleotide) <= set("ACGT"):  # Count valid dinucleotides only
                total_dinucleotides += 1
                if dinucleotide in dinucleotide_counts:
                    dinucleotide_counts[dinucleotide] += 1
                else:
                    dinucleotide_counts[dinucleotide] = 1
    # Normalize counts to relative frequencies
    for dinucleotide in dinucleotide_counts:
        dinucleotide_counts[dinucleotide] /= total_dinucleotides
    return dinucleotide_counts

# Process a .fna file to calculate dinucleotide counts
def process_fna_file(file_path):
    """
    Extract the DNA sequence from a .fna file and calculate dinucleotide counts.

    Parameters:
        file_path: Path to the input .fna file.
    Returns:
        dinucleotide_counts: A dictionary of dinucleotide frequencies.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Concatenate all sequence lines, ignoring header lines (starting with '>')
            genome_sequence = ''.join(line.strip().upper() for line in lines if not line.startswith('>'))
            return count_dinucleotides(genome_sequence)
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
    df = pd.read_csv(input_file, header=None)  # Read input CSV
    df_transposed = df.T  # Transpose the DataFrame
    df_transposed.to_csv(output_file, index=False, header=False)  # Save transposed data

# Prepare data by processing rows and columns and calculating differences
def prepare_data(cols_file):
    """
    Prepare data by calculating the difference between rows and repeated columns.

    Parameters:
        cols_file: Path to the input CSV file for columns.
    Returns:
        temp_output_file: Path to the temporary output file.
    """
    rows_file = 'chr126_2_dra_both.csv'  # File containing row data
    rows_df = pd.read_csv(rows_file, header=0, index_col=0)  # Load row data
    cols_df = pd.read_csv(cols_file, header=0, index_col=0)  # Load column data

    rows_header = rows_df.columns.values  # Header from row data
    rows_256_sequences = rows_df.values  # Convert rows to numpy array
    cols_256_sequences = cols_df.values  # Convert columns to numpy array

    # Repeat column data to match the row structure
    cols_repeated = np.repeat(cols_256_sequences, 126, axis=0)  # Repeat for 126 rows

    # Calculate the difference between rows and repeated columns
    data = rows_256_sequences - cols_repeated

    # Save the resulting data to a temporary CSV file
    df = pd.DataFrame(data)
    temp_output_file = os.path.join(output_directory, 'temp_testdata.csv')
    df.to_csv(temp_output_file, index=False, header=rows_header)
    return temp_output_file

# Train a Random Forest model and predict test data
def train_and_predict(input_file, output_directory):
    """
    Train a Random Forest model on labeled data and predict labels for test data.

    Parameters:
        input_file: Path to the prepared test data.
        output_directory: Directory to save results.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load training data
    train_data = pd.read_csv("data_with_labels.csv", delimiter=",", skiprows=0, usecols=range(0, 17))
    X_train = train_data.iloc[:, :-1]  # Features
    y_train = train_data.iloc[:, -1]   # Labels

    # Initialize and train the Random Forest model
    model = RandomForestClassifier(max_depth=20, n_estimators=100, min_samples_leaf=2, min_samples_split=2, random_state=42)
    model.fit(X_train, y_train)

    # Load test data
    test_data = pd.read_csv(input_file, delimiter=",", skiprows=0, usecols=range(0, 16))
    predictions = model.predict(test_data)

    # Save predictions to a CSV file
    output_file_path = os.path.join(output_directory, "Predictions.csv")
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Prediction"])
        writer.writerows(predictions.reshape(-1, 1))  # Write predictions

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

    # Step 1: Count dinucleotides
    all_dinucleotides = generate_all_dinucleotides()  # Generate all dinucleotide combinations
    dinucleotide_counts = process_fna_file(input_fasta_file)  # Count dinucleotides in the input sequence

    # Save dinucleotide counts to a CSV file
    output_file = os.path.join(output_directory, 'dinucleotide_counts.csv')
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Dinucleotide', 'Count'])
        for dinucleotide in all_dinucleotides:
            writer.writerow([dinucleotide, dinucleotide_counts.get(dinucleotide, 0)])

    # Step 2: Transpose the CSV file
    transposed_output_file = os.path.join(output_directory, transposed_output_file)
    transpose_csv(output_file, transposed_output_file)

    # Step 3: Prepare data for prediction
    input_file = prepare_data(transposed_output_file)

    # Step 4: Train the model and predict
    train_and_predict(input_file, output_directory)
