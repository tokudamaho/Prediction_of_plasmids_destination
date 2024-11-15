# -*- coding: utf-8 -*-
import os
import csv
import sys
import itertools
import pandas as pd

# Generate all possible n-nucleotides (k-mers) from 'ACGT'
# For example, if n=2, it generates ['AA', 'AC', 'AG', ..., 'TT']
def generate_all_nucleotides(n):
    """Generate all possible n-nucleotides from 'ACGT'."""
    return [''.join(p) for p in itertools.product('ACGT', repeat=n)]

# Create the reverse complement of a DNA sequence
# A -> T, C -> G, G -> C, T -> A
def reverse_complement(sequence):
    """Generate the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence))

# Remove non-standard DNA bases (anything other than A, C, G, T) from a sequence
def remove_non_standard_bases(sequence):
    """Remove non-standard bases (anything other than A, C, G, T) from the sequence."""
    return ''.join(base for base in sequence if base in "ACGT")

# Count occurrences of all n-nucleotides (k-mers) in a genome sequence and its reverse complement
def count_nucleotides(genome_sequence, n):
    """
    Count the occurrences of each n-nucleotide in the given genome sequence and its reverse complement.
    
    Parameters:
        genome_sequence: A string representing the genome sequence.
        n: The size of the k-mers (e.g., n=3 for trinucleotides).
    
    Returns:
        A dictionary with k-mer counts and the total number of k-mers counted.
    """
    nucleotide_counts = {}  # Dictionary to store counts of each k-mer
    total_nucleotides = 0   # Counter for the total number of k-mers
    genome_sequence = remove_non_standard_bases(genome_sequence)  # Clean up the sequence
    sequences = [genome_sequence, reverse_complement(genome_sequence)]  # Include both strands
    for seq in sequences:
        for i in range(len(seq) - n + 1):  # Slide a window of size n along the sequence
            nucleotide = seq[i:i+n]
            if set(nucleotide) <= set("ACGT"):  # Only count valid k-mers
                total_nucleotides += 1
                if nucleotide in nucleotide_counts:
                    nucleotide_counts[nucleotide] += 1
                else:
                    nucleotide_counts[nucleotide] = 1
    return nucleotide_counts, total_nucleotides

# Process a .fna or .fasta file to calculate k-mer relative frequencies
def process_fna_file(file_path, n):
    """
    Process a .fna or .fasta file and return the relative frequency of n-nucleotides.
    
    Parameters:
        file_path: Path to the .fna or .fasta file.
        n: The size of the k-mers (e.g., n=4 for tetranucleotides).
    
    Returns:
        A dictionary containing the relative frequencies of each k-mer.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()  # Read all lines from the file
            current_sequence = []  # To accumulate sequence data for each record
            nucleotide_counts = {}  # To store k-mer counts for the entire file
            total_nucleotides = 0   # Total number of k-mers across all sequences
            for line in lines:
                if line.startswith('>'):  # If the line starts with '>', it's a header line
                    if current_sequence:  # If there's a sequence collected, process it
                        sequence = ''.join(current_sequence)
                        counts, count_total = count_nucleotides(sequence, n)
                        for nucleotide, count in counts.items():
                            if nucleotide in nucleotide_counts:
                                nucleotide_counts[nucleotide] += count
                            else:
                                nucleotide_counts[nucleotide] = count
                        total_nucleotides += count_total
                        current_sequence = []  # Reset for the next sequence
                else:
                    current_sequence.append(line.strip().upper())  # Add the sequence line, converting to uppercase
            
            if current_sequence:  # Process the last sequence if any
                sequence = ''.join(current_sequence)
                counts, count_total = count_nucleotides(sequence, n)
                for nucleotide, count in counts.items():
                    if nucleotide in nucleotide_counts:
                        nucleotide_counts[nucleotide] += count
                    else:
                        nucleotide_counts[nucleotide] = count
                total_nucleotides += count_total

            # Convert counts to relative frequencies
            for nucleotide in nucleotide_counts:
                nucleotide_counts[nucleotide] /= total_nucleotides

            return nucleotide_counts
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}

# Process all .fna or .fasta files in a folder and write k-mer counts to a CSV file
def process_files(input_folder, output_file, n):
    """
    Process all files in the input folder and write the nucleotide counts to the output file.
    
    Parameters:
        input_folder: Path to the folder containing input files.
        output_file: Path to the CSV file where the results will be written.
        n: The size of the k-mers (e.g., n=3 for trinucleotides).
    """
    all_nucleotides = generate_all_nucleotides(n)  # Generate all possible k-mers of size n
    file_nucleotide_counts = {}  # To store k-mer counts for each file
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".fasta") or file_name.endswith(".fna"):  # Process only .fasta or .fna files
            file_path = os.path.join(input_folder, file_name)
            nucleotide_counts = process_fna_file(file_path, n)
            file_nucleotide_counts[file_name] = nucleotide_counts
    
    # Write k-mer counts to the CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'{n}-Nucleotide'] + list(file_nucleotide_counts.keys()))  # Header row
        for nucleotide in all_nucleotides:
            # Create a row with the k-mer and its count in each file
            row = [nucleotide] + [file_nucleotide_counts[file_name].get(nucleotide, 0) for file_name in file_nucleotide_counts.keys()]
            writer.writerow(row)

# Transpose a CSV file (swap rows and columns)
def transpose_csv(input_file, output_file):
    """
    Transpose the CSV file.
    
    Parameters:
        input_file: Path to the input CSV file.
        output_file: Path to the output CSV file (transposed version).
    """
    df = pd.read_csv(input_file, header=None)  # Read the CSV file as a DataFrame
    df_transposed = df.T  # Transpose the DataFrame
    df_transposed.to_csv(output_file, index=False, header=False)  # Write the transposed DataFrame to a new file

# Main function to process files and handle intermediate output
def main(input_folder, output_file, n):
    """
    Main function to process files and write transposed output.
    
    Parameters:
        input_folder: Path to the folder containing input files.
        output_file: Path to the final output file.
        n: The size of the k-mers (e.g., n=4 for tetranucleotides).
    """
    intermediate_file = "intermediate_file.csv"  # Temporary file for intermediate results
    process_files(input_folder, intermediate_file, n)  # Process files and write results to the intermediate file
    transpose_csv(intermediate_file, output_file)  # Transpose the results and write to the final output file
    os.remove(intermediate_file)  # Clean up the intermediate file

# Entry point of the script
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python k-merfreq_both.py <input_folder> <output_file> <n>")  # Provide usage instructions
        sys.exit(1)

    input_folder = sys.argv[1]  # First command-line argument: Input folder
    output_file = sys.argv[2]  # Second command-line argument: Output file
    n = int(sys.argv[3])       # Third command-line argument: k-mer size
    main(input_folder, output_file, n)  # Call the main function
