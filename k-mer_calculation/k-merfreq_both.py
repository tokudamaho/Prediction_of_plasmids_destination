# -*- coding: utf-8 -*-
import os
import csv
import sys
import itertools
import pandas as pd

def generate_all_nucleotides(n):
    """Generate all possible n-nucleotides from 'ACGT'."""
    return [''.join(p) for p in itertools.product('ACGT', repeat=n)]

def reverse_complement(sequence):
    """Generate the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence))

def remove_non_standard_bases(sequence):
    """Remove non-standard bases (anything other than A, C, G, T) from the sequence."""
    return ''.join(base for base in sequence if base in "ACGT")

def count_nucleotides(genome_sequence, n):
    """Count the occurrences of each n-nucleotide in the given genome sequence and its reverse complement."""
    nucleotide_counts = {}
    total_nucleotides = 0
    genome_sequence = remove_non_standard_bases(genome_sequence)
    sequences = [genome_sequence, reverse_complement(genome_sequence)]  # óºçΩÇä‹ÇﬂÇÈ
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            nucleotide = seq[i:i+n]
            if set(nucleotide) <= set("ACGT"):  # Ensure only ACGT characters are counted
                total_nucleotides += 1
                if nucleotide in nucleotide_counts:
                    nucleotide_counts[nucleotide] += 1
                else:
                    nucleotide_counts[nucleotide] = 1
    return nucleotide_counts, total_nucleotides

def process_fna_file(file_path, n):
    """Process the .fna or .fasta file and return the relative frequency of n-nucleotides."""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            current_sequence = []
            nucleotide_counts = {}
            total_nucleotides = 0
            for line in lines:
                if line.startswith('>'):
                    if current_sequence:
                        sequence = ''.join(current_sequence)
                        counts, count_total = count_nucleotides(sequence, n)
                        for nucleotide, count in counts.items():
                            if nucleotide in nucleotide_counts:
                                nucleotide_counts[nucleotide] += count
                            else:
                                nucleotide_counts[nucleotide] = count
                        total_nucleotides += count_total
                        current_sequence = []
                else:
                    current_sequence.append(line.strip().upper())
            
            if current_sequence:
                sequence = ''.join(current_sequence)
                counts, count_total = count_nucleotides(sequence, n)
                for nucleotide, count in counts.items():
                    if nucleotide in nucleotide_counts:
                        nucleotide_counts[nucleotide] += count
                    else:
                        nucleotide_counts[nucleotide] = count
                total_nucleotides += count_total

            # Normalize counts to relative frequencies
            for nucleotide in nucleotide_counts:
                nucleotide_counts[nucleotide] /= total_nucleotides

            return nucleotide_counts
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}

def process_files(input_folder, output_file, n):
    """Process all files in the input folder and write the nucleotide counts to the output file."""
    all_nucleotides = generate_all_nucleotides(n)
    file_nucleotide_counts = {}
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".fasta") or file_name.endswith(".fna"):
            file_path = os.path.join(input_folder, file_name)
            nucleotide_counts = process_fna_file(file_path, n)
            file_nucleotide_counts[file_name] = nucleotide_counts
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'{n}-Nucleotide'] + list(file_nucleotide_counts.keys()))
        for nucleotide in all_nucleotides:
            row = [nucleotide] + [file_nucleotide_counts[file_name].get(nucleotide, 0) for file_name in file_nucleotide_counts.keys()]
            writer.writerow(row)

def transpose_csv(input_file, output_file):
    """Transpose the CSV file."""
    df = pd.read_csv(input_file, header=None)
    df_transposed = df.T
    df_transposed.to_csv(output_file, index=False, header=False)

def main(input_folder, output_file, n):
    """Main function to process files and write transposed output."""
    intermediate_file = "intermediate_file.csv"
    process_files(input_folder, intermediate_file, n)
    transpose_csv(intermediate_file, output_file)
    os.remove(intermediate_file)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python k-merfreq_both.py <input_folder> <output_file> <n>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_file = sys.argv[2]
    n = int(sys.argv[3])
    main(input_folder, output_file, n)
