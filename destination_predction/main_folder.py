import os
import subprocess
import sys

def run_for_all_fasta_files(input_directory, output_directory):
    """
    Run the 'predict_main.py' script for all FASTA files in the input directory.

    Parameters:
        input_directory: Path to the directory containing input FASTA files.
        output_directory: Path to the directory where output files will be saved.
    """
    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".fasta"):  # Process only files with .fasta extension
            fasta_file = os.path.join(input_directory, filename)  # Full path to the FASTA file
            base_name = os.path.splitext(filename)[0]  # Extract the base name (without extension)
            final_output_file = f"{base_name}_output.csv"  # Create output file name
            
            print(f"Processing {fasta_file}...")  # Log the file being processed
            
            # Run the 'predict_main.py' script with the current FASTA file and output configuration
            subprocess.run([
                'python', 'predict_main.py', 
                fasta_file, 
                output_directory, 
                final_output_file
            ])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python batch_script.py <input_directory> <output_directory>")
        sys.exit(1)

    # Parse command-line arguments
    input_directory = sys.argv[1]  # Directory containing FASTA files
    output_directory = sys.argv[2]  # Directory to save output files

    # Run the batch processing
    run_for_all_fasta_files(input_directory, output_directory)
