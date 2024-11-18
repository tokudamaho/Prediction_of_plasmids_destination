import subprocess
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import mode

def run_prediction_scripts(input_fasta_file, output_directory):
    """
    Run a series of prediction scripts sequentially.

    Parameters:
        input_fasta_file: Path to the input FASTA file.
        output_directory: Directory to save the prediction results.
    """
    # List of prediction scripts to execute
    script_names = [
        'script/prediction2_tune.py',
        'script/prediction3_tune.py',
        'script/prediction4_tune.py',
        'script/prediction5_tune.py',
        'script/prediction6_tune.py',
        'script/prediction7_tune.py',
        'script/prediction8_tune.py'
    ]
    
    # Execute each script
    for script in script_names:
        print(f"Running {script}")
        subprocess.run(['python', script, input_fasta_file, output_directory])

def concatenate_and_calculate_mode(output_directory, final_output_file):
    """
    Concatenate prediction results from multiple scripts and calculate the mode.

    Parameters:
        output_directory: Directory containing the prediction results.
        final_output_file: Name of the final output file to save.
    """
    # Paths to the prediction CSV files
    prediction_files = [
        os.path.join(output_directory, 'Predictions2.csv'),
        os.path.join(output_directory, 'Predictions3.csv'),
        os.path.join(output_directory, 'Predictions4.csv'),
        os.path.join(output_directory, 'Predictions5.csv'),
        os.path.join(output_directory, 'Predictions6.csv'),
        os.path.join(output_directory, 'Predictions7.csv'),
        os.path.join(output_directory, 'Predictions8.csv')
    ]
    
    # Read each CSV file and store in a list of DataFrames
    data_frames = [pd.read_csv(file, header=0) for file in prediction_files]
    
    # Concatenate all DataFrames column-wise
    combined_df = pd.concat(data_frames, axis=1)
    
    # Calculate the mode (most frequent value) for each row
    combined_df['Most_Frequent'] = combined_df.mode(axis=1)[0]
    
    # Save the concatenated DataFrame with mode calculation
    final_output_path = os.path.join(output_directory, final_output_file)
    combined_df.to_csv(final_output_path, index=False)
    print(f"Final concatenated file with mode written to {final_output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main_script.py <input_fasta_file> <output_directory> <final_output_file>")
        sys.exit(1)

    # Parse command-line arguments
    input_fasta_file = sys.argv[1]  # Path to the input FASTA file
    output_directory = sys.argv[2]  # Directory to save prediction results
    final_output_file = sys.argv[3]  # Name of the final output file
    
    # Step 1: Run all prediction scripts
    run_prediction_scripts(input_fasta_file, output_directory)
    
    # Step 2: Concatenate results and calculate the mode
    concatenate_and_calculate_mode(output_directory, final_output_file)
