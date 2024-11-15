import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Generate a violin plot from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')  # Input CSV file path
    parser.add_argument('output_file', type=str, help='Path to save the output SVG file')  # Output SVG file path
    args = parser.parse_args()

    # Load the CSV file into a DataFrame
    df = pd.read_csv(args.csv_file)  # Read CSV data into a pandas DataFrame

    # Create a violin plot
    plt.figure(figsize=(10, 6))  # Set figure size (width, height) in inches
    sns.violinplot(x='results', y='log3MD', data=df)  # Create violin plot with 'results' on x-axis and 'log3MD' on y-axis

    # Customize plot appearance
    plt.title('Violin Plot from CSV Data')  # Set the title of the plot
    plt.xlabel('results')                   # Label for x-axis
    plt.ylabel('3freq')                     # Label for y-axis

    # Save the plot as an SVG file
    plt.savefig(args.output_file, format='svg')  # Save the plot to the specified file path in SVG format

    # Show the plot (optional)
    plt.show()  # Display the plot on the screen

if __name__ == '__main__':
    main()  # Call the main function when the script is run
