import numpy as np
import pandas as pd

# Define file paths for input data
rows_file = 'chr127_3_dra_both.csv'  # CSV file containing numerical data (127 rows, variable columns)
cols_file = 'plasmid3_both.csv'      # CSV file containing numerical data (14 rows, variable columns)
labels_file = 'label2.csv'           # CSV file containing labels for a 14×127 matrix

# Load numerical data from rows and columns into DataFrames
rows_df = pd.read_csv(rows_file, header=0, index_col=0)  # Rows data with headers
cols_df = pd.read_csv(cols_file, header=0, index_col=0)  # Columns data with headers

# Dynamically obtain the number of features (columns) from the rows data
num_features = rows_df.shape[1]  # Number of numerical columns in the rows file

# Extract column headers for the numerical data from rows_df
rows_header = rows_df.columns.values  # List of column names (used later for the final CSV header)

# Load numerical data as numpy arrays for calculations
rows_256_sequences = rows_df.values  # Convert rows data to a numpy array (shape: 127 × num_features)
cols_256_sequences = cols_df.values  # Convert columns data to a numpy array (shape: 14 × num_features)

# Load the label data as a numpy array
labels = pd.read_csv(labels_file, header=None).values  # Labels as a numpy array (shape: 14 × 127)

# Display the shapes of the loaded data for verification
print("Shape of rows_256_sequences:", rows_256_sequences.shape)  # (127, num_features)
print("Shape of cols_256_sequences:", cols_256_sequences.shape)  # (14, num_features)
print("Shape of labels:", labels.shape)  # (14, 127)

# Repeat the rows data vertically to match the label and column structure
rows_repeated = np.tile(rows_256_sequences, (14, 1, 1))  # Repeat along the first dimension (shape: 14 × 127 × num_features)

# Repeat the columns data horizontally to match the label and row structure
cols_repeated = np.tile(cols_256_sequences[:, np.newaxis, :], (1, 127, 1))  # Add new axis and repeat (shape: 14 × 127 × num_features)

# Display the shapes of the repeated data for verification
print("Shape of rows_repeated:", rows_repeated.shape)  # (14, 127, num_features)
print("Shape of cols_repeated:", cols_repeated.shape)  # (14, 127, num_features)

# Perform element-wise subtraction between rows and columns data (row - col)
data = rows_repeated - cols_repeated  # Resulting shape: (14, 127, num_features)

# Concatenate labels as an additional feature to the data
# Expand the labels array along the third dimension to match data shape
data_with_labels = np.concatenate((data, labels[:, :, np.newaxis]), axis=2)  # Final shape: (14, 127, num_features + 1)

# Flatten the 3D array to 2D for saving to a CSV file
data_with_labels_flattened = data_with_labels.reshape(14 * 127, num_features + 1)  # Flattened shape: (14*127 × num_features+1)

# Convert the flattened data with labels into a pandas DataFrame
df_with_labels = pd.DataFrame(data_with_labels_flattened)  # Create a DataFrame for easy CSV writing

# Prepare the header for the final CSV file by combining feature names with 'Label'
header = list(rows_header) + ['Label']  # Column names from rows_header and a 'Label' column

# Save the resulting DataFrame to a CSV file
df_with_labels.to_csv('data_with_labels32.csv', index=False, header=header)  # Write the CSV with the header

# Display the shape of the final data for verification
print("Data with labels shape:", data_with_labels_flattened.shape)  # Final shape should be (14*127, num_features+1)
