import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Load data from a CSV file into a DataFrame
# 'chr126_k_dra_bothmod.csv' is assumed to contain rows as samples and columns as features
# "k indicates the number of consecutive nucleotides."
data_file = 'chr126_k_dra_bothmod.csv'
df = pd.read_csv(data_file, index_col=0)  # Load CSV, set the first column as row index

# Perform hierarchical clustering on rows
# 'Z' is the linkage matrix, which stores hierarchical clustering information
# Use 'average' linkage method and Euclidean distance metric
Z = linkage(df, method='average', metric='euclidean')

# Set up the figure for the dendrogram plot
plt.figure(figsize=(10, 12))  # Set figure size (width, height) in inches

# Generate dendrogram from the linkage matrix 'Z'
dendrogram(
    Z,
    labels=df.index,       # Use row labels from the DataFrame as dendrogram labels
    leaf_rotation=90       # Rotate labels by 90 degrees for better readability
)

# Adjust font size of x-axis labels for clarity, especially with many labels
ax = plt.gca()  # Get the current Axes instance
for label in ax.get_xmajorticklabels():
    label.set_fontsize(4)  # Set font size of x-axis labels to 4 for compactness

# Add plot title and axis labels for context
plt.title('Dendrogram of Hierarchical Clustering')  # Title of the dendrogram
plt.xlabel('Sample Index')                         # Label for x-axis
plt.ylabel('Distance')                             # Label for y-axis

# Save the dendrogram plot as a PNG file with high resolution
output_file = 'dendrogramkchr.png'  # Specify file name for saved plot
plt.savefig(output_file, format='png', dpi=300)  # Save as PNG with 300 DPI
plt.show()  # Display the plot in the output

# Print a message indicating the saved file path
print(f'Dendrogram saved as {output_file}')


  
