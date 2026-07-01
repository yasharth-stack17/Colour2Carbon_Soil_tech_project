import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set visual style
sns.set_theme(style="whitegrid", context="talk")

file_path = 'soil_RGB_SOC_data.csv'
print(f"Loading dataset from {file_path}...")
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.\n")
except FileNotFoundError:
    print(f"Error: Could not find {file_path}.")
    exit()

# 1. Advanced Basic Info
print("--- Advanced Basic Information ---")
print(f"Total Records: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
print("\n")

print("Generating detailed plots...")

# 2. Distribution and Boxplots (Outlier Detection)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
features = ['R', 'G', 'B', 'SOC']
colors = ['red', 'green', 'blue', 'brown']

for i, (col, color) in enumerate(zip(features, colors)):
    # Histograms with KDE
    sns.histplot(df[col], kde=True, bins=50, color=color, ax=axes[0, i])
    axes[0, i].set_title(f'Distribution of {col}')
    
    # Boxplots for outliers
    sns.boxplot(x=df[col], color=color, ax=axes[1, i])
    axes[1, i].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.savefig('detailed_distributions_and_outliers.png')
print("Saved 'detailed_distributions_and_outliers.png'")
plt.close()

# 3. Feature Engineering & Density Analysis
# Create a derived feature 'Brightness'
df['Brightness'] = (df['R'] + df['G'] + df['B']) / 3.0

plt.figure(figsize=(10, 6))
sns.histplot(df['Brightness'], kde=True, bins=50, color='orange')
plt.title('Distribution of Derived Feature: Brightness')
plt.savefig('brightness_distribution.png')
print("Saved 'brightness_distribution.png'")
plt.close()

# 4. High-Density Scatter Analysis (Hexbin)
# For 50,000 points, hexbin is much better than regular scatter
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

hb1 = axes[0].hexbin(df['R'], df['SOC'], gridsize=40, cmap='Reds', mincnt=1)
axes[0].set_xlabel('Red')
axes[0].set_ylabel('SOC')
axes[0].set_title('Density: Red vs SOC')
fig.colorbar(hb1, ax=axes[0])

hb2 = axes[1].hexbin(df['G'], df['SOC'], gridsize=40, cmap='Greens', mincnt=1)
axes[1].set_xlabel('Green')
axes[1].set_ylabel('SOC')
axes[1].set_title('Density: Green vs SOC')
fig.colorbar(hb2, ax=axes[1])

hb3 = axes[2].hexbin(df['B'], df['SOC'], gridsize=40, cmap='Blues', mincnt=1)
axes[2].set_xlabel('Blue')
axes[2].set_ylabel('SOC')
axes[2].set_title('Density: Blue vs SOC')
fig.colorbar(hb3, ax=axes[2])

plt.tight_layout()
plt.savefig('high_density_hexbin_SOC_vs_RGB.png')
print("Saved 'high_density_hexbin_SOC_vs_RGB.png'")
plt.close()

# 5. Brightness vs SOC
plt.figure(figsize=(10, 6))
plt.hexbin(df['Brightness'], df['SOC'], gridsize=50, cmap='inferno', mincnt=1)
plt.colorbar(label='Count in bin')
plt.xlabel('Brightness (Average of R, G, B)')
plt.ylabel('SOC (Soil Organic Carbon)')
plt.title('Density: Brightness vs SOC')
plt.savefig('brightness_vs_SOC_hexbin.png')
print("Saved 'brightness_vs_SOC_hexbin.png'")
plt.close()

# 6. Detailed Correlation Matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
# Use a mask to only show the lower triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".3f", linewidths=.5)
plt.title('Detailed Correlation Matrix')
plt.tight_layout()
plt.savefig('detailed_correlation_matrix.png')
print("Saved 'detailed_correlation_matrix.png'")
plt.close()

print("\nDetailed EDA completed. All new high-resolution graphs have been saved.")
