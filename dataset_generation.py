import numpy as np
import pandas as pd

print("Step 1: Generating 60,000 realistic soil samples...")
np.random.seed(17) 
num_samples = 50000

# Simulate realistic normalized RGB values (0.0 to 1.0)
# Darker soils (lower RGB) generally have higher organic carbon (SOC)
R = np.random.uniform(0.15, 0.8, num_samples)
G = R * np.random.uniform(0.8, 0.95, num_samples)  # Soil is brownish/yellowish
B = G * np.random.uniform(0.7, 0.85, num_samples)

# Calculate Soil Organic Carbon (SOC) based on color darkness
brightness = (R + G + B) / 3.0
SOC_Percentage = 5.0 - (brightness * 6.0) + np.random.normal(0, 0.35, num_samples)
SOC_Percentage = np.clip(SOC_Percentage, 0.05, 5.0)  # Realistic bounds

# Create DataFrame and save to CSV
df = pd.DataFrame({'R': R, 'G': G, 'B': B, 'SOC': SOC_Percentage})
df.to_csv('soil_RGB_SOC_data.csv', index=False)

print("SUCCESS: 'soil_RGB_SOC_data.csv' created locally with 50,000 rows.")
print(df.head())