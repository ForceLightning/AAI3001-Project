import pandas as pd
import matplotlib.pyplot as plt

# Read the results from the CSV file
df = pd.read_csv("results.csv")

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot mean accuracy
for alpha in df['Alpha'].unique():
    subset = df[df['Alpha'] == alpha]
    ax1.plot(subset['Epsilon'], subset['Mean'], label=f'Alpha={alpha}')

ax1.set_xlabel('Epsilon')
ax1.set_ylabel('Mean Pixel Accuracy')
ax1.legend(title='Alpha')
ax1.set_title('Epsilon vs Mean Pixel Accuracy for different Alpha values')

# Plot standard deviation
for alpha in df['Alpha'].unique():
    subset = df[df['Alpha'] == alpha]
    ax2.plot(subset['Epsilon'], subset['Standard Deviation'], label=f'Alpha={alpha}')

ax2.set_xlabel('Epsilon')
ax2.set_ylabel('Standard Deviation')
ax2.legend(title='Alpha')
ax2.set_title('Epsilon vs Standard Deviation for different Alpha values')

plt.show()
