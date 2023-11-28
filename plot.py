import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the CSV files
models_dir = './models/'

# List all CSV files in the directory
csv_files = [file for file in os.listdir(models_dir) if file.endswith('.csv')]

# Plotting training and validation loss for each file
plt.figure(figsize=(12, 5))

for csv_file in csv_files:
    # Load the CSV file
    df = pd.read_csv(os.path.join(models_dir, csv_file))

    # Extract the fold number from the file name
    fold_number = int(csv_file.split('_')[1].split('.')[0])

    # Plotting
    plt.plot(df['epoch'], df['train_loss'], label=f'Training Loss (Fold {fold_number})', marker='o')
    plt.plot(df['epoch'], df['valid_loss'], label=f'Validation Loss (Fold {fold_number})', marker='o')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()
