from sklearn.datasets import make_moons
import numpy as np
import sys
import os

# --- Configuration ---
# Default total number of samples
n_samples_total = 100000 
if len(sys.argv) > 1:
    # Use the first command line argument as the total number of samples
    n_samples_total = int(sys.argv[1])

RANDOM_STATE = 3 # For reproducibility
NOISE_LEVEL = 0.08 # A good challenge level

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Generate the Moons dataset
print(f"Generating {n_samples_total} total samples...")
X, y = make_moons(n_samples_total, noise=NOISE_LEVEL, random_state=RANDOM_STATE)

# 2. Save the complete data set
X_FILENAME = os.path.join(OUTPUT_DIR, "X_data.txt")
Y_FILENAME = os.path.join(OUTPUT_DIR, "y_data.txt")

np.savetxt(X_FILENAME, X)
np.savetxt(Y_FILENAME, y, fmt="%d")

print("\n--- Dataset Summary ---")
print(f"Total samples generated: {n_samples_total}")
print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")

print(f"\n[SUCCESS] Data saved to {X_FILENAME} and {Y_FILENAME}.")
print(f"Your C program will now load these files and split them internally.")
