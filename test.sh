#!/bin/bash

echo "======================================"
echo "Quick Test: Batch vs Mini-batch GD"
echo "======================================"
echo ""

# Create directories
mkdir -p data output

# Generate small dataset for quick testing
echo "Generating dataset (1000 samples)..."
python3 << EOF
from sklearn.datasets import make_moons
import numpy as np

X, y = make_moons(1000, noise=0.20, random_state=3)
np.savetxt("data/data_X.txt", X)
np.savetxt("data/data_y.txt", y, fmt="%d")
print("Dataset generated: 1000 samples")
EOF

# Compile
echo ""
echo "Compiling..."
gcc -Wall -O2 -o nn_train main.c model.c utils.c -lm

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# Run batch GD
echo "======================================"
echo "Running BATCH GD (100 iterations)..."
echo "======================================"
./nn_train batch | head -15

echo ""
echo "======================================"
echo "Running MINI-BATCH GD (100 epochs, batch_size=32)..."
echo "======================================"
./nn_train minibatch 32 | head -15

echo ""
echo "======================================"
echo "Comparing final losses..."
echo "======================================"

# Extract final losses
if [ -f "output/benchmark_batch.txt" ]; then
    batch_loss=$(tail -1 output/benchmark_batch.txt | cut -d',' -f2)
    batch_time=$(tail -1 output/benchmark_batch.txt | cut -d',' -f3)
    echo "Batch GD:"
    echo "  Final loss: $batch_loss"
    echo "  Time: $batch_time seconds"
fi

if [ -f "output/benchmark_minibatch.txt" ]; then
    mini_loss=$(tail -1 output/benchmark_minibatch.txt | cut -d',' -f2)
    mini_time=$(tail -1 output/benchmark_minibatch.txt | cut -d',' -f3)
    echo ""
    echo "Mini-batch GD:"
    echo "  Final loss: $mini_loss"
    echo "  Time: $mini_time seconds"
fi

echo ""
echo "======================================"
echo "Key Differences:"
echo "======================================"
echo ""
echo "1. Mini-batch processes data in small chunks (32 samples)"
echo "2. Mini-batch updates weights MORE frequently (313 updates vs 100)"
echo "3. Mini-batch shuffles data each epoch for better generalization"
echo "4. Mini-batch converges faster in wall-clock time"
echo ""
echo "Visualize results with: python3 compare_methods.py"
echo ""
