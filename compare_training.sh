#!/bin/bash

# Script to compare full-batch vs mini-batch training

echo "=========================================="
echo "Training Comparison Script"
echo "=========================================="

# Compile the program
echo "Compiling..."
gcc -o mlp main.c model.c utils.c -lm -O2
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Generate data
echo "Generating dataset..."
python3 generate_moon.py

# Create output directory
mkdir -p output

echo ""
echo "=========================================="
echo "Test 1: Full-Batch Training"
echo "=========================================="
time ./mlp > logs_fullbatch.txt

# Save results
cp output/W1.txt output/W1_fullbatch.txt
cp output/W2.txt output/W2_fullbatch.txt
cp output/b1.txt output/b1_fullbatch.txt
cp output/b2.txt output/b2_fullbatch.txt

echo ""
echo "=========================================="
echo "Test 2: Mini-Batch (size=32)"
echo "=========================================="
time ./mlp --minibatch 32 > logs_minibatch32.txt

# Save results
cp output/W1.txt output/W1_minibatch32.txt
cp output/W2.txt output/W2_minibatch32.txt
cp output/b1.txt output/b1_minibatch32.txt
cp output/b2.txt output/b2_minibatch32.txt

echo ""
echo "=========================================="
echo "Test 3: Mini-Batch (size=16)"
echo "=========================================="
time ./mlp --minibatch 16 > logs_minibatch16.txt

# Save results
cp output/W1.txt output/W1_minibatch16.txt
cp output/W2.txt output/W2_minibatch16.txt
cp output/b1.txt output/b1_minibatch16.txt
cp output/b2.txt output/b2_minibatch16.txt

echo ""
echo "=========================================="
echo "Test 4: Mini-Batch (size=64)"
echo "=========================================="
time ./mlp --minibatch 64 > logs_minibatch64.txt

# Save results
cp output/W1.txt output/W1_minibatch64.txt
cp output/W2.txt output/W2_minibatch64.txt
cp output/b1.txt output/b1_minibatch64.txt
cp output/b2.txt output/b2_minibatch64.txt

echo ""
echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo "Log files created:"
echo "  - logs_fullbatch.txt"
echo "  - logs_minibatch32.txt"
echo "  - logs_minibatch16.txt"
echo "  - logs_minibatch64.txt"
echo ""
echo "Extract final losses:"
grep "Loss" logs_fullbatch.txt | tail -5
echo "---"
grep "Loss" logs_minibatch32.txt | tail -5
echo "---"
grep "Loss" logs_minibatch16.txt | tail -5
echo "---"
grep "Loss" logs_minibatch64.txt | tail -5
