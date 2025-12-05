#!/bin/bash
################################################################################
# STEP 3: Mini-batch Gradient Descent Testing
# Compare full-batch (original) vs mini-batch gradient descent
################################################################################

set -e

echo "=================================================================="
echo "STEP 3: Mini-batch Gradient Descent Testing"
echo "=================================================================="
echo "Start time: $(date)"
echo ""

# Create directories
mkdir -p results/step3_minibatch
mkdir -p logs

# Generate dataset (10,000 samples)
echo "Generating dataset (10,000 samples)..."
python3 generate_moon.py 10000 0.20 3
echo "✓ Dataset generated"
echo ""

# Compile
echo "Compiling code..."
make clean && make
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi
echo "✓ Compilation successful"
echo ""

################################################################################
# Test 1: Full-batch (simulate by using very large batch size)
################################################################################
echo "=================================================================="
echo "Test 1: Full-batch Gradient Descent (baseline)"
echo "=================================================================="

echo "Running with batch_size=10000 (full-batch)..."
{ time ./mlp leaky_relu --threads 4 --batch-size 10000 2>&1; } | tee logs/fullbatch_output.txt

# Extract results
fullbatch_time=$(grep "Training completed" logs/fullbatch_output.txt | awk '{print $4}')
fullbatch_loss=$(grep "Epoch 19900" logs/fullbatch_output.txt | tail -1 | awk '{print $3}' | tr -d ',')

echo ""
echo "Full-batch Results:"
echo "  Time: ${fullbatch_time}s"
echo "  Final Loss: ${fullbatch_loss}"
echo ""

################################################################################
# Test 2: Mini-batch with different sizes
################################################################################
echo "=================================================================="
echo "Test 2: Mini-batch Gradient Descent"
echo "=================================================================="

batch_sizes=(8 16 32 64 128 256)
batch_times=()
batch_losses=()
batch_epochs=()

for bs in "${batch_sizes[@]}"; do
    echo "----------------------------------------------------------"
    echo "Testing batch_size=$bs..."
    
    output_file="logs/batch_${bs}_output.txt"
    { time ./mlp leaky_relu --threads 4 --batch-size $bs 2>&1; } | tee $output_file
    
    # Extract metrics
    time_val=$(grep "Training completed" $output_file | awk '{print $4}')
    loss_val=$(grep "Epoch 19900" $output_file | tail -1 | awk '{print $3}' | tr -d ',')
    
    # Count updates per epoch (num_samples / batch_size)
    updates_per_epoch=$((10000 / bs))
    
    batch_times+=($time_val)
    batch_losses+=($loss_val)
    batch_epochs+=($updates_per_epoch)
    
    echo "  Time: ${time_val}s"
    echo "  Final Loss: ${loss_val}"
    echo "  Updates per epoch: ${updates_per_epoch}"
    echo ""
done

################################################################################
# Save results to JSON
################################################################################
echo "=================================================================="
echo "Saving Results"
echo "=================================================================="

# Create JSON file
cat > results/step3_minibatch/results.json << EOF
{
  "description": "Mini-batch vs Full-batch Gradient Descent Comparison",
  "dataset_size": 10000,
  "epochs": 20000,
  "fullbatch": {
    "batch_size": 10000,
    "time": ${fullbatch_time},
    "final_loss": ${fullbatch_loss},
    "updates_per_epoch": 1
  },
  "minibatch": {
    "batch_sizes": [$(IFS=,; echo "${batch_sizes[*]}")],
    "times": [$(IFS=,; echo "${batch_times[*]}")],
    "final_losses": [$(IFS=,; echo "${batch_losses[*]}")],
    "updates_per_epoch": [$(IFS=,; echo "${batch_epochs[*]}")]
  }
}
EOF

echo "✓ Results saved to: results/step3_minibatch/results.json"
echo ""

################################################################################
# Generate comparison plot
################################################################################
echo "=================================================================="
echo "Generating Comparison Plot"
echo "=================================================================="

cat > plot_step3.py << 'PLOTSCRIPT'
#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('results/step3_minibatch/results.json', 'r') as f:
    data = json.load(f)

fullbatch = data['fullbatch']
minibatch = data['minibatch']

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Add full-batch as first point
batch_sizes_all = [fullbatch['batch_size']] + minibatch['batch_sizes']
times_all = [fullbatch['time']] + minibatch['times']
losses_all = [fullbatch['final_loss']] + minibatch['final_losses']
updates_all = [fullbatch['updates_per_epoch']] + minibatch['updates_per_epoch']

# Plot 1: Training Time vs Batch Size
ax1.plot(batch_sizes_all, times_all, 'o-', linewidth=2, markersize=10, color='#2E86AB')
ax1.axvline(x=fullbatch['batch_size'], color='red', linestyle='--', alpha=0.5, label='Full-batch')
ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax1.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Training Time vs Batch Size', fontsize=14, fontweight='bold')
ax1.set_xscale('log', base=2)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Find and mark optimal
optimal_idx = np.argmin(times_all)
ax1.plot(batch_sizes_all[optimal_idx], times_all[optimal_idx], 'r*', 
         markersize=20, label=f'Optimal: {batch_sizes_all[optimal_idx]}')

# Plot 2: Final Loss vs Batch Size
ax2.plot(batch_sizes_all, losses_all, 's-', linewidth=2, markersize=10, color='#A23B72')
ax2.axvline(x=fullbatch['batch_size'], color='red', linestyle='--', alpha=0.5, label='Full-batch')
ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax2.set_ylabel('Final Loss', fontsize=12, fontweight='bold')
ax2.set_title('Convergence Quality vs Batch Size', fontsize=14, fontweight='bold')
ax2.set_xscale('log', base=2)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Updates per Epoch vs Batch Size
ax3.plot(batch_sizes_all, updates_all, '^-', linewidth=2, markersize=10, color='#06A77D')
ax3.axvline(x=fullbatch['batch_size'], color='red', linestyle='--', alpha=0.5, label='Full-batch')
ax3.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax3.set_ylabel('Weight Updates per Epoch', fontsize=12, fontweight='bold')
ax3.set_title('Update Frequency vs Batch Size', fontsize=14, fontweight='bold')
ax3.set_xscale('log', base=2)
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.savefig('results/step3_minibatch/comparison_plot.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: results/step3_minibatch/comparison_plot.png")

# Print summary
print("\n" + "="*60)
print("SUMMARY - Mini-batch vs Full-batch")
print("="*60)
print(f"\nFull-batch (baseline):")
print(f"  Batch size: {fullbatch['batch_size']}")
print(f"  Time: {fullbatch['time']:.2f}s")
print(f"  Final loss: {fullbatch['final_loss']:.6f}")
print(f"  Updates/epoch: {fullbatch['updates_per_epoch']}")

print(f"\nBest Mini-batch:")
best_idx = optimal_idx - 1 if optimal_idx > 0 else 0
if best_idx >= 0 and best_idx < len(minibatch['batch_sizes']):
    print(f"  Batch size: {minibatch['batch_sizes'][best_idx]}")
    print(f"  Time: {minibatch['times'][best_idx]:.2f}s")
    print(f"  Final loss: {minibatch['final_losses'][best_idx]:.6f}")
    print(f"  Updates/epoch: {minibatch['updates_per_epoch'][best_idx]}")
    
    speedup = fullbatch['time'] / minibatch['times'][best_idx]
    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {fullbatch['time'] - minibatch['times'][best_idx]:.2f}s")

print("\n" + "="*60)
PLOTSCRIPT

chmod +x plot_step3.py
python3 plot_step3.py

echo ""
echo "=================================================================="
echo "STEP 3 COMPLETE"
echo "=================================================================="
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - JSON: results/step3_minibatch/results.json"
echo "  - Plot: results/step3_minibatch/comparison_plot.png"
echo "  - Logs: logs/batch_*_output.txt"
echo ""
echo "Next step: Run step4_test_activations.sh"
echo "=================================================================="
