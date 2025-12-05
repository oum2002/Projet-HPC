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
