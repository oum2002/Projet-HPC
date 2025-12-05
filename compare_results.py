import numpy as np
import matplotlib.pyplot as plt
import re

# Load data
X = np.loadtxt("data/data_X.txt")
y = np.loadtxt("data/data_y.txt", dtype=int)

def load_model(prefix):
    """Load model weights from files"""
    try:
        W1 = np.loadtxt(f"output/W1_{prefix}.txt")
        b1 = np.loadtxt(f"output/b1_{prefix}.txt")
        W2 = np.loadtxt(f"output/W2_{prefix}.txt")
        b2 = np.loadtxt(f"output/b2_{prefix}.txt")
        
        nn_input_dim = 2
        nn_hdim = len(b1)
        nn_output_dim = len(b2)
        
        W1 = W1.reshape((nn_input_dim, nn_hdim))
        W2 = W2.reshape((nn_hdim, nn_output_dim))
        b1 = b1.reshape((1, nn_hdim))
        b2 = b2.reshape((1, nn_output_dim))
        
        return W1, b1, W2, b2
    except:
        return None

def forward(X, W1, b1, W2, b2):
    """Forward pass"""
    a1 = np.tanh(X.dot(W1) + b1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def plot_decision_boundary(X, y, W1, b1, W2, b2, title, ax):
    """Plot decision boundary"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = forward(np.c_[xx.ravel(), yy.ravel()], W1, b1, W2, b2).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.6)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu, edgecolor='k', s=50)
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

def extract_losses(log_file):
    """Extract loss values from log file"""
    losses = []
    iterations = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'Loss' in line:
                    match = re.search(r'(\d+).*?([\d.]+)$', line)
                    if match:
                        iterations.append(int(match.group(1)))
                        losses.append(float(match.group(2)))
    except:
        pass
    return iterations, losses

# Load models
models = {
    'Full Batch': 'fullbatch',
    'Mini-Batch (32)': 'minibatch32',
    'Mini-Batch (16)': 'minibatch16',
    'Mini-Batch (64)': 'minibatch64'
}

loaded_models = {}
for name, prefix in models.items():
    model = load_model(prefix)
    if model:
        loaded_models[name] = model

if not loaded_models:
    print("No models found! Run training first.")
    exit(1)

# Create comparison plot
n_models = len(loaded_models)
fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))

if n_models == 1:
    axes = [axes]

for ax, (name, (W1, b1, W2, b2)) in zip(axes, loaded_models.items()):
    plot_decision_boundary(X, y, W1, b1, W2, b2, name, ax)

plt.tight_layout()
plt.savefig('decision_boundaries_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: decision_boundaries_comparison.png")
plt.show()

# Plot training curves
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

log_files = {
    'Full Batch': 'logs_fullbatch.txt',
    'Mini-Batch (32)': 'logs_minibatch32.txt',
    'Mini-Batch (16)': 'logs_minibatch16.txt',
    'Mini-Batch (64)': 'logs_minibatch64.txt'
}

for name, log_file in log_files.items():
    iterations, losses = extract_losses(log_file)
    if losses:
        ax.plot(iterations, losses, label=name, marker='o', markersize=3)

ax.set_xlabel('Iteration / Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: training_curves_comparison.png")
plt.show()

# Calculate accuracy for each model
print("\n" + "="*50)
print("Model Accuracy Comparison")
print("="*50)
for name, (W1, b1, W2, b2) in loaded_models.items():
    predictions = forward(X, W1, b1, W2, b2)
    accuracy = np.mean(predictions == y) * 100
    print(f"{name:20s}: {accuracy:.2f}%")
print("="*50)
