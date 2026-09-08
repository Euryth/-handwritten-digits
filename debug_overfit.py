import numpy as np

from data_loader import load_dataset, one_hot_encode
from src.layers import Dense
from src.activations import ReLU
from src.losses import SoftmaxCrossEntropyLoss
from src.optimizers import SGD
from src.network import Sequential

np.random.seed(0)

print("Loading dataset (from cache)...")
X, y = load_dataset()
y_onehot = one_hot_encode(y)

# --- Check 1: are classes actually distinguishable? ---
print("\n--- Class separability check ---")
for digit in range(10):
    class_mean = X[y == digit].mean(axis=0)
    print(f"Digit {digit}: mean pixel value = {class_mean.mean():.4f}, "
          f"std across pixels = {class_mean.std():.4f}")

overall_variance = X.var(axis=0).mean()
print(f"Average per-pixel variance across ALL images: {overall_variance:.6f}")

# --- Check 2: can the network overfit a tiny subset? ---
print("\n--- Tiny overfit check (20 samples, no dropout, no momentum) ---")
n_small = 20
X_small = X[:n_small]
y_small = y_onehot[:n_small]

net = Sequential([
    Dense(784, 64, seed=1),
    ReLU(),
    Dense(64, 10, seed=2),
])
loss_fn = SoftmaxCrossEntropyLoss()
optimizer = SGD(layers=net.get_trainable_layers(), lr=0.05, momentum=0.0)

for step in range(300):
    logits = net.forward(X_small)
    loss = loss_fn.forward(logits, y_small)
    grad = loss_fn.backward()
    net.backward(grad)
    optimizer.step()

    if step % 50 == 0 or step == 299:
        preds = np.argmax(logits, axis=1)
        true = np.argmax(y_small, axis=1)
        acc = np.mean(preds == true)
        print(f"Step {step}: loss={loss:.4f}, train accuracy on these 20 samples={acc:.2f}")