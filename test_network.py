import numpy as np
from src.layers import Dense
from src.activations import ReLU
from src.losses import SoftmaxCrossEntropyLoss
from src.optimizers import SGD
from src.network import Sequential

np.random.seed(0)

# tiny synthetic dataset: 2 classes, 4 samples
X = np.random.randn(8, 5)
y_true = np.zeros((8, 2))
y_true[np.arange(8), np.random.randint(0, 2, 8)] = 1.0

net = Sequential([
    Dense(5, 8, seed=1),
    ReLU(),
    Dense(8, 2, seed=2),
])

loss_fn = SoftmaxCrossEntropyLoss()
optimizer = SGD(layers=net.get_trainable_layers(), lr=0.1)

losses = []
for step in range(50):
    logits = net.forward(X)
    loss = loss_fn.forward(logits, y_true)
    losses.append(loss)

    grad = loss_fn.backward()
    net.backward(grad)
    optimizer.step()

print("Loss at step 0:", losses[0])
print("Loss at step 49:", losses[-1])
assert losses[-1] < losses[0], "Loss did not decrease - something is wrong in the pipeline!"
print("PASS: Full pipeline (Dense -> ReLU -> Dense -> Loss -> SGD) trains correctly")