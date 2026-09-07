import numpy as np
from src.losses import SoftmaxCrossEntropyLoss

np.random.seed(0)

loss_fn = SoftmaxCrossEntropyLoss()

batch_size, num_classes = 4, 3
logits = np.random.randn(batch_size, num_classes)
y_true = np.zeros((batch_size, num_classes))
y_true[np.arange(batch_size), np.random.randint(0, num_classes, batch_size)] = 1.0

loss = loss_fn.forward(logits, y_true)
print("Loss value:", loss)
assert loss > 0, "Cross-entropy loss must be positive"

grad_logits = loss_fn.backward()
print("Gradient shape:", grad_logits.shape)
assert grad_logits.shape == logits.shape

# Numerical gradient checking
eps = 1e-5
numerical_grad = np.zeros_like(logits)
for i in range(logits.shape[0]):
    for j in range(logits.shape[1]):
        logits_plus = logits.copy()
        logits_plus[i, j] += eps
        loss_plus = loss_fn.forward(logits_plus, y_true)

        logits_minus = logits.copy()
        logits_minus[i, j] -= eps
        loss_minus = loss_fn.forward(logits_minus, y_true)

        numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

# restore state after perturbation checks
loss_fn.forward(logits, y_true)

diff = np.abs(numerical_grad - grad_logits).max()
print("Gradient check diff:", diff)
assert diff < 1e-4, "Backward pass is INCORRECT!"

print("PASS: SoftmaxCrossEntropyLoss forward/backward correct")