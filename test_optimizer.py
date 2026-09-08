import numpy as np
from src.layers import Dense
from src.optimizers import SGD

np.random.seed(0)

layer = Dense(in_features=3, out_features=2, seed=42)
W_before = layer.W.copy()
b_before = layer.b.copy()

# simulate a forward/backward pass to populate gradients
x = np.random.randn(5, 3)
out = layer.forward(x)
grad_output = np.random.randn(5, 2)
layer.backward(grad_output)

optimizer = SGD(layers=[layer], lr=0.1)
optimizer.step()

expected_W = W_before - 0.1 * layer.grad_W
expected_b = b_before - 0.1 * layer.grad_b

assert np.allclose(layer.W, expected_W), "SGD update for W is incorrect"
assert np.allclose(layer.b, expected_b), "SGD update for b is incorrect"
print("PASS: SGD (no momentum) updates weights correctly")

# --- test with momentum ---
layer2 = Dense(in_features=3, out_features=2, seed=42)
optimizer2 = SGD(layers=[layer2], lr=0.1, momentum=0.9)

for step in range(3):
    out = layer2.forward(x)
    layer2.backward(grad_output)
    optimizer2.step()

print("PASS: SGD with momentum ran 3 steps without error")
print("Final W sample:", layer2.W[0, :2])