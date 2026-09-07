import numpy as np
from src.activations import ReLU, Sigmoid, Softmax

np.random.seed(0)

# --- Test ReLU ---
relu = ReLU()
x = np.array([[-2.0, 0.0, 3.0], [1.0, -1.0, 5.0]])
out = relu.forward(x)
expected = np.array([[0.0, 0.0, 3.0], [1.0, 0.0, 5.0]])
assert np.allclose(out, expected), f"ReLU forward sai: {out}"

grad_output = np.ones_like(x)
grad_input = relu.backward(grad_output)
expected_grad = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
assert np.allclose(grad_input, expected_grad), f"ReLU backward sai: {grad_input}"
print("✅ ReLU PASS")

# --- Test Sigmoid với numerical gradient checking ---
sigmoid = Sigmoid()
x = np.random.randn(4, 3)
out = sigmoid.forward(x)
assert np.all(out > 0) and np.all(out < 1), "Sigmoid output phải nằm trong (0,1)"

grad_output = np.random.randn(4, 3)
grad_input = sigmoid.backward(grad_output)

eps = 1e-5
numerical_grad = np.zeros_like(x)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x_plus = x.copy(); x_plus[i, j] += eps
        x_minus = x.copy(); x_minus[i, j] -= eps
        loss_plus = np.sum(sigmoid.forward(x_plus) * grad_output)
        loss_minus = np.sum(sigmoid.forward(x_minus) * grad_output)
        numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

diff = np.abs(numerical_grad - grad_input).max()
print("Sigmoid gradient check diff:", diff)
assert diff < 1e-4, "Sigmoid backward SAI!"
print("✅ Sigmoid PASS")

# --- Test Softmax ---
softmax = Softmax()
x = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1001.0]])  # dòng 2 test ổn định số học
out = softmax.forward(x)
print("Softmax output:\n", out)
assert np.allclose(np.sum(out, axis=1), 1.0), "Tổng xác suất mỗi hàng phải = 1"
assert not np.any(np.isnan(out)), "Softmax bị NaN - lỗi ổn định số học!"
print("✅ Softmax PASS")

print("\n✅ Tất cả test activation PASS!")