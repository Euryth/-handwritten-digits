import numpy as np
from src.layers import Dense

np.random.seed(0)

# Test 1: kiểm tra shape đúng
layer = Dense(in_features=4, out_features=3, seed=42)
x = np.random.randn(5, 4)  # batch=5, in_features=4
out = layer.forward(x)
print("Forward output shape:", out.shape)  # kỳ vọng (5, 3)
assert out.shape == (5, 3), "Sai shape forward!"

grad_output = np.random.randn(5, 3)  # giả lập gradient từ layer sau truyền về
grad_input = layer.backward(grad_output)
print("Backward grad_input shape:", grad_input.shape)  # kỳ vọng (5, 4)
assert grad_input.shape == (5, 4), "Sai shape backward!"
print("grad_W shape:", layer.grad_W.shape)  # kỳ vọng (4, 3)
print("grad_b shape:", layer.grad_b.shape)  # kỳ vọng (1, 3)

# Test 2: numerical gradient checking cho grad_W (kiểm tra backward tính đúng công thức)
def loss_fn(W):
    layer.W = W
    out = layer.forward(x)
    return np.sum(out ** 2)  # loss giả định đơn giản

eps = 1e-5
W_orig = layer.W.copy()
numerical_grad = np.zeros_like(W_orig)

for i in range(W_orig.shape[0]):
    for j in range(W_orig.shape[1]):
        W_plus = W_orig.copy()
        W_plus[i, j] += eps
        loss_plus = loss_fn(W_plus)

        W_minus = W_orig.copy()
        W_minus[i, j] -= eps
        loss_minus = loss_fn(W_minus)

        numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

layer.W = W_orig
out = layer.forward(x)
analytic_grad = (2 * out).T @ x  # d(sum(out^2))/dW theo công thức tay, transpose vì grad_W = x.T @ grad_output

analytic_grad = analytic_grad.T  # khớp shape (in_features, out_features)

diff = np.abs(numerical_grad - analytic_grad).max()
print("Max difference (numerical vs analytic):", diff)
assert diff < 1e-4, "Gradient checking FAILED - backward tính sai!"

print("✅ Tất cả test PASS!")