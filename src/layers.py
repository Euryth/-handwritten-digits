import numpy as np


class Dense:
    """Fully connected layer: output = input @ W + b"""

    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        # He initialization - tốt cho các layer theo sau là ReLU
        self.W = rng.standard_normal((in_features, out_features)) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))

        # cache cho backward
        self.x = None

        # gradient (tính ở backward, dùng cho optimizer update)
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x  # lưu lại để dùng trong backward
        return x @ self.W + self.b

    def backward(self, grad_output):
        self.grad_W = self.x.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.W.T
        return grad_input

    def update(self, lr):
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b