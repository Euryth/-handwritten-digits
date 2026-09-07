import numpy as np


class ReLU:
    def __init__(self):
        self.mask = None  # cache: vị trí nào bị "tắt" (x <= 0)

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1 - self.out)


class Softmax:
    def forward(self, x):
        # trừ max để ổn định số học (tránh tràn số khi exp)
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)