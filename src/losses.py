import numpy as np


class SoftmaxCrossEntropyLoss:
    """Combines Softmax + Cross-Entropy for numerically stable gradients."""

    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
        # y_true: one-hot encoded, shape (batch, num_classes)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        self.probs = probs
        self.y_true = y_true

        batch_size = logits.shape[0]
        # add small epsilon to avoid log(0)
        eps = 1e-12
        loss = -np.sum(y_true * np.log(probs + eps)) / batch_size
        return loss

    def backward(self):
        batch_size = self.y_true.shape[0]
        grad_logits = (self.probs - self.y_true) / batch_size
        return grad_logits