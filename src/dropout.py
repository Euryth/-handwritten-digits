import numpy as np


class Dropout:
    """Randomly zeroes out units during training to prevent overfitting."""

    def __init__(self, drop_prob=0.5):
        self.drop_prob = drop_prob
        self.mask = None
        self.training = True  # toggle with .train() / .eval()

    def forward(self, x):
        if not self.training:
            return x  # no dropout at inference time

        # inverted dropout: scale by 1/(1-p) during training
        # so no rescaling is needed at inference time
        keep_prob = 1.0 - self.drop_prob
        self.mask = (np.random.rand(*x.shape) < keep_prob) / keep_prob
        return x * self.mask

    def backward(self, grad_output):
        if not self.training:
            return grad_output
        return grad_output * self.mask

    def train(self):
        self.training = True

    def eval(self):
        self.training = False