class SGD:
    """Stochastic Gradient Descent optimizer with optional momentum."""

    def __init__(self, layers, lr=0.01, momentum=0.0):
        self.layers = layers  # list of layer objects that have .W, .b, .grad_W, .grad_b
        self.lr = lr
        self.momentum = momentum

        # velocity buffers for momentum (one per layer)
        self.velocity_W = [None] * len(layers)
        self.velocity_b = [None] * len(layers)

    def step(self):
        for i, layer in enumerate(self.layers):
            if not hasattr(layer, "W"):
                continue  # skip layers with no trainable params (e.g. activations)

            if self.momentum == 0.0:
                layer.W -= self.lr * layer.grad_W
                layer.b -= self.lr * layer.grad_b
            else:
                if self.velocity_W[i] is None:
                    self.velocity_W[i] = 0.0
                    self.velocity_b[i] = 0.0
                self.velocity_W[i] = self.momentum * self.velocity_W[i] - self.lr * layer.grad_W
                self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.lr * layer.grad_b
                layer.W += self.velocity_W[i]
                layer.b += self.velocity_b[i]