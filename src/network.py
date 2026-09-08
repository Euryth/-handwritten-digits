import numpy as np

class Sequential:
    """Chains a list of layers together (Dense, ReLU, etc.) in order."""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x  # these are raw logits (before softmax) if last layer is Dense

    def backward(self, grad_output):
        # backward goes in REVERSE order through the layers
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def get_trainable_layers(self):
        # only layers with weights (e.g. Dense) matter for the optimizer
        return [layer for layer in self.layers if hasattr(layer, "W")]

    def predict(self, x):
        # returns raw logits; caller applies argmax or softmax as needed
        return self.forward(x)

    def train(self):
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()

    def save_weights(self, path):
        weights = {}
        for i, layer in enumerate(self.get_trainable_layers()):
            weights[f"W{i}"] = layer.W
            weights[f"b{i}"] = layer.b
        np.savez(path, **weights)
        print(f"Saved model weights to {path}")

    def load_weights(self, path):
        data = np.load(path)
        for i, layer in enumerate(self.get_trainable_layers()):
            layer.W = data[f"W{i}"]
            layer.b = data[f"b{i}"]
        print(f"Loaded model weights from {path}")