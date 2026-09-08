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