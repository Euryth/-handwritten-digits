import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

from src.layers import Dense
from src.activations import ReLU
from src.dropout import Dropout
from src.network import Sequential
from config import IMAGE_SIZE

CANVAS_SIZE = 280  # bigger canvas for easier drawing, downscaled to 28x28 for the model
BRUSH_RADIUS = 8


def build_network():
    net = Sequential([
        Dense(784, 128, seed=1),
        ReLU(),
        Dropout(drop_prob=0.3),
        Dense(128, 64, seed=2),
        ReLU(),
        Dense(64, 10, seed=3),
    ])
    net.load_weights("model_weights.npz")
    net.eval()  # disable dropout at inference time
    return net


class DrawApp:
    def __init__(self, root, net):
        self.net = net
        self.root = root
        self.root.title("Draw a digit (0-9)")

        # black background canvas, matching the training data (bright stroke on dark background)
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        # PIL image kept in sync with the canvas, used for the actual prediction
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)
        tk.Button(button_frame, text="Predict", command=self.predict).pack(side="left", padx=5)
        tk.Button(button_frame, text="Clear", command=self.clear).pack(side="left", padx=5)

        self.result_label = tk.Label(root, text="Draw a digit and click Predict", font=("Arial", 14))
        self.result_label.pack(pady=5)

    def paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit and click Predict")

    def predict(self):
        img_small = self.image.resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.array(img_small, dtype=np.float32) / 255.0
        x = arr.flatten().reshape(1, -1)

        logits = self.net.forward(x)
        shifted = logits - logits.max()
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        predicted = int(np.argmax(probs))
        confidence = float(probs[0, predicted])

        self.result_label.config(text=f"Predicted: {predicted}  (confidence: {confidence:.2%})")


if __name__ == "__main__":
    net = build_network()
    root = tk.Tk()
    app = DrawApp(root, net)
    root.mainloop()