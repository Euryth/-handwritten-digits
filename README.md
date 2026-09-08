# Handwritten Digits Recognition — Neural Network from Scratch

A neural network built entirely from scratch using only NumPy (no PyTorch/TensorFlow) to classify handwritten digits (0-9), created as a hands-on project to learn Git/GitHub workflows and understand the internals of neural networks.

## Dataset

[Handwritten Digits Dataset (not MNIST)](https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist) from Kaggle — 107,730 images, balanced across 10 digit classes (10,773 images per class), stored as 28x28 RGBA PNGs (digit stroke encoded in the alpha channel).

## Architecture

Input (784) → Dense(128) → ReLU → Dropout(0.3) → Dense(64) → ReLU → Dense(10) → Softmax + Cross-Entropy


All core components implemented from scratch:
- `src/layers.py` — Dense (fully connected) layer with forward/backward pass
- `src/activations.py` — ReLU, Sigmoid, Softmax
- `src/dropout.py` — Inverted dropout with train/eval modes
- `src/losses.py` — Combined Softmax + Cross-Entropy loss
- `src/optimizers.py` — SGD with optional momentum
- `src/network.py` — Sequential container chaining layers together

Every module was verified with **numerical gradient checking** before being merged, to confirm the backward pass math is correct.

## Results

| | From-scratch (NumPy) | PyTorch |
|---|---|---|
| Test accuracy | 100% | 100% |
| Training time (10 epochs) | 25.3s | 11.1s |

See `training_curves.png` and `confusion_matrix.png` for the from-scratch model's training curves and confusion matrix (PyTorch equivalents: `training_curves_pytorch.png`, `confusion_matrix_pytorch.png`).

## Key learning: the domain gap problem

Despite 100% test accuracy, the model performs noticeably worse on hand-drawn digits (via `predict_draw.py`). This is because the training dataset consists of uniformly rendered digits (consistent stroke width, centering), while hand-drawn input has very different stroke characteristics. This is a hands-on demonstration that high test accuracy does not guarantee real-world generalization when the input distribution shifts.

## Project structure

handwritten-digits/
├── src/
│ ├── layers.py
│ ├── activations.py
│ ├── dropout.py
│ ├── losses.py
│ ├── optimizers.py
│ └── network.py
├── config.py # paths and hyperparameters
├── data_loader.py # dataset loading, caching, preprocessing
├── train.py # from-scratch training script
├── train_pytorch.py # PyTorch comparison script
├── predict_draw.py # Tkinter GUI to draw a digit and predict
├── test_*.py # unit tests with gradient checking for each module
└── requirements.txt


## How to run

```bash
pip install -r requirements.txt
python train.py              # train the from-scratch model
python train_pytorch.py      # train the PyTorch comparison model
python predict_draw.py       # draw a digit and get a live prediction
```

## Notes

This project was also used to practice a full Git/GitHub workflow: branching, committing, resolving merge conflicts, and Pull Requests for every module added.