import numpy as np
import matplotlib.pyplot as plt
import time

from data_loader import load_dataset, one_hot_encode, train_val_test_split
from src.layers import Dense
from src.activations import ReLU
from src.dropout import Dropout
from src.losses import SoftmaxCrossEntropyLoss
from src.optimizers import SGD
from src.network import Sequential


def accuracy(logits, y_true_onehot):
    predicted_labels = np.argmax(logits, axis=1)
    true_labels = np.argmax(y_true_onehot, axis=1)
    return np.mean(predicted_labels == true_labels)


def iterate_minibatches(X, y, batch_size, shuffle=True):
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def compute_confusion_matrix(y_pred_labels, y_true_labels, num_classes=10):
    """Manual confusion matrix (no sklearn needed). Rows = true label, cols = predicted label."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true_labels, y_pred_labels):
        matrix[true_label, pred_label] += 1
    return matrix


def plot_training_curves(train_losses, val_accuracies, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, marker="o")
    axes[0].set_title("Training Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(val_accuracies, marker="o", color="green")
    axes[1].set_title("Validation Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training curves to {save_path}")
    plt.close(fig)


def plot_confusion_matrix(matrix, save_path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (Test Set)")

    # annotate each cell with its count
    for i in range(10):
        for j in range(10):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")
    plt.close(fig)


def main():
    print("Loading dataset...")
    X, y = load_dataset()
    y_onehot = one_hot_encode(y)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y_onehot)
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    net = Sequential([
        Dense(784, 128, seed=1),
        ReLU(),
        Dropout(drop_prob=0.3),
        Dense(128, 64, seed=2),
        ReLU(),
        Dense(64, 10, seed=3),
    ])

    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = SGD(layers=net.get_trainable_layers(), lr=0.01, momentum=0.9)

    num_epochs = 10
    batch_size = 64

    train_loss_history = []
    val_acc_history = []

    start_time = time.time()

    for epoch in range(num_epochs):
        net.train()
        epoch_losses = []

        for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size):
            logits = net.forward(X_batch)
            loss = loss_fn.forward(logits, y_batch)
            epoch_losses.append(loss)

            grad = loss_fn.backward()
            net.backward(grad)
            optimizer.step()

        net.eval()
        val_logits = net.forward(X_val)
        val_acc = accuracy(val_logits, y_val)

        train_loss_history.append(np.mean(epoch_losses))
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train loss: {train_loss_history[-1]:.4f} | "
              f"Val accuracy: {val_acc:.4f}")


    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed:.1f} seconds")

    net.eval()
    test_logits = net.forward(X_test)
    test_acc = accuracy(test_logits, y_test)
    print(f"\nFinal test accuracy: {test_acc:.4f}")

    net.save_weights("model_weights.npz")

    # --- plots ---
    plot_training_curves(train_loss_history, val_acc_history)

    y_pred_labels = np.argmax(test_logits, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    conf_matrix = compute_confusion_matrix(y_pred_labels, y_true_labels)
    plot_confusion_matrix(conf_matrix)


if __name__ == "__main__":
    main()