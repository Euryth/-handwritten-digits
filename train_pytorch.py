import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from data_loader import load_dataset, train_val_test_split


class DigitNet(nn.Module):
    """Same architecture as the from-scratch model, for a fair comparison."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)  # raw logits, same convention as the from-scratch model


def compute_confusion_matrix(y_pred, y_true, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def plot_confusion_matrix(matrix, save_path="confusion_matrix_pytorch.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Oranges")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - PyTorch (Test Set)")
    for i in range(10):
        for j in range(10):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")
    plt.close(fig)


def plot_training_curves(losses, accs, save_path="training_curves_pytorch.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses, marker="o")
    axes[0].set_title("Training Loss per Epoch (PyTorch)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(accs, marker="o", color="green")
    axes[1].set_title("Validation Accuracy per Epoch (PyTorch)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training curves to {save_path}")
    plt.close(fig)


def main():
    print("Loading dataset...")
    X, y = load_dataset()  # y here is integer labels (0-9), not one-hot
    # train_val_test_split expects a 2D y (it was written for one-hot), so
    # we reuse it with y reshaped to a column vector, then squeeze back.
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y.reshape(-1, 1)
    )
    y_train = y_train.flatten()
    y_val = y_val.flatten()
    y_test = y_test.flatten()

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    device = torch.device("cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    model = DigitNet().to(device)
    criterion = nn.CrossEntropyLoss()  # combines softmax + cross-entropy, like our from-scratch version
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 10
    batch_size = 64
    n_train = X_train_t.shape[0]

    train_loss_history = []
    val_acc_history = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(n_train)
        epoch_losses = []

        for start in range(0, n_train, batch_size):
            idx = permutation[start:start + batch_size]
            X_batch = X_train_t[idx].to(device)
            y_batch = y_train_t[idx].to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        train_loss_history.append(np.mean(epoch_losses))
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train loss: {train_loss_history[-1]:.4f} | "
              f"Val accuracy: {val_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed:.1f} seconds")

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = (test_preds == y_test_t).float().mean().item()
    print(f"Final test accuracy: {test_acc:.4f}")

    torch.save(model.state_dict(), "model_weights_pytorch.pt")
    print("Saved model to model_weights_pytorch.pt")

    plot_training_curves(train_loss_history, val_acc_history)
    conf_matrix = compute_confusion_matrix(
        test_preds.cpu().numpy(), y_test_t.cpu().numpy()
    )
    plot_confusion_matrix(conf_matrix)


if __name__ == "__main__":
    main()