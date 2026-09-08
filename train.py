import numpy as np

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

    for epoch in range(num_epochs):
        net.train()  # enable dropout
        epoch_losses = []

        for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size):
            logits = net.forward(X_batch)
            loss = loss_fn.forward(logits, y_batch)
            epoch_losses.append(loss)

            grad = loss_fn.backward()
            net.backward(grad)
            optimizer.step()

        net.eval()  # disable dropout for validation
        val_logits = net.forward(X_val)
        val_acc = accuracy(val_logits, y_val)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train loss: {np.mean(epoch_losses):.4f} | "
              f"Val accuracy: {val_acc:.4f}")

    net.eval()
    test_logits = net.forward(X_test)
    test_acc = accuracy(test_logits, y_test)
    print(f"\nFinal test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()