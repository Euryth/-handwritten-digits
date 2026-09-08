import os
import numpy as np
from PIL import Image

from config import DATASET_DIR, CLASSES, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED

CACHE_PATH = "dataset_cache.npz"


def load_image(path):
    """Read one image. This dataset stores the digit stroke in the
    alpha channel (RGB channels are constant), so extract alpha directly
    instead of using standard grayscale conversion."""
    img = Image.open(path)
    if img.mode == "RGBA":
        arr = np.array(img)[:, :, 3]  # alpha channel holds the actual digit
    else:
        arr = np.array(img.convert("L"))

    img_resized = Image.fromarray(arr).resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return arr.flatten()


def load_dataset(dataset_dir=DATASET_DIR, cache_path=CACHE_PATH, use_cache=True):
    """
    Loads all digit images. If a cache file already exists, loads from it
    directly (much faster). Otherwise reads all PNGs from disk and creates
    the cache file for next time.
    """
    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        data = np.load(cache_path)
        return data["X"], data["y"]

    X = []
    y = []
    for label in CLASSES:
        digit_folder = os.path.join(dataset_dir, label, label)
        if not os.path.isdir(digit_folder):
            print(f"[WARNING] Folder not found: {digit_folder}")
            continue
        files = [f for f in os.listdir(digit_folder) if f.lower().endswith(".png")]
        print(f"Reading {len(files)} images for label '{label}'...")
        for fname in files:
            path = os.path.join(digit_folder, fname)
            X.append(load_image(path))
            y.append(int(label))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    if use_cache:
        print(f"Saving cache to {cache_path} for next time...")
        np.savez_compressed(cache_path, X=X, y=y)

    return X, y


def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot


def train_val_test_split(X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = rng.permutation(n)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


if __name__ == "__main__":
    X, y = load_dataset()
    print(f"Tổng số ảnh: {X.shape[0]}, kích thước mỗi ảnh (flatten): {X.shape[1]}")

    y_onehot = one_hot_encode(y)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y_onehot)
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")