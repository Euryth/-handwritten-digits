import os

# Thư mục gốc chứa dataset (mỗi thư mục con 0-9 chứa ảnh PNG)
DATASET_DIR = r"D:\git\dataset"

CLASSES = [str(i) for i in range(10)]  # "0".."9"

IMAGE_SIZE = 28  # resize mỗi ảnh về kích thước vuông IMAGE_SIZE x IMAGE_SIZE

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42