from PIL import Image
import numpy as np
import os

path = os.path.join(r"D:\git\dataset", "0", "0", "0.png")
img = Image.open(path)

print("Mode:", img.mode)
print("Size:", img.size)

arr = np.array(img)
print("Raw array shape:", arr.shape)
print("Raw array dtype:", arr.dtype)
print("Min:", arr.min(), "Max:", arr.max(), "Mean:", arr.mean())

img_gray = img.convert("L")
arr_gray = np.array(img_gray)
print("\nAfter convert('L'):")
print("Min:", arr_gray.min(), "Max:", arr_gray.max(), "Mean:", arr_gray.mean())

img_gray.save("debug_check.png")
print("\nSaved debug_check.png - open this file with Photos/Paint to view it")