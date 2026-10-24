# src/prepare.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.makedirs("data/raw", exist_ok=True)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
np.savez_compressed(
    "data/raw/fashion_mnist_raw.npz",
    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
)
print("Saved raw data:", x_train.shape, x_test.shape)