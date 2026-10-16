# src/preprocess.py
import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["preprocess"]

raw = np.load("data/raw/fashion_mnist_raw.npz")
x_train = raw["x_train"].astype("float32") / 255.0   # normalize to [0, 1]
x_test = raw["x_test"].astype("float32") / 255.0

x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, raw["y_train"],
    test_size=params["test_size"],
    random_state=params["seed"],
    stratify=raw["y_train"],
)

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed(
    "data/processed/data.npz",
    x_train=x_tr, y_train=y_tr,
    x_val=x_val, y_val=y_val,
    x_test=x_test, y_test=raw["y_test"],
)
print("Train/val/test:", x_tr.shape, x_val.shape, x_test.shape)