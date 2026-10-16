# evaluate.py  (will be moved to src/ in A8)
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

d = np.load("data/processed/data.npz")
model = keras.models.load_model("models/model.h5")

loss, acc = model.evaluate(d["x_test"], d["y_test"], verbose=0)
preds = np.argmax(model.predict(d["x_test"], verbose=0), axis=1)

os.makedirs("reports", exist_ok=True)
cm = confusion_matrix(d["y_test"], preds)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.savefig("reports/confusion_matrix.png", dpi=120, bbox_inches="tight")

with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)
print(f"test_loss={loss:.4f} test_accuracy={acc:.4f}")