import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from config import TERRAIN_NAMES

data = np.load("results/dataset.npz")
X_test, y_test = data["X_test"], data["y_test"]

with open("results/model.pkl", "rb") as f:
    model = pickle.load(f)

class_names = [TERRAIN_NAMES[i] for i in range(5)]
predictions = model.predict(X_test)

disp = ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, display_labels=class_names,
    cmap="Blues", xticks_rotation=30
)
disp.ax_.set_title("AI classifier: confusion matrix (unseen test maps)")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.show()
print("Saved to results/confusion_matrix.png")