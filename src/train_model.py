import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from config import TERRAIN_NAMES
from features import FEATURE_NAMES

data = np.load("results/dataset.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

print("Training Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=150,      # number of decision trees voting together
    max_depth=10,          # limits how complicated each tree can get (reduces overfitting)
    random_state=42,       # makes the training reproducible
    class_weight="balanced"  # compensates for rock/steep being rarer classes
)
model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training accuracy: {train_accuracy:.3f}")
print(f"Test accuracy (unseen maps): {test_accuracy:.3f}")

print("\nFeature importance (which numbers the model relies on most):")
for name, importance in sorted(zip(FEATURE_NAMES, model.feature_importances_),
                                 key=lambda pair: -pair[1]):
    print(f"  {name:18s} {importance:.3f}")

print("\nPer-class report on the test set:")
class_names = [TERRAIN_NAMES[i] for i in range(5)]
print(classification_report(y_test, model.predict(X_test), target_names=class_names, zero_division=0))

cm = confusion_matrix(y_test, model.predict(X_test))
print("Confusion matrix (rows = true class, columns = predicted class):")
print(class_names)
print(cm)

with open("results/model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved to results/model.pkl")