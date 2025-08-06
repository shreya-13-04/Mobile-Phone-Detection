import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Class labels (0 = Phone, 1 = No Phone)
class_names = ["Phone", "No Phone"]

# Simulated true and predicted labels (example scenario)
np.random.seed(42)
y_true = np.random.choice([0, 1], size=100, p=[0.3, 0.7])   # 30% phones, 70% no phones
y_pred = y_true.copy()

# Introduce classification errors (simulate imperfect detection)
for i in range(15):
    idx = np.random.randint(0, 100)
    y_pred[idx] = 1 - y_pred[idx]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap (Phone Detection)")
plt.tight_layout()
plt.show()
