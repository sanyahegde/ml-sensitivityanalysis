# rf_analysis.py

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --------------------------------
# Load Digits Dataset
# --------------------------------
data = load_digits()
X, y = data.data, data.target

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# Train Base Random Forest Model
# --------------------------------
rf_base = RandomForestClassifier(random_state=42)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

print("Base Accuracy (default settings):", accuracy_score(y_test, y_pred_base))

# --------------------------------
# Hyperparameter: n_estimators
# --------------------------------
n_values = list(range(10, 1001, 100))
acc_n = []

for n in n_values:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc_n.append(accuracy_score(y_test, y_pred))

plt.figure()
plt.plot(n_values, acc_n, marker='o')
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.title("Effect of n_estimators on Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------
# Hyperparameter: max_depth
# --------------------------------
depth_values = list(range(5, 101, 10))
acc_depth = []

for d in depth_values:
    rf = RandomForestClassifier(max_depth=d, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc_depth.append(accuracy_score(y_test, y_pred))

plt.figure()
plt.plot(depth_values, acc_depth, marker='o')
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Effect of max_depth on Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------
# Hyperparameter: min_samples_leaf
# --------------------------------
leaf_values = list(range(1, 201, 20))
acc_leaf = []

for l in leaf_values:
    rf = RandomForestClassifier(min_samples_leaf=l, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc_leaf.append(accuracy_score(y_test, y_pred))

plt.figure()
plt.plot(leaf_values, acc_leaf, marker='o')
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.title("Effect of min_samples_leaf on Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()
