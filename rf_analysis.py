from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


data = load_digits()
X, y = data.data, data.target

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_base = RandomForestClassifier(random_state=42)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, y_pred_base))


n_vals = np.linspace(10, 1000, 10, dtype=int)
acc_n = []

for n in n_vals:
    m = RandomForestClassifier(n_estimators=n, random_state=0)
    m.fit(X_train, y_train)
    acc_n.append(accuracy_score(y_test, m.predict(X_test)))

plt.plot(n_vals, acc_n)
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.title("Vary n_estimators")
plt.grid(True)
plt.tight_layout()
plt.show()

d_vals = np.linspace(5, 100, 10, dtype=int)
acc_d = []

for d in d_vals:
    m = RandomForestClassifier(max_depth=d, random_state=0)
    m.fit(X_train, y_train)
    acc_d.append(accuracy_score(y_test, m.predict(X_test)))

plt.plot(d_vals, acc_d)
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Vary max_depth")
plt.grid(True)
plt.tight_layout()
plt.show()


l_vals = np.linspace(1, 200, 10, dtype=int)
acc_l = []

for l in l_vals:
    m = RandomForestClassifier(min_samples_leaf=l, random_state=0)
    m.fit(X_train, y_train)
    acc_l.append(accuracy_score(y_test, m.predict(X_test)))

plt.plot(l_vals, acc_l)
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.title("Vary min_samples_leaf")
plt.grid(True)
plt.tight_layout()
plt.show()
