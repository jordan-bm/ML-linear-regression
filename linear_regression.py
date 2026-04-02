# Jordan Burmylo-Magrann
# Linear Regression from Scratch
# Predicts salary from years of experience using gradient descent
# No sklearn - only NumPy, Pandas, and Matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv("Salary_Data.csv")
X = df["YearsExperience"].values
y = df["Salary"].values
n = len(X)

# 2. Normalize Features 
# Keeps gradient descent stable - without this, large salary values
# cause the loss surface to be steep and learning rate sensitive
X_mean, X_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()

X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std

# 3. Gradient Descent
# Model: y_hat = w * x + b
# Loss:  MSE = (1/n) * sum((y_hat - y)^2)
# Gradients:
#   dL/dw = (2/n) * sum((y_hat - y) * x)
#   dL/db = (2/n) * sum((y_hat - y))

learning_rate = 0.01
epochs = 1000

w = 0.0
b = 0.0
loss_history = []

for epoch in range(epochs):
    y_hat = w * X_norm + b
    error = y_hat - y_norm

    dw = (2 / n) * np.dot(error, X_norm)
    db = (2 / n) * np.sum(error)

    w -= learning_rate * dw
    b -= learning_rate * db

    mse = np.mean(error ** 2)
    loss_history.append(mse)

print(f"Final MSE (normalized): {loss_history[-1]:.6f}")
print(f"Learned weight: {w:.4f}, bias: {b:.4f}")

# 4. Predictions in Original Scale 
X_line = np.linspace(X.min(), X.max(), 200)
X_line_norm = (X_line - X_mean) / X_std
y_line_norm = w * X_line_norm + b
y_line = y_line_norm * y_std + y_mean  # denormalize

# R² score
y_pred_norm = w * X_norm + b
ss_res = np.sum((y_norm - y_pred_norm) ** 2)
ss_tot = np.sum((y_norm - y_norm.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"R² Score: {r2:.4f}")

# 5. Plot 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Linear Regression from Scratch — Salary vs. Experience", fontsize=13, fontweight="bold")

# Left: regression line
axes[0].scatter(X, y, color="#2563eb", alpha=0.75, edgecolors="white", linewidth=0.5, s=70, label="Data points")
axes[0].plot(X_line, y_line, color="#dc2626", linewidth=2, label=f"Regression line (R²={r2:.3f})")
axes[0].set_xlabel("Years of Experience")
axes[0].set_ylabel("Salary ($)")
axes[0].set_title("Regression Fit")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: loss curve
axes[1].plot(loss_history, color="#16a34a", linewidth=1.5)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE Loss (normalized)")
axes[1].set_title("Training Loss Curve")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("linear_regression_plot.png", dpi=150, bbox_inches="tight")
print("Plot saved.")
