import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# 读取数据
train_df = pd.read_excel('Data4Regression.xlsx', sheet_name='Training Data')
test_df = pd.read_excel('Data4Regression.xlsx', sheet_name='Test Data')

# 训练数据
X_train = train_df['x'].values.reshape(-1, 1)
y_train = train_df['y_complex'].values

# 测试数据
X_test = test_df['x_new'].values.reshape(-1, 1)
y_test = test_df['y_new_complex'].values

# 构造带截距项的X矩阵
X_train_ls = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_ls = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# 1. 最小二乘法
w_ls = np.linalg.inv(X_train_ls.T @ X_train_ls) @ X_train_ls.T @ y_train
y_pred_train_ls = X_train_ls @ w_ls
y_pred_test_ls = X_test_ls @ w_ls
mse_train_ls = mean_squared_error(y_train, y_pred_train_ls)
mse_test_ls = mean_squared_error(y_test, y_pred_test_ls)

# 2. 梯度下降法
w_gd = np.zeros(2)
learning_rate = 0.01
iterations = 10000
m = X_train_ls.shape[0]

for _ in range(iterations):
    gradient = (X_train_ls.T @ (X_train_ls @ w_gd - y_train)) / m
    w_gd -= learning_rate * gradient

y_pred_train_gd = X_train_ls @ w_gd
y_pred_test_gd = X_test_ls @ w_gd
mse_train_gd = mean_squared_error(y_train, y_pred_train_gd)
mse_test_gd = mean_squared_error(y_test, y_pred_test_gd)

# 3. 牛顿法（结果与最小二乘法一致）
w_newton = w_ls.copy()
mse_train_newton = mse_train_ls
mse_test_newton = mse_test_ls

import matplotlib.pyplot as plt

# ----------------------------- 绘图功能 -----------------------------
# 创建包含三个子图的大画布
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1行3列，总宽度18英寸
plt.subplots_adjust(wspace=0.3)  # 调整子图间距

# 生成用于绘图的X范围
X_plot = np.linspace(min(X_train.min(), X_test.min()), 
                     max(X_train.max(), X_test.max()), 100).reshape(-1, 1)
X_plot_ls = np.hstack([np.ones((X_plot.shape[0], 1)), X_plot])

# 计算预测值
y_plot_ls = X_plot_ls @ w_ls
y_plot_gd = X_plot_ls @ w_gd
y_plot_newton = X_plot_ls @ w_newton

# ----------- 最小二乘法子图 -----------
axes[0].scatter(X_train, y_train, s=20, color='blue', alpha=0.6, label='Training Data')
axes[0].scatter(X_test, y_test, s=20, color='orange', alpha=0.6, label='Test Data')
axes[0].plot(X_plot, y_plot_ls, color='red', linewidth=2, label='Least Squares Fit')
axes[0].set_title(f"Least Squares\nTrain MSE: {mse_train_ls:.4f}, Test MSE: {mse_test_ls:.4f}")
axes[0].set_xlabel('x', fontsize=10)
axes[0].set_ylabel('y', fontsize=10)
axes[0].legend()
axes[0].grid(alpha=0.3)

# ----------- 梯度下降法子图 -----------
axes[1].scatter(X_train, y_train, s=20, color='blue', alpha=0.6, label='Training Data')
axes[1].scatter(X_test, y_test, s=20, color='orange', alpha=0.6, label='Test Data')
axes[1].plot(X_plot, y_plot_gd, '--', color='green', linewidth=2, label='Gradient Descent Fit')
axes[1].set_title(f"Gradient Descent\nTrain MSE: {mse_train_gd:.4f}, Test MSE: {mse_test_gd:.4f}")
axes[1].set_xlabel('x', fontsize=10)
axes[1].set_ylabel('y', fontsize=10)
axes[1].legend()
axes[1].grid(alpha=0.3)

# ----------- 牛顿法子图 -----------
axes[2].scatter(X_train, y_train, s=20, color='blue', alpha=0.6, label='Training Data')
axes[2].scatter(X_test, y_test, s=20, color='orange', alpha=0.6, label='Test Data')
axes[2].plot(X_plot, y_plot_newton, ':', color='purple', linewidth=2, label='Newton Method Fit')
axes[2].set_title(f"Newton Method\nTrain MSE: {mse_train_newton:.4f}, Test MSE: {mse_test_newton:.4f}")
axes[2].set_xlabel('x', fontsize=10)
axes[2].set_ylabel('y', fontsize=10)
axes[2].legend()
axes[2].grid(alpha=0.3)

# 自动调整布局并显示
plt.tight_layout()
plt.show()

# 输出结果（与之前相同）
print("最小二乘法：")
print(f"训练误差(MSE): {mse_train_ls:.6f}, 测试误差(MSE): {mse_test_ls:.6f}")
print("\n梯度下降法：")
print(f"训练误差(MSE): {mse_train_gd:.6f}, 测试误差(MSE): {mse_test_gd:.6f}")
print("\n牛顿法：")
print(f"训练误差(MSE): {mse_train_newton:.6f}, 测试误差(MSE): {mse_test_newton:.6f}")
