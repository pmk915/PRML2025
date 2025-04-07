import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 生成数据的函数（原题提供）
def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y

# 生成训练和测试数据
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)  # 1000训练样本
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)    # 500测试样本

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义模型
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'AdaBoost + Decision Tree': AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=5), n_estimators=50, random_state=42
    ),
    'SVM Linear': SVC(kernel='linear', random_state=42),
    'SVM Poly': SVC(kernel='poly', degree=3, random_state=42),
    'SVM RBF': SVC(kernel='rbf', random_state=42)
}

# 新增的3D可视化函数
def plot_3d_classification(X, y_pred, model_name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制预测结果
    scatter = ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        c=y_pred, cmap='viridis', s=30, alpha=0.8
    )
    
    # 设置图形参数
    ax.set_title(f'Classification by {model_name}\n(Test Set Results)', fontsize=14)
    ax.set_xlabel('X (Standardized)', fontsize=12)
    ax.set_ylabel('Y (Standardized)', fontsize=12)
    ax.set_zlabel('Z (Standardized)', fontsize=12)
    
    # 添加图例
    legend = ax.legend(*scatter.legend_elements(),
                       title="Predicted Class",
                       loc="upper right",
                       bbox_to_anchor=(1.2, 0.8))
    ax.add_artist(legend)
    
    # 调整视角
    ax.view_init(elev=20, azim=-35)
    plt.tight_layout()
    plt.show()

# 训练与评估 + 可视化
for name, model in models.items():
    # 训练模型
    model.fit(X_train_scaled, y_train)
    
    # 预测结果
    y_pred = model.predict(X_test_scaled)
    
    # 输出性能指标
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # 可视化分类结果
    plot_3d_classification(X_test_scaled, y_pred, name)