import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

#
# class ManualPCA:
#     def __init__(self, n_components=None):
#         self.n_components = n_components
#         self.components = None
#         self.explained_variance_ratio = None
#
#     def fit(self, X):
#         # 1. 数据标准化
#         self.mean = np.mean(X, axis=0)
#         self.std = np.std(X, axis=0)
#         X_std = (X - self.mean) / self.std
#
#         # 2. 计算协方差矩阵
#         cov_mat = np.cov(X_std.T)
#
#         # 3. 特征值分解
#         eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
#
#         # 4. 排序特征值(降序)
#         idx = np.argsort(eigenvalues)[::-1]
#         eigenvalues = eigenvalues[idx]
#         eigenvectors = eigenvectors[:, idx]
#
#         # 5. 计算方差贡献率
#         total_var = np.sum(eigenvalues)
#         self.explained_variance_ratio = eigenvalues / total_var
#
#         # 6. 选择主成分数
#         if self.n_components is not None:
#             self.components = eigenvectors[:, :self.n_components]
#         else:
#             # 自动选择累计贡献率>85%的主成分
#             cum_var = np.cumsum(self.explained_variance_ratio)
#             self.n_components = np.argmax(cum_var >= 0.85) + 1
#             self.components = eigenvectors[:, :self.n_components]
#
#         return self
#
#     def transform(self, X):
#         X_std = (X - self.mean) / self.std
#         return np.dot(X_std, self.components)
#
#     def fit_transform(self, X):
#         self.fit(X)
#         return self.transform(X)
#

# 1. 加载数据
# 在所有绘图代码前添加以下配置
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置支持中文的字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 或者更精确的字体设置
font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')  # 黑体
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

# 2. 数据预处理
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 3. PCA降维
pca = PCA(n_components=0.85)  # 保留85%方差
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print("主成分数:", pca.n_components_)
print("累计方差贡献率:", np.cumsum(pca.explained_variance_ratio_)[-1])

# 4. 建立分类模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_pca, y_train)

# 5. 模型评估
y_pred = model.predict(X_test_pca)

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 混淆矩阵可视化
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('混淆矩阵')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()


# 6. 主成分空间决策边界可视化
def plot_decision_boundary(X, y, model, title):
    # 网格点生成
    h = 0.02  # 步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测每个点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(),
    np.zeros((xx.ravel().shape[0], pca.n_components_ - 2))])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Set3)

    # 绘制数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y,
                          cmap=plt.cm.Set1, edgecolor='k', s=70)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.legend(handles=scatter.legend_elements()[0],
               labels=target_names,
               loc='upper right')

    plt.savefig('decision_boundary.png', dpi=300)
    plt.show()


# 绘制训练集的决策边界
plot_decision_boundary(X_train_pca, y_train, model,
                       "主成分空间决策边界 (训练集)")

# 绘制测试集的决策边界
plot_decision_boundary(X_test_pca, y_test, model,
                       "主成分空间决策边界 (测试集)")
