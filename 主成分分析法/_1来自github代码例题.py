# # 假设数据如下
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
#
# # 构造示例数据
# data = pd.DataFrame({
#     'PM2.5': [35, 40, 45, 50, 55],
#     'PM10': [80, 85, 90, 95, 100],
#     'SO2': [10, 12, 14, 16, 18],
#     'NO2': [20, 22, 24, 26, 28],
#     'CO': [1.2, 1.3, 1.4, 1.5, 1.6],
#     'O3': [60, 65, 70, 75, 80]
# })
#
# # 1. 标准化
# scaler = StandardScaler()
# data_std = scaler.fit_transform(data)
#
# # 2. PCA降维
# pca = PCA(n_components=2)  # 保留2个主成分
# principal_components = pca.fit_transform(data_std)
#
# # 3. 查看主成分贡献率
# print("各主成分方差贡献率：", pca.explained_variance_ratio_)
# print("主成分矩阵：\n", principal_components)
#
# # 4. 主成分载荷矩阵
# print("主成分载荷矩阵：\n", pca.components_)

# 完整可运行的葡萄酒数据集PCA分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine  # 正确导入load_wine函数
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# 在所有绘图代码前添加以下配置
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置支持中文的字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 或者更精确的字体设置
font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')  # 黑体
# 1. 加载葡萄酒数据集
wine = load_wine()

# 2. 查看数据集基本信息
print("特征名称:", wine.feature_names)
print("目标类别:", wine.target_names)

# 3. 创建DataFrame以便查看数据
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target
wine_df['class'] = wine_df['target'].map({0: wine.target_names[0],
                                         1: wine.target_names[1],
                                         2: wine.target_names[2]})

# 4. 显示数据集的前5行
print("\n数据集前5行:")
print(wine_df.head())

# 5. 显示数据集统计信息
print("\n数据集统计信息:")
print(wine_df.describe().T)

# 6. 绘制特征相关性热力图
plt.figure(figsize=(12, 10))
corr_matrix = wine_df.iloc[:, :-2].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("葡萄酒特征相关性热力图")
plt.tight_layout()
plt.savefig('wine_correlation.png', dpi=300)
plt.show()

# 7. 数据标准化
X = wine.data
y = wine.target
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 8. PCA分析
pca = PCA(n_components=3)  # 保留3个主成分
X_pca = pca.fit_transform(X_std)

# 9. 创建PCA结果DataFrame
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
pca_df['class'] = wine.target_names[y]
pca_df['actual_class'] = y

# 10. 打印PCA结果
print("\n主成分方差贡献率:", pca.explained_variance_ratio_)
print("累计方差贡献率:", np.cumsum(pca.explained_variance_ratio_))
print("\n主成分系数:")
for i, component in enumerate(pca.components_):
    print(f"PC{i+1}: {component}")

# 11. 可视化结果
plt.figure(figsize=(18, 12))

# 11.1 二维散点图
plt.subplot(2, 2, 1)
sns.scatterplot(x='PC1', y='PC2', hue='class', data=pca_df,
                palette='viridis', s=100, alpha=0.8)
plt.title('PCA投影 - PC1 vs PC2')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')

# 11.2 三维散点图
ax = plt.subplot(2, 2, 2, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                    c=pca_df['actual_class'], cmap='viridis', s=50)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('三维PCA投影')

# 创建图例
legend_labels = [plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=plt.cm.viridis(i/2),
                            markersize=10, label=wine.target_names[i])
                for i in range(3)]
ax.legend(handles=legend_labels, title='葡萄酒类别')

# 11.3 主成分的载荷图
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)  # 计算载荷
plt.subplot(2, 2, 3)
for i, feature in enumerate(wine.feature_names):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.05, head_length=0.05, fc='k', ec='k')
    plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature, fontsize=9)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('主成分载荷图 (PC1 vs PC2)')
plt.axhline(y=0, color='gray', linestyle='--')
plt.axvline(x=0, color='gray', linestyle='--')
plt.grid()

# 11.4 方差解释率条形图
plt.subplot(2, 2, 4)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.8, align='center', label='各主成分方差')
plt.step(range(1, len(cumulative_variance)+1), cumulative_variance, where='mid', label='累计方差')
plt.xlabel('主成分')
plt.ylabel('解释方差比例')
plt.title('主成分方差解释率')
plt.legend(loc='best')
plt.ylim(0, 1.1)

plt.tight_layout()
plt.savefig('wine_pca_analysis.png', dpi=300)
plt.show()

# 12. 基于PCA的葡萄酒分类模型
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# 使用PCA降维
pca_for_model = PCA(n_components=3)
X_train_pca = pca_for_model.fit_transform(X_train)
X_test_pca = pca_for_model.transform(X_test)

# 训练随机森林分类器
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_pca, y_train)

# 评估模型
y_pred = model.predict(X_test_pca)
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=wine.target_names,
           yticklabels=wine.target_names)
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('混淆矩阵')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# 13. 保存完整数据集到CSV
wine_df.to_csv('wine_dataset.csv', index=False)
print("\n完整数据集已保存为 'wine_dataset.csv'")
