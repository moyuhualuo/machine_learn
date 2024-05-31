import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: 获取MNIST数据集并显示原始图像
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']
print(X.shape)
# 显示原始图像
def plot_digits(data, title):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks':[], 'yticks':[]})
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(28, 28), cmap='gray')
    plt.show()

plot_digits(X, "Original Images")

# Step 2: 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: 使用PCA计算最适降维参数k
pca = PCA()
pca.fit(X_scaled)

# 计算累计方差解释率
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
k = np.argmax(cumulative_variance >= 0.95) + 1
print(f"保留 95% 方差的 k 值: {k}")

# Step 4: 降维并重构图像
pca = PCA(n_components=k)
X_pca = pca.fit_transform(X_scaled)
X_reconstructed = pca.inverse_transform(X_pca)

# Step 5: 展示降维后重构的图像
plot_digits(X_reconstructed, "Reconstructed Images with k = " + str(k))

# 计算并显示原始和重构图像之间的均方误差
mse = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"均值方差MSE: {mse}")


# Step 6: 分析和展示不同k值下的主成分权重和
k_values = range(1, 400, 50)
explained_variances = []

for k in k_values:
    pca = PCA(n_components=k)
    pca.fit(X_scaled)
    explained_variances.append(np.sum(pca.explained_variance_ratio_))

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(k_values, explained_variances, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Components (k)')
plt.ylabel('Sum of Explained Variance Ratio')
plt.title('Sum of Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()
