import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: 获取MNIST数据集并显示原始图像
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']

# Step 2: 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: 使用PCA计算累计方差解释率和最佳降维参数k
pca = PCA()
pca.fit(X_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
k_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"保留 95% 方差的 k 值: {k_95}")

# Step 4: 计算不同k值下的重构误差
k_values = range(320, 340)
mse_values = []

for k in k_values:
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)
    mse = np.mean((X_scaled - X_reconstructed) ** 2)
    mse_values.append(mse)
    print(f"k = {k}, Mean Squared Error = {mse}")

# Step 5: 绘制k值与MSE之间的关系图
plt.figure(figsize=(10, 6))
plt.plot(k_values, mse_values, marker='o', linestyle='-', color='b')
plt.axvline(x=k_95, color='r', linestyle='--', label=f'k = {k_95} (95% variance)')
plt.xlabel('Number of Components (k)')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs. Number of Components')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: 找到MSE最低的k值
best_k = k_values[np.argmin(mse_values)]
print(f"最佳 k 值: {best_k}, 最小均方误差: {min(mse_values)}")

# Step 7: 使用最佳k值进行PCA并重构图像
pca_best = PCA(n_components=best_k)
X_pca_best = pca_best.fit_transform(X_scaled)
X_reconstructed_best = pca_best.inverse_transform(X_pca_best)

# 显示降维后重构的图像
def plot_digits(data, title):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks':[], 'yticks':[]})
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(28, 28), cmap='gray')
    plt.show()

plot_digits(X_reconstructed_best, "Reconstructed Images with k = " + str(best_k))
