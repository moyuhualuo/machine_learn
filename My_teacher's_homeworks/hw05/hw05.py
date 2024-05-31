import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 定义中心点
blob_centers = np.array(
    [[0.2, 2.3],
     [-1.5, 2.3],
     [-2.8, 1.8],
     [-2.8, 2.8],
     [-2.8, 1.3]]
)

blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
# 生成聚类样本
X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=42)

# 可视化聚类样本
plt.scatter(X[:, 0], X[:, 1], c='blue',s=10, alpha=0.5)
plt.title('Generated Blob Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#利用sk-learn聚类
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X)
print('labels_:', kmeans.labels_)
centroids = kmeans.cluster_centers_
print('中心点：', centroids)

# 绘制原始数据点和聚类中心
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=10, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')

# 绘制决策边界
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', linewidths=1, alpha=0.5)
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap='viridis', aspect='auto', origin='lower', alpha=0.3)

plt.title('KMeans Clustering Result with Decision Boundaries')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
