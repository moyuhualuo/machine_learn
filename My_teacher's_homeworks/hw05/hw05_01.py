import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
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
plt.scatter(X[:, 0], X[:, 1], c='blue', s=10, alpha=0.5)
plt.title('Generated Blob Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 使用 DBSCAN 聚类
dbscan = DBSCAN(eps=0.3, min_samples=10)
dbscan.fit(X)
print('labels_:', dbscan.labels_)

# 绘制聚类结果
plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_, s=10, cmap='viridis', alpha=0.5)
plt.title('DBSCAN Clustering Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
