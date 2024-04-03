import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 生成随机数据
np.random.seed(0)
num_samples = 100
X1 = np.random.randn(num_samples) * 2 + 5  # 特征X1，均值为5，标准差为2
X2 = np.random.randn(num_samples) * 3 + 10  # 特征X2，均值为10，标准差为3

# 生成对应的类别标签
y = np.zeros(num_samples)
y[X1 + X2 > 15] = 1  # 简单的分类规则：如果X1 + X2 > 15，则类别标签为1，否则为0

# 将数据集划分为训练集和测试集
X = np.column_stack((X1, X2))  # 将特征合并为一个矩阵
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 可视化数据集
plt.figure(figsize=(8, 6))
plt.scatter(X1, X2, c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Simple Classification Dataset')
plt.colorbar(label='Class')
plt.show()

classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


print('y_test:', y_test)
print('y_pred:', y_pred)
print("准确率:", accuracy)
