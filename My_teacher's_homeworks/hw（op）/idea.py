import numpy as np
from sklearn.model_selection import train_test_split

# 生成随机数据
np.random.seed(43)
num_samples = 50
X1 = np.random.randn(num_samples) * 2 + 5  # 特征X1，均值为5，标准差为2
X2 = np.random.randn(num_samples) * 3 + 10  # 特征X2，均值为10，标准差为3

y = np.zeros(num_samples)
y[X1 + X2 > 15] = 1

X = np.column_stack((X1, X2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y1_samples = np.sum(y)
y0_samples = num_samples - y1_samples

p_y0 = y0_samples / num_samples
p_y1 = y1_samples / num_samples
