import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
X1=np.array([[5,1.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# 引用
lr_model = LogisticRegression()
# 拟合
lr_model.fit(X, y)

# 预测 y
y_pred = lr_model.predict(X1)
print("Prediction on training set:", y_pred)
# 拟合率
print("Accuracy on training set:", lr_model.score(X, y))