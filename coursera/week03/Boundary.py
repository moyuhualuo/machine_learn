import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)

# 分割样本数据为两个类别
X_class0 = X[y[:, 0] == 0]
X_class1 = X[y[:, 0] == 1]
print(X_class1)

# 绘制样本点
fig, (ax, ax0) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax.scatter(X_class0[:, 0], X_class0[:, 1], c='b', label='Class 0')
ax.scatter(X_class1[:, 0], X_class1[:, 1], c='r', label='Class 1')
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_title('Scatter plot of sample data')
ax.legend()
ax.grid(True)


x0 = np.arange(0,6)
x1 = 3 - x0

# 𝑓(𝐱)=𝑔(−3+𝑥0+𝑥1)
def f(X):
    return sigmoid(-3 + X[:, 0] + X[:, 1])
res = f(X)
print(res)
ax0.axis([0, 4, 0, 3.5])
ax0.plot(x0, x1, c='b')
ax0.scatter(X_class0[:, 0], X_class0[:, 1], c='b', label='Class 0')
ax0.scatter(X_class1[:, 0], X_class1[:, 1], c='r', label='Class 1')
ax0.grid(True)
ax0.set_xlabel('$x_0$')
ax0.set_ylabel('$x_1$')
ax0.fill_between(x0, x1, alpha=0.2)
ax0.set_title('$f(x) = x_0 + x_1 - 3$')

plt.show()