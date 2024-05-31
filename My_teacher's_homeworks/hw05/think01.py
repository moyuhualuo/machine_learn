import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 生成数据
X, y = make_blobs(n_samples=10000, centers=[[0.5, 1.5], [-1.5, 2.5]], cluster_std=0.5, random_state=42)
X_rounded = np.around(X, decimals=2)
# 控制数据点坐标小数位数
x_min, x_max = X_rounded[:, 0].min(), X_rounded[:, 0].max()
y_min, y_max = X_rounded[:, 1].min(), X_rounded[:, 1].max()

def make_random_point():
    random_point = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])
    random_point = np.around(random_point, decimals=2)
    return random_point

def make_draw(x1, x2, my, b):
    # 绘制散点图
    plt.scatter(x1[0], x1[1], c='red', s=10)
    plt.scatter(x2[0], x2[1], c='red', s=10)
    plt.scatter(X_rounded[:, 0], X_rounded[:, 1], c='blue', s=1)
    plt.title('Generated Blob Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # 生成用于绘制中垂线的x值范围
    x_values = np.linspace(x_min, x_max, 400)
    y_values = my * x_values + b
    plt.plot(x_values, y_values, label=f'y = {my}x + {b:.2f}', c='pink')
    plt.legend()
    plt.show()

# 生成两个随机点
x1 = make_random_point()
x2 = make_random_point()
print(x1, x2)

def make_point(x1, x2):
    mid = ((x1[0] + x2[0]) / 2, (x1[1] + x2[1]) / 2)
    print(mid)
    k = (x2[1] - x1[1]) / (x2[0] - x1[0])
    mk = -1 / k
    mb = mid[1] - mk * mid[0]
    return mk, mb

mk, mb = make_point(x1, x2)
print("Slope of perpendicular bisector:", mk)
print("Intercept of perpendicular bisector:", mb)

#make_draw(x1, x2, mk, mb)

# 分类显示
labels = None
def classify_points(mk, mb):
    global labels
    mY = mk * X[:, 0] + mb
    labels = np.where(X[:, 1] > mY, 1, 0)
    plt.scatter(X_rounded[labels == 0, 0], X_rounded[labels == 0, 1], c='yellow', label='Label 0', s=1)
    plt.scatter(X_rounded[labels == 1, 0], X_rounded[labels == 1, 1], c='green', label='Label 1', s=1)
    plt.plot(X_rounded[:, 0], mY, c='black', label='Perpendicular Bisector')
    plt.scatter(x1[0], x1[1], c='red', s=10)
    plt.scatter(x2[0], x2[1], c='red', s=10)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Blob Data with Perpendicular Bisector')
    plt.legend()
    mean_x = np.mean(X[labels == 1, 0]), np.mean(X[labels == 0, 0])
    mean_y = np.mean(X[labels == 1, 1]), np.mean(X[labels == 0, 1])
    plt.scatter(mean_x[0], mean_y[0], c='b', s=25, marker='x')
    plt.scatter(mean_x[1], mean_y[1], c='b', s=25, marker='x')
    plt.show()
    return mean_x, mean_y

ls = []
for i in range(10):
   mean_x, mean_y = classify_points(mk, mb)
   x1 = (mean_x[0], mean_y[0])
   x2 = (mean_x[1], mean_y[1])
   mk, mb = make_point(x1, x2)
   accuracy = np.mean(y == labels)
   ls.append(accuracy)


indices = range(len(ls))
values = ls

# 绘制折线图
plt.plot(indices, values, marker='o', linestyle='-')
plt.title('Line Plot of List Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.xticks(indices)  # 设置 x 轴刻度为索引值
plt.grid(True)  # 显示网格线
plt.show()
# 计算准确率



print(ls)