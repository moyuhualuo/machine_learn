import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成一些示例数据
x = np.random.rand(10)
y = np.random.rand(10)
z = np.random.rand(10)

# 创建一个新的3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制3D散点图
ax.scatter(x, y, z, c='r', marker='o', label='Points')

# 添加标签和标题
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')

# 显示图形
plt.show()
