import numpy as np
import matplotlib.pyplot as plt



# 生成一些 z 值
z = np.linspace(-10, 10, 100)

# 创建图表和坐标轴对象
fig, ax = plt.subplots()
ax.axis([-10, 10, -0.1, 1.1])
# 绘制 sigmoid 函数的图像
ax.plot(z, sigmoid(z), c="b")
ax.axvline(x=0, color='g', linestyle='--', label='x=0, left=red')
ax.fill_between(z, 1, where=(z <= 0), color='r', alpha=0.5)
ax.fill_between(z, 1, where=(z > 0), color='g', alpha=0.5)
# ax.fill_between(where=(z < 0), color='r', alpha=0.5)

# 设置坐标轴标签
ax.set_xlabel('z')
ax.set_ylabel('sigmoid(z)')
ax.set_title('Sigmoid Function')

# 显示图表
plt.show()
