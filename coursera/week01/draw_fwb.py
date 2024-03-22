import numpy as np
import Gradient_Descent as GD
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
start_time = time.time()

w_range = GD.my_instance.w
b_range = np.linspace(-100.0, 500.0, w_range.shape[0])
print(w_range.shape, b_range.shape)
J = np.zeros((w_range.shape[0], b_range.shape[0]))


for i, w in enumerate(w_range):
    for j, b in enumerate(b_range):
        GD.my_instance.set_b(b)
        # print(GD.my_instance.get_b())
        cost = GD.my_instance.compute_cost(w)
        J[i, j] = cost

fig = plt.figure()
ax1 = fig.add_subplot(131, projection='3d')

W, B = np.meshgrid(w_range, b_range)
'''
# 计算J的最小值和最大值
min_J = np.min(J)
max_J = np.max(J)
# 对J进行归一化处理
normalized_J = (J - min_J) / (max_J - min_J)
surf = ax1.plot_surface(W, B, normalized_J, cmap='viridis')
'''
surf = ax1.plot_surface(W, B, J, cmap='viridis')


ax1.set_xlabel('W')
ax1.set_ylabel('B')
ax1.set_zlabel('J(W, B)')
ax1.set_title('J(W,B)')

# 等高图
contour_levels = np.linspace(np.min(J), np.max(J), 100)
ax2 = fig.add_subplot(132)
ax2.contour(W, B, J, levels=contour_levels, cmap='terrain')



# 设置 w, b, a模拟过程
def fwb_y(w, x, y, b):
    return w * x + b - y

w, b = 150, 90
a = 0.5
m = GD.my_instance.x.shape[0]
x = GD.my_instance.x
y = GD.my_instance.y

# 收集散点数据的列表
scatter_data = []
contour_data = []
J_data = []
for i in range(100):
    temp_w = w - a * (1 / m) * np.dot((fwb_y(w, x, y, b)), x)
    temp_b = b - a * (1 / m) * np.sum(fwb_y(w, x, y, b))
    w, b = temp_w, temp_b
    GD.my_instance.set_b(b)
    J_wb = GD.my_instance.compute_cost(w)
    J_data.append((w, J_wb))
    scatter_data.append((w, b, J_wb))  # 将散点数据添加到列表中
    contour_data.append((w, b, J_wb))
    # print(w, b)

# 绘制所有散点
for data1, data2 in zip(scatter_data, contour_data):
    ax1.scatter(*data1, marker='x', c='r')
    ax2.scatter(*data2, marker='x', c='r')
ax3 = fig.add_subplot(133)
J_data_w = [item[0] for item in J_data]
J_data_J = [item[1] for item in J_data]
ax3.plot(J_data_w, J_data_J)
ax3.scatter(J_data_w, J_data_J, marker = 'x', c ='r')
ax3.set_xlabel('W')
ax3.set_ylabel('J')
ax1.view_init(elev=30, azim=45)
plt.show()
print(w, b)
print(time.time() - start_time)