import numpy as np
import Gradient_Descent as GD
import matplotlib.pyplot as plt

"""fixed_b = 100
>>> draw J(w, b) of W
>>> 0.5m * sum (w * xi - yi + b) ** 2  // sum 代表求和符号
"""
# 画出图像
cost_arr = []
for w in GD.my_instance.w:
    cost = GD.my_instance.compute_cost(w)
    cost_arr.append(cost)
print(len(cost_arr))
plt.plot(GD.my_instance.w, cost_arr)
plt.xlabel('w')
plt.ylabel('Cost')

# 梯度下降

# 计算fwb - y 的值
def fwb_y(w, x, y, b = GD.my_instance.b):
    '''
    arr = []
    for i in range(m):
        cost = w * x[i] + b - y[i]
        arr.append(cost)
    return arr
    '''
    return w * x + b - y
x = GD.my_instance.x
y = GD.my_instance.y
w = 100
b = GD.my_instance.b
m = x.shape[0]
# a 的值设置，表示下降梯度值
a = 0.1
# 迭代十次，演示梯度下降，值的注意的是b 是固定的
for i in range(10):
    temp_w = w - a * (1 / m * fwb_y(w, x, y, b) @ x)
    # temp_b = b - a * (1 / m * sum(fwb_y(w, x, y, b)))
    w = temp_w
    # b = temp_b
    print(w, b)
    GD.my_instance.set_b(b)
    J_wb = GD.my_instance.compute_cost(w)
    plt.scatter(w, J_wb, marker='x', c='r')

#重新设置w值，观察图形
w = 300
for i in range(10):
    temp_w = w - a * (1 / m * fwb_y(w, x, y, b) @ x)
    # temp_b = b - a * (1 / m * sum(fwb_y(w, x, y, b)))
    w = temp_w
    # b = temp_b
    print(w, b)
    GD.my_instance.set_b(b)
    J_wb = GD.my_instance.compute_cost(w)
    plt.scatter(w, J_wb, marker='x', c='y')
plt.grid(True)
plt.legend(['J(w, b) of W', 'new_val'])
plt.show()
