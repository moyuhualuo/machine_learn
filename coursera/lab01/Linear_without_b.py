import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
"""
>>> 对于函数图像，w值改变，两点值改变，直线也改变，从w，b改变，但我们并没用改变b。

"""
# 二次函数
def compute(x, y, w_array):
    arr = np.zeros_like(w_array)  # 创建一个与 w_array 相同形状的数组，用于存储每个 w 对应的误差
    for j, w in enumerate(w_array):
        m = x.shape[0]  # 获取样本数量
        tol = 0
        for i in range(m):
            tol += (w * x[i] - y[i]) ** 2
        arr[j] = (1 / (2 * m)) * tol
    return arr
#更新函数
def update(val):
    w = slider.val
    y = w * x_train
    line.set_ydata(y)  # 重新设置y值
    # 删除上一次的差值记录
    for arrow in ax[0].patches:
        arrow.remove()
    for text in ax[0].texts:
        text.remove()

    for arrow in ax[1].patches:
        arrow.remove()
    for text in ax[1].texts:
        text.remove()

    for i in range(len(x_train)):
        # 误差坐标
        dx = x_train[i] * 0.01  # 设置箭头位置
        dy = w * x_train[i] * 0.01
        # 设置误差线
        ax[0].annotate('', xy=(x_train[i], w * x_train[i]), xytext=(x_train[i], y_train[i]),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='gray', lw=1),
                            fontsize=8)
        # 设置误差
        error = np.abs(y_train[i] - w * x_train[i])
        ax[0].text(x_train[i] + dx, w * x_train[i] + dy, 'er:'+str(error), fontsize=8)
    '''
        ax[1].annotate('', xy=(w, 0), xytext=(w, y_range[int(w)]),
                       arrowprops=dict(arrowstyle='<-', connectionstyle='arc3', color='gray', lw=1),
                       fontsize=8)

        ax[1].annotate('', xy=(0, y_range[int(w)]), xytext=(w, y_range[int(w)]),
                       arrowprops=dict(arrowstyle='<-', connectionstyle='arc3', color='gray', lw=1),
                       fontsize=8)
    '''
    # 在update函数中，获取与 w 最接近的索引
    nearest_index = np.abs(w_range - w).argmin()
    # 使用最接近的索引来确定箭头的位置
    ax[1].annotate('', xy=(w, 0), xytext=(w_range[nearest_index], y_range[nearest_index]),
                   arrowprops=dict(arrowstyle='-', connectionstyle='arc3', color='gray', lw=1),
                   fontsize=8)

    ax[1].annotate('', xy=(0, y_range[nearest_index]), xytext=(w_range[nearest_index], y_range[nearest_index]),
                   arrowprops=dict(arrowstyle='-', connectionstyle='arc3', color='gray', lw=1),
                   fontsize=8)
    ax[1].text(w, y_range[nearest_index], f'point:{(w, y_range[nearest_index])}', fontsize=8)
    fig.canvas.draw_idle()  # 重画图像
# 初始化数据
x_train = np.array([1.0, 2.0])
y_train = np.array([250.0, 500.0])

# 创建一个图形窗口和子图
fig, ax = plt.subplots(1, 2)
plt.subplots_adjust(bottom=0.25)

# 设置初始的 w 值
initial_w = 0.0

# 绘制初始的预测结果
y = initial_w * x_train
line, = ax[0].plot(x_train, y, c='b', label='Prediction')
ax[0].scatter(x_train, y_train, marker='x', c='r')

# 二次函数
w_range = np.arange(-50, 501, 1)
y_range = compute(x_train, y_train, w_range)
# 创建二次函数图像
lin, = ax[1].plot(w_range, y_range, label='f = wx')
ax[1].set_xlabel('W')
ax[1].set_ylabel('dis')
ax[0].legend()
ax[1].legend()
'''
def compute_a(x, w):
    m = x.shape[0]
    y_temp = w * x
    res = 0
    for i in range(len(y_temp)):
        res += (1 / (2 * m)) * (y_temp[i] - y_train[i]) ** 2
    return res

line_dis = ax[1].plot(initial_w, compute_a(x_train, initial_w), c='b', label='distance')
'''
# 创建一个滑块控件
slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(slider_ax, 'w', -50, 500, valinit=initial_w, valstep=0.1)
# 将更新函数与滑块控件关联
slider.on_changed(update)
# 显示图形
plt.show()
