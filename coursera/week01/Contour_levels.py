import numpy as np
import matplotlib.pyplot as plt
# 类的完美性和掉包使用的简便
'''
import Linear
w = Linear.w_range
b = Linear.b_range
y = Linear.y_range
由于没用很好设置类，如果掉包，将重新设置代码结构，至此我深深体会到了Class的重要性，折断的骨才是最好的说教
'''
def compute(x, y, w_array, b_array):
    m = x.shape[0]
    total_costs = np.zeros((len(w_array), len(b_array)))
    for i, w in enumerate(w_array):
        for j, b in enumerate(b_array):
            cost_sum = 0
            for k in range(m):
                f_wb = w * x[k] + b
                cost = (f_wb - y[k]) ** 2
                cost_sum += cost
            total_costs[i, j] = (1 / (2 * m)) * np.sum(cost_sum)

    return total_costs
x_train = np.array([1.0, 2.0])
y_train = np.array([250.0, 500.0])
w_range = np.arange(-50, 501, 1)
b_range = np.arange(-50, 500, 1)
y_range = compute(x_train, y_train, w_range, b_range)

W, B = np.meshgrid(w_range, b_range)

contour_levels = np.linspace(np.min(y_range), np.max(y_range), 100)
plt.contour(W, B, y_range.T, levels=contour_levels, cmap='terrain')

plt.show()

