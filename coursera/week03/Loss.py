import matplotlib.pyplot as plt
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
import numpy as np
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.longdouble)
y_train = np.array([0, 0, 0, 1, 1, 1], dtype=np.longdouble)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# 遍历每个点，根据类别设置颜色
for x, y in zip(x_train, y_train):
    color = 'r' if y == 1 else 'b'
    marker = 'o'
    ax.scatter(x, y, facecolors='none', edgecolors=color, marker=marker)
ax.grid(True)
plt.show()
plt.close('all')

soup_bowl()

plt_logistic_squared_error(x_train,y_train)
plt.show()
plt.close('all')

plt_two_logistic_loss_curves()
cst = plt_logistic_cost(x_train,y_train)
