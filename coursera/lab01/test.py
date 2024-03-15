import numpy as np
import matplotlib.pyplot as plt

# 定义二次方程的系数
a = 1
b = -2
c = 1

# 选择 x 的范围
x = np.linspace(-10, 10, 100)

# 计算对应的 y 值
y = a * x**2 + b * x + c

# 绘制图形
plt.plot(x, y, label=f'{a}x^2 + {b}x + {c}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Equation')
plt.legend()
plt.grid(True)
plt.show()
