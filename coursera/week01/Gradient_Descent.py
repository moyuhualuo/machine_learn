import numpy as np


class MyClass:

    def __init__(self, x, y, w = None, fixed_b = None):
        self.b = fixed_b
        self.x = x
        self.y = y
        self.w = w

    def get_x_y(self):
        return self.x, self.y

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_b(self, b):
        self.b = b

    def set_w(self, w):
        self.w = w

    def get_b(self):
        return self.b

    # 固定 w, b 计算 cost
    # 误差计算
    def compute_cost(self, w_val):
        m = self.x.shape[0]
        cost = 0

        for i in range(m):
            f_wb = w_val * self.x[i] + self.b
            cost = cost + (f_wb - self.y[i]) ** 2

        total_cost = (1 / (2 * m)) * cost
        return total_cost


x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
w_range = np.linspace(-50.0, 500, 500)
my_instance = MyClass(x_train, y_train, w_range, 100)

