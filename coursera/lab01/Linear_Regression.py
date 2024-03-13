###################
# HOUSING PRICES  #
###################

import numpy as np
import matplotlib.pyplot as plt


def compute_model_output(x, w, b):

    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    ''' or code:
    f_wb = w * x + b
    return f_wb
    '''
    return f_wb


num_samples = 100
# x means square_meters
x = np.random.uniform(10, 300, num_samples)
# y means prices
y = 1000 * x + np.random.normal(0, 20000, num_samples)
# output 1 ~ 10
for i in range(10):
    print('square_meters: ', x[i], 'prices: ', y[i])
# show plt.
# f(x) = w * x + b // here w = 1000, b = 500
w, b = 1000, 500
tmp_f_wb = compute_model_output(x, 1000, 500)

plt.plot(x, tmp_f_wb, c='b', label='Prediction')
plt.scatter(x, y, marker='x', c='r')
plt.title('Housing prices')
plt.ylabel('Prices (RMB)')
plt.xlabel('square_meters')
plt.show()

x_i = 1600
cost = w * x_i + b
print(f'{x_i} meters housing prices: ', cost)
