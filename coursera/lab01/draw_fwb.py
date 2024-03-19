import Gradient_Descent as GD
import matplotlib.pyplot as plt

"""fixed_b = 100
>>> draw J(w, b) of W
>>> 0.5m * sum (w * xi - yi + b) ** 2  // sum 代表求和符号
"""
cost_arr = []

for w in GD.my_instance.w:
    cost = GD.my_instance.compute_cost(w)
    cost_arr.append(cost)

print(len(cost_arr))
plt.plot(GD.my_instance.w, cost_arr)
plt.xlabel('w')
plt.ylabel('Cost')

plt.grid(True)
plt.legend(['J(w, b) of W'])
plt.show()