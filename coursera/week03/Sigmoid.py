import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


x = np.array([-1, 0, 1, 2, 3, 4, 5])
y = np.exp(x)
print(np.c_[x, y])

out = sigmoid(x)
print(np.c_[x, out])

fig, (ax0, ax) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

colors = ['b' if val < 0.5 else 'r' for val in x]
ax0.axhline(y=1, color='g', linestyle='--')
ax0.scatter(x, out, c=colors, marker='o', edgecolors=colors, fc='None', alpha=1)
ax0.set_xlabel('x')
ax0.set_ylabel('Sigmoid')
ax0.grid(True)

z = np.linspace(-10, 10, 100)

ax.axis([-10, 10, -0.1, 1.1])

ax.plot(z, sigmoid(z), c="b")
ax.axvline(x=0, color='g', linestyle='--', label='x=0, left=red')
ax.fill_between(z, 1, where=(z <= 0), color='r', alpha=0.5)
ax.fill_between(z, 1, where=(z > 0), color='g', alpha=0.5)

ax.set_xlabel('z')
ax.set_ylabel('sigmoid(z)')
ax.set_title('Sigmoid Function')
plt.show()
