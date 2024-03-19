import numpy as np

# 打印shape 和 dtype
def pt(a):
    print(f'a shape = {a.shape}, a data type = {a.dtype}')

# 3 行 4 列 * 2
a = np.zeros((3, 4, 2))

# 这行代码使用 NumPy 的 random.random_sample() 函数生成一个形状为 (2, 4) 的数组，其中的元素是 [0, 1) 范围内的随机浮点数。每个元素都是从均匀分布中随机抽取的。
b = np.random.random_sample((2, 4))

# 这行代码使用 NumPy 的 `arange()` 函数创建了一个一维数组 `c`，其中包含了从 0 开始到 8（不包括 9）的整数序列。
c = np.arange(9)

# 这行代码使用 NumPy 的 `random.rand()` 函数生成一个长度为 4 的一维数组，其中的元素是 [0, 1) 范围内的随机浮点数。这些随机数是从均匀分布中随机抽取的。
d = np.random.rand(4)

# 注意 点
e = np.array([5, 4, 3, 2])
f = np.array([5., 4, 3, 2])

# 断言错误 可以用于debug
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)

'''
a = np.arange(10)
c = a[2:7:1];     print("a[2:7:1] = ", c)
c = a[2:7:3];     print("a[2:7:3] = ", c)
c = a[3:];        print("a[3:]    = ", c)
c = a[:3];        print("a[:3]    = ", c)
# -3 -> -2 -> -1 
c = a[-3:];         print("a[:]     = ", c)
'''
# 求平均值
b = np.mean(a)

# np数组可以整个数组运算，但形状必须相同
'''
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a - b}")

c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)
'''
##############
# dot api   #
'''
NumPy的`dot`函数用于计算两个数组的点积（内积），即对应位置元素相乘并求和。点积操作适用于一维和二维数组，以及高维数组的最后两个轴（维度）。
在NumPy中，有几种方式可以调用dot函数：
1. 作为numpy模块中的函数使用：`numpy.dot(a, b)`
2. 数组对象的方法：`a.dot(b)`
3. 在数组对象上使用运算符 `@`： `a @ b`
这些方式都会执行相同的操作，计算数组 `a` 和 `b` 的点积，并返回结果。
如下
a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 3, 4])
print(np.dot(a, b), a.dot(b), a @ b)
'''
# 重置数组形状，-1 表示自动
'''
a = np.arange(120).reshape(4, 30)

a = np.arange(20).reshape(-1, 5)
a[0, 2:7:1]
a[:, 2:7:1]
'''