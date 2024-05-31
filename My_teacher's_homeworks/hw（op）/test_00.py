import numpy as np
from sklearn.model_selection import train_test_split

# 生成随机数据
np.random.seed(43)
num_samples = 50
X1_train = np.random.randn(num_samples) * 2 + 5  # 特征X1，均值为5，标准差为2
X2_train = np.random.randn(num_samples) * 3 + 10  # 特征X2，均值为10，标准差为3

y_train = np.zeros(num_samples)
y_train[X1_train + X2_train > 15] = 1

# 计算 P(y)
y1_samples = np.sum(y_train)
y0_samples = num_samples - y1_samples
p_y0 = y0_samples / num_samples
p_y1 = y1_samples / num_samples
print("P(y=0):", p_y0)
print("P(y=1):", p_y1)

# 计算 P(x1, x2|y)
x1_values = np.unique(X1_train)  # 特征 X1 的取值
x2_values = np.unique(X2_train)  # 特征 X2 的取值
p_x_y0 = np.zeros((len(x1_values), len(x2_values)))
p_x_y1 = np.zeros((len(x1_values), len(x2_values)))

for i, x1 in enumerate(x1_values):
    for j, x2 in enumerate(x2_values):
        count_y0 = np.sum((X1_train == x1) & (X2_train == x2) & (y_train == 0))
        count_y1 = np.sum((X1_train == x1) & (X2_train == x2) & (y_train == 1))
        p_x_y0[i, j] = count_y0 / y0_samples
        p_x_y1[i, j] = count_y1 / y1_samples

# 对概率取对数并映射到 (0, pi/2) 范围
p_y0_log = np.log(p_y0)
p_y1_log = np.log(p_y1)
p_x_y0_log = np.log(p_x_y0)
p_x_y1_log = np.log(p_x_y1)

# 映射到 (0, pi/2) 范围
p_y0_mapped = np.arcsin(p_y0_log) * (2 / np.pi)
p_y1_mapped = np.arcsin(p_y1_log) * (2 / np.pi)
p_x_y0_mapped = np.arcsin(p_x_y0_log) * (2 / np.pi)
p_x_y1_mapped = np.arcsin(p_x_y1_log) * (2 / np.pi)


# 定义预测函数
def predict(x1, x2):
    # 获取 x1, x2 对应的概率值索引
    i = np.where(x1_values == x1)[0][0]
    j = np.where(x2_values == x2)[0][0]

    # 计算 P(y|x1,x2)，并映射到 (0, 10^6*sin(pi/2)) 范围
    p_y0_x = p_y0_mapped + p_x_y0_mapped[i, j]
    p_y1_x = p_y1_mapped + p_x_y1_mapped[i, j]
    p_y0_x_mapped = 10 ** 6 * np.sin(p_y0_x)
    p_y1_x_mapped = 10 ** 6 * np.sin(p_y1_x)

    # 根据概率比较确定分类结果
    if p_y0_x_mapped > p_y1_x_mapped:
        return 0
    else:
        return 1


# 测试模型
num_test_samples = 10
X1_test = np.random.randn(num_test_samples) * 2 + 5
X2_test = np.random.randn(num_test_samples) * 3 + 10
y_test = np.zeros(num_test_samples)
y_test[X1_test + X2_test > 15] = 1

correct_predictions = 0
for i in range(num_test_samples):
    y_pred = predict(X1_test[i], X2_test[i])
    if y_pred == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / num_test_samples
print("Accuracy:", accuracy)