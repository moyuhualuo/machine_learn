# 多元线性回归问题理论分析🚀️ `ing...`

> 多元线性回归是一种回归分析方法，用于研究自变量（特征）与因变量（目标）之间的线性关系，其中自变量可以有多个。在多元线性回归中，假设因变量和自变量之间的关系可以用一个线性模型来描述，即：

$$
y = w{_{1}}x{_{1}} +  w{_{2}}x{_{1}} + w{_{3}}m{_{3}}+...+w{_{n}}x{_{n}} + b
$$

其中：

- \( y \) 是因变量（目标）
- ( x ) 是自变量
- ( w ) 是模型的系数（也称为回归系数）
- ( b ) 是误差项 / 偏差项

多元线性回归的目标是找到最佳的回归系数 ，使得模型预测的值与实际观测值之间的误差最小化。通常采用最小二乘法来估计这些系数。

根据：

$$
f_{\mathbf{w},b}(\mathbf{x}) =  w_0x_0 + w_1x_1 +... + w_{n-1}x_{n-1} + b \tag{1}
$$

Numpy tool dot:

$$
f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b  \tag{2}
$$

> 简单理解为权重不同，例如房间的面积，有几室几厅，有什么设备，是几十年的房子等等，更好估测房价。

欧式误差计算：

$$
J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 \tag{3}
$$

Numpy tool dot:

$$
f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b  \tag{4}
$$

更新 w, b：

> for i in range(m):

$$
w{_i} = w{_i} - \alpha \frac{\partial }{\partial w{_j}} J(w{_i}, b) \tag{5}
$$

$$
b = b - \alpha \frac{\partial }{\partial b} J(w, b) \tag{6}
$$

$$
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{7}
$$

$$
\frac{\partial J(\mathbf{w},b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{8}
$$

> 根据梯度下降，依次求每个w对应的最佳值，和b的最佳值，只需初始化迭代更新
> 最终根据训练集得出准确度最拟合的w, b值，来通过测试集预测结果准确性是否符合

--- 
