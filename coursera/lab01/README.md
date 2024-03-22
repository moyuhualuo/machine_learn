# `Lab01🚀️week01`（无他，唯 py 熟尔:yum: ）

### 介绍：由于笔者学校教学内容为 ML，所以自学`coursera`ML, md 文档为介绍功能，方便后期更快速回顾

# Coding with Pychram.:yum:（图像运行文件自动生成）

## Linear_Regression

---

> 此文件为入门线性回归文件，大致步骤

- Numpy 随机生成 x， 由`f(x) = fixed_w * x + b(随机噪声)`
- 根据 `f(x)` 得到对应`（x, y）` **marker** 标记
- 设置 fixed_w, fixed_b
- 根据`computer_cost` 函数得到固定 `f(x)`线性函数
- 根据`plt`画出线性函数，观测

---

## Linear_without_b

$$
图像f(x) = w * x
$$

$$
图像dis = \frac{1}{2m}\times  \sum_{i = 0}^{m - 1}\left ( f(x_{i}) - y_{i} \right ) ^{2}
$$

- 根据 w 的范围设置滑块
- 设置训练数据 marker
- 根据 w 的改变，更新`f(x)` 图像，同时`computer`计算`dis`,更新图像
- 设置差值虚线
- 观测最低 dis，找到最拟合训练集的 w

---

## Linear

$$
f(x) = w * x + b
$$

$$
dis = \frac{1}{2m}\times  \sum_{i = 0}^{m - 1}\left ( f(x_{i}) - y_{i} \right ) ^{2}
$$

- w, b 设置滑块，根据 w, b 的改变绘制 f(x) 函数图像
- 根据 w, b 的取值范围绘出 dis 函数的 2 D 图像
- 根据 w, b, dis 的取值绘制出 3D 图像

---

## Gradient_Descent

> 简单的类函数，内置 api 接口，方便对 w, b, 以及计算 dis 的使用
> 初始话参数 （x_train, y_train, w, b） 以及 computer_cost 计算函数的使用

- 设置 class 类方便掉包

---

## GD_fixed_b

> fixed_b 固定 b =100
> w 取范围值，画出 **dis** 图像

$$
f(x) = w \times x + b
$$

$$
J(w, b) =  \frac{1}{2m}\times  \sum_{i = 0}^{m - 1}\left ( f(x_{i}) - y_{i} \right ) ^{2}
$$

$$
w = w - \alpha \frac{\partial }{\partial w} J(w, b)
$$

$$
b = b - \alpha \frac{\partial }{\partial b} J(w, b)
$$

- 设置 w 的值为固定值 2 次，分别模拟 w 的**梯度下降**10 次，用**红色**和**绿色**`marker`标记

---

## Contour_levels

> 根据 W， B，J(w, b) 范围画出局部等高线

---

## draw_fwb

- 三个子图，便于观察梯度下降
  - 第一个，`J（w，b）, w, b`的 3D 图像，只不过增加 marker 标记追踪**更新后**的 J(w, b)
  - 第二个，`W， B，J(w, b)` 范围画出局部等高线，增加 marker 标记追踪**更新后**的 J(w, b)
  - 第三个，`W，J(w, b)`绘制 2D 图形，同时 marker 追踪

---
