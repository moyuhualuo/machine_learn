# `Lab01🚀️week01`（无他，唯py熟尔:yum: ）

### 介绍：由于笔者学校教学内容为ML，所以自学`coursera`ML, md文档为介绍功能，方便后期更快速回顾

# Coding with Pychram.:yum:（图像运行文件自动生成）

## Linear_Regression

---

> 此文件为入门线性回归文件，大致步骤

- Numpy 随机生成 x， 由`f(x) = fixed_w * x + b(随机噪声)`
- 根据 `f(x)` 得到对应`（x, y）` **marker** 标记
- 设置fixed_w, fixed_b
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
- 设置训练数据marker
- 根据w的改变，更新`f(x)` 图像，同时`computer`计算`dis`,更新图像
- 设置差值虚线
- 观测最低dis，找到最拟合训练集的w

---

## Linear

$$
f(x) = w * x + b
$$

$$
dis = \frac{1}{2m}\times  \sum_{i = 0}^{m - 1}\left ( f(x_{i}) - y_{i} \right ) ^{2}
$$

- w, b 设置滑块，根据w, b的改变绘制 f(x) 函数图像
- 根据w,  b 的取值范围绘出 dis函数的2 D图像
- 根据w, b, dis 的取值绘制出 3D 图像

---

## Gradient_Descent

> 简单的类函数，内置api接口，方便对w, b, 以及计算 dis的使用
> 初始话参数 （x_train, y_train, w, b） 以及 computer_cost 计算函数的使用

- 设置class类方便掉包

---

## GD_fixed_b

> fixed_b 固定b =100
> w取范围值，画出 **dis** 图像

$$
f(x) = w \times x + b
$$

$$
J(w, b) =  \frac{1}{2m}\times  \sum_{i = 0}^{m - 1}\left ( f(x_{i}) - y_{i} \right ) ^{2}
$$

$$
w = w - \alpha \frac{\partial }{\partial x} J(w, b)
$$

$$
b = b - \alpha \frac{\partial }{\partial x} J(w, b)
$$

- 设置w的值为固定值2次，分别模拟 w 的**梯度下降**10次，用**红色**和**绿色**`marker`标记

---

## Contour_levels

> 根据W， B，J(w, b) 范围画出局部等高线

---

## draw_fwb

- 三个子图，便于观察梯度下降
  - 第一个，`J（w，b）, w, b`的 3D 图像，只不过增加marker标记追踪**更新后**的 J(w, b)
  - 第二个，`W， B，J(w, b)` 范围画出局部等高线，增加marker标记追踪**更新后**的J(w, b)
  - 第三个，`W，J(w, b)`绘制 2D 图形，同时marker追踪

---
