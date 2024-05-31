### Homework 01

---

# K-NN 实现（只了解过程，使用[scikit-learn](https://scikit-learn.org/)`api`实现，并不注重数学推导）

---

## 导入癌细胞数据集，根据细胞呈阴阳，使用K-NN `api`，预测准确率

> 为了方便观察，均借助图像工具实现

- *Breast_cancer.py*
  
  - 根据不同k值预测结果，对比训练集得出准确率

- *cross_knn_Breast_cancer.py*
  
  - K-NN 交叉验证，设置cv值，根据不同的k值预测结果（取平均值），对比训练集得出准确率

---

一般使用欧式距离：

$$
Distance(x_i,x_j) = \sqrt{\sum_{k=1}^{n} (x_{i_k}-x_{j_k})^2}
$$

用距离衡量样本相似度，然后根据

$$
max\{p(x_i),p(x_j)...\} \\
where: p(x_i) = \frac{Number(x_i)}{K} 
$$

$Number(x_i)$ 表示是预测为$x_i$的个数总和