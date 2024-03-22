import numpy as np
import matplotlib.pyplot as plt
from Files.lab_utils_multi import  load_house_data, run_gradient_descent
from Files.lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from Files.lab_utils_common import dlc

np.set_printoptions(precision=2)
plt.style.use(r'deeplearning.mplstyle')

# 导入数据
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']


'''
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# 获取第一行所有元素
first_row = arr[0, :]
print("第一行所有元素:", first_row)

# 获取第二列所有元素
second_column = arr[:, 1]
print("第二列所有元素:", second_column)
'''
def draw_p1():
    # 绘制4子图，共享 y 轴， marker标记,
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("Price (1000's)")
    plt.title('F(1)')

    """对不不同学习率下J（w,b）, 迭代次数, w的变化情况，通过图像工具直接观察收敛"""
    #set alpha to 9.9e-7
    _, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = 9.9e-7)
    print()
    # 迭代次数与 J（w, b）,J(w, b)与w[0]的梯度makrer
    plot_cost_i_w(X_train, y_train, hist)

    #set alpha to 9e-7
    _, _, hist = run_gradient_descent(X_train, y_train, 100, alpha = 9e-7)
    print()
    plot_cost_i_w(X_train, y_train, hist)

    #set alpha to 1e-7
    _, _, hist = run_gradient_descent(X_train, y_train, 100, alpha = 1e-7)
    print()
    plot_cost_i_w(X_train, y_train, hist)

"""引入高斯分布（Z-score）
>>>为什么引入
引入特征标准化的概念是为了解决不同特征之间尺度不同的问题。
在许多机器学习算法中，如果特征具有不同的尺度，可能会导致某些特征对模型的训练过程产生更大的影响，从而影响模型的收敛速度和性能。
通过标准化特征，我们可以消除尺度差异，使得模型能够更好地处理各个特征，并且更有效地学习特征之间的关系，从而提高模型的泛化能力和性能。
>>>标准化的方法
特征标准化是指将每个特征的值减去其均值，然后再除以其标准差，以使得特征的值具有零均值和单位方差。
这样做的目的是使得不同特征的尺度相似，从而帮助模型更好地收敛并提高模型的性能。
md文件给出理论公式
"""
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu = np.mean(X, axis=0)  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)

"""
归一化
正态分布
标准正态分布
"""
mu     = np.mean(X_train,axis=0)
sigma  = np.std(X_train,axis=0)
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma
def draw_p2():
    fig1,ax1=plt.subplots(1, 3, figsize=(12, 3))
    ax1[0].scatter(X_train[:,0], X_train[:,3])
    ax1[0].set_xlabel(X_features[0]); ax1[0].set_ylabel(X_features[3]);
    ax1[0].set_title("unnormalized")
    ax1[0].axis('equal')

    ax1[1].scatter(X_mean[:,0], X_mean[:,3])
    ax1[1].set_xlabel(X_features[0]); ax1[0].set_ylabel(X_features[3]);
    ax1[1].set_title(r"X - $\mu$")
    ax1[1].axis('equal')

    ax1[2].scatter(X_norm[:,0], X_norm[:,3])
    ax1[2].set_xlabel(X_features[0]); ax1[0].set_ylabel(X_features[3]);
    ax1[2].set_title(r"Z-score normalized")
    ax1[2].axis('equal')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.suptitle("distribution of features before, during, after normalization")

# normalize the original features
# 归一化处理与原数据对比
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

"""
归一化前后对比
"""
def draw_p3():
    fig,ax=plt.subplots(1, 4, figsize=(12, 3))
    for i in range(len(ax)):
        norm_plot(ax[i],X_train[:,i],)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("count");
    fig.suptitle("distribution of features before normalization")
    plt.show()
    fig,ax=plt.subplots(1,4,figsize=(12,3))
    for i in range(len(ax)):
        norm_plot(ax[i],X_norm[:,i],)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("count");
    fig.suptitle("distribution of features after normalization")

"""归一化，梯度递减结果"""
w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 100, 1.0e-1, )

"""
缩放后的特征可以更快地获得非常准确的结果！ 请注意，在这个相当短的运行结束时，每个参数的梯度都很小。
0.1 的学习率是使用归一化特征进行回归的良好开端。 
让我们绘制预测值与目标值的关系图。 请注意，预测是使用归一化特征进行的，而绘图是使用原始特征值显示的。
通过归一化处理，得出特征值，带入到原始图形，观测
"""
def draw_p4():
    m = X_norm.shape[0]
    yp = np.zeros(m)
    for i in range(m):
        yp[i] = np.dot(X_norm[i], w_norm) + b_norm

        # plot predictions and targets versus original features
    fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train, label = 'target')
        ax[i].set_xlabel(X_features[i])
        ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"], label = 'predict')
    ax[0].set_ylabel("Price"); ax[0].legend();
    fig.suptitle("target versus prediction using z-score normalized model")

"""
根据训练后的特征值预测样例
"""
# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")
"""
相关特征值下的图像与归一化的图像
"""
def draw_p5():
    plt_equal_scale(X_train, X_norm, y_train)
print(X_train)
# 依次为均值和标准差
print(X_train.mean(axis=0))
print(X_train.std(axis=0))
"""
>>> Drop '#' to show figure
"""
# draw_p1()
# draw_p2()
# draw_p3()
# draw_p4()
# draw_p5()

plt.show()