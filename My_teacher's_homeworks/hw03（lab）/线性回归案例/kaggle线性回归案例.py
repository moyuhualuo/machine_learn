import numpy as np # 导入NumPy数学工具箱
import pandas as pd # 导入Pandas数据处理工具箱
# 读入数据并显示前面几行的内容，这是为了确保我们的文件读入的正确性
# 示例代码是在Kaggle中数据集中读入文件，如果在本机中需要指定具体本地路径
df_ads = pd.read_csv('advertising.csv')
print(df_ads.head())

#导入数据可视化所需要的库
import matplotlib.pyplot as plt #Matplotlib – Python画图工具库
import seaborn as sns #Seaborn – 统计学数据可视化工具库
# 显示销量和各种广告投放量的散点图
sns.pairplot(df_ads, x_vars=['wechat', 'weibo', 'others'],
                          y_vars='sales',
                          height=4, aspect=1, kind='scatter')
plt.show() # 绘图

X = np.array(df_ads.wechat) #构建特征集，只有微信广告一个特征
y = np.array(df_ads.sales) #构建标签集，销售金额
print ("张量X的阶:",X.ndim)
print ("张量X的形状:", X.shape)

X = X.reshape(-1,1) #通过reshape函数把向量转换为矩阵，len函数返回样本个数
y = y.reshape(-1,1) #通过reshape函数把向量转换为矩阵，len函数返回样本个数
print ("张量X的阶:",X.ndim)
print ("张量X的形状:", X.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                   test_size=0.2, random_state=0)

def scaler(train, test): # 定义归一化函数 ，进行数据压缩
    min = train.min(axis=0) # 训练集最小值
    max = train.max(axis=0) # 训练集最大值
    gap = max - min # 最大值和最小值的差
    train -= min # 所有数据减最小值
    train /= gap # 所有数据除以大小值差
    test -= min #把训练集最小值应用于测试集
    test /= gap #把训练集大小值差应用于测试集
    return train, test # 返回压缩后的数据
X_train,X_test = scaler(X_train,X_test) # 对特征归一化
y_train,y_test = scaler(y_train,y_test) # 对标签也归一化


# lineX = np.linspace(X_norm.min(), X_norm.max(),100)
#用之前已经导入的matplotlib.pyplot中的plot方法显示散点图
plt.plot(X_train,y_train,'r.', label='Training data')
plt.xlabel('Wechat Ads') # x轴Label
plt.ylabel('Sales') # y轴Label
plt.legend() # 显示图例
plt.show() # 显示绘图结果

def predict(weight,bias,X): # 定义预测函数
    y_hat = weight*X + bias # 这是假设函数,其中已经应用了Python的广播功能
    return y_hat # 返回预测分类的结果

def cost_function(X, y, w, b): # 手工定义一个MSE均方误差函数
    y_hat = w*X + b # 这是假设函数,其中已经应用了Python的广播功能
    loss = y_hat-y # 求出每一个y’和训练集中真实的y之间的差异
    cost = np.sum(loss**2)/len(X) # 这是均方误差函数的代码实现
    return cost # 返回当前模型的均方误差值

print ("当权重5，偏置3时，损失为：", cost_function(X_train, y_train, w=5, b=3))
print ("当权重100，偏置1时，损失为：", cost_function(X_train, y_train, w=100, b=1))

# 线性回归的梯度下降实现
def gradient_descent(X, y, w, b, lr, iter): # 定义一个实现梯度下降的函数
    l_history = np.zeros(iterations) # 初始化记录梯度下降过程中损失的数组
    w_history = np.zeros(iterations) # 初始化记录梯度下降过程中权重的数组
    b_history = np.zeros(iterations) # 初始化记录梯度下降过程中偏置的数组
    for iter in range(iterations): # 进行梯度下降的迭代，就是下多少级台阶
        y_hat  = w*X + b # 这个是向量化运行实现的假设函数
        loss = y_hat-y # 这是中间过程,求得的是假设函数预测的y和真正的y值之间的差值
        derivative_weight = X.T.dot(loss)/len(X)*2 # 对权重求导，len(X)就是数据集样本数N
        derivative_bias = sum(loss)*1/len(X)*2 # 对偏置求导，len(X)就是数据集样本数N
        w = w - lr*derivative_weight # 结合下降速率alpha更新权重
        b = b - lr*derivative_bias # 结合下降速率alpha更新偏置
        l_history[iter] = cost_function(X, y, w,b) # 梯度下降过程中损失的历史
        w_history[iter] = w # 梯度下降过程中权重的历史
        b_history[iter] = b # 梯度下降过程中偏置的历史
    return l_history, w_history, b_history # 返回梯度下降过程数据

# 定义线性回归模型 - 核心就是调用梯度下降
def linear_regression(X, y, weight, bias, alpha, iterations):
    loss_history, weight_history, bias_history = gradient_descent(X, y,
                                                                  weight, bias,
                                                                  alpha, iterations)
    print("训练最终损失:", loss_history[-1]) # 打印最终损失
    y_pred = predict(weight_history[-1],bias_history[-1],X) # 预测
    traning_acc = 100 - np.mean(np.abs(y_pred - y))*100 # 计算准确率
    print("线性回归训练准确率: {:.2f}%".format(traning_acc))  # 打印准确率
    return loss_history, weight_history, bias_history # 返回训练历史记录

iterations = 500; # 迭代1500次
alpha = 0.5; #学习速率设为1,0.5和0.01，分别试一下
weight = -5 # 权重
bias = 3 # 偏置
# 计算一下初始权重和偏置值所带来的损失
print ('当前损失：',cost_function(X_train, y_train, weight, bias))

plt.plot(X_train, y_train,'r.', label='Training data') # 显示训练集散点图
line_X = np.linspace(X_train.min(), X_train.max(), 500) # X值域
line_y = [weight*xx + bias for xx in line_X] # 假设函数y_hat
plt.plot(line_X,line_y,'b--', label='Current hypothesis' ) # 显示当前拟合函数
plt.xlabel('Wechat Ads') # x轴Label
plt.ylabel('Sales') # y轴Label
plt.legend() # 显示图例
plt.show() # 显示绘图

loss_history, weight_history, bias_history = \
   linear_regression(X_train,y_train,weight,bias,alpha,iterations)

plt.plot(loss_history,'g--',label='Loss Curve')
plt.xlabel('Iterations') # x轴Label
plt.ylabel('Loss') # y轴Label
plt.legend() # 显示图例
plt.show() # 显示损失曲线

plt.plot(X_train, y_train,'r.', label='Training data') # 显示训练集散点图
line_X = np.linspace(X_train.min(), X_train.max(), 500) # X值域
# 关于weight_history[-1],这里的索引[-1]，我们讲过，就代表迭代500次后的最后一个W值
line_y = [weight_history[-1]*xx + bias_history[-1] for xx in line_X] # 假设函数y_hat
plt.plot(line_X,line_y,'b--', label='Current hypothesis' ) # 显示当前拟合函数
plt.xlabel('Wechat Ads') # x轴Label
plt.ylabel('Sales') # y轴Label
plt.legend() # 显示图例
plt.show() # 显示绘图

print ('当前损失：',cost_function(X_train, y_train, weight_history[-1], bias_history[-1]))
print ('当前权重：',weight_history[-1])
print ('当前偏置：',bias_history[-1])

print ('测试集损失：',cost_function(X_test, y_test, weight_history[-1], bias_history[-1]))
# 同时绘制训练集和测试集损失曲线
loss_test ,a , b = gradient_descent(X_test, y_test, weight, bias, alpha, iterations)
plt.plot(loss_history,'g--',label='Traning Loss Curve')
plt.plot(loss_test,'r',label='Test Loss Curve')
plt.xlabel('Iterations') # x轴Label
plt.ylabel('Loss') # y轴Label
plt.legend() # 显示图例
plt.show()

