import numpy as np
from sklearn.neural_network import MLPClassifier #MLPClassifier 人工神经网络分类器或多层感知机分类器
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:,0:2]
Y = iris.target
data = np.hstack((X,Y.reshape(Y.size,1)))#按横向列堆放数据
print('data:',data)
np.random.seed(0)
np.random.shuffle(data)
print('data:',data)
X = data[:,0:2]
print('X:',X)
print(X.shape)
Y = data[:,-1]
X_train = X[0:-30]
X_test = X[-30:]
y_train = Y[0:-30]
y_test = Y[-30:]

def plot_samples(ax,x,y):
    n_classes = 3
    plot_colors = "bry" # 颜色数组。每个类别的样本使用一种颜色
    for i, color in zip(range(n_classes), plot_colors):# zip() 函数用于将可迭代的对象作为参数,将对象中对应的元素打包成一个个元组,然后返回由这些元组组成的列表
        idx = np.where(y == i)# 把同一类的值全部拿出来
        #print('idx:',idx)
        #print(iris.target_names[i])
        # 散点图 特征数据，数据颜色，背景颜色，三种数据的标记，边框大小，字迹大小
        ax.scatter(x[idx, 0], x[idx, 1], c=color,label=iris.target_names[i]) # 绘图cmap=plt.cm.Paired相近颜色输出

def plot_classifier_predict_meshgrid(ax,clf,x_min,x_max,y_min,y_max):
    plot_step = 0.02 # 步长
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))#np.meshgrid： 会返回两个np.arange类型的列表
    #print('xx',xx)
    #print(xx.shape)
    #print('xx.ravel',xx.ravel)#xx.ravel()，将xx转变为一维数组。
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #np.c_：是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
    #将多维列表转换为一维列表，
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired) # 用于绘制填充轮廓，xx,yy,Z为坐标的横纵坐标及高度，**params是传入的图形参数。

    #让我们利用scikit-learn提供的MLPClassifier()函数进行拟合预测，绘图分析。
fig=plt.figure()#
ax=fig.add_subplot(1,1,1)#h画子图，num = 111意思就是表示一行一列，而第一个子图。同样，num = 211意思就是表示两行一列，而第一个子图。
classifier=MLPClassifier(activation='logistic',max_iter=10000,hidden_layer_sizes=(30,))# activation :激活函数,{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
#max_iter最大迭代次数
classifier.fit(X_train,y_train)
train_score=classifier.score(X_train,y_train)
test_score=classifier.score(X_test,y_test)
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 2
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 2
plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
plot_samples(ax,X_train,y_train)
ax.legend(loc='best')
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_title("train score:%f;test score:%f"%(train_score,test_score))
plt.show()

fig=plt.figure()
hidden_layer_sizes=[(10,),(30,),(100,),(5,5),(10,10),(30,30)] # 候选的 hidden_layer_sizes 参数值组成的数组
for itx,size in enumerate(hidden_layer_sizes):
    ax=fig.add_subplot(2,3,itx+1)
    classifier=MLPClassifier(activation='logistic',max_iter=10000,hidden_layer_sizes=size)
    classifier.fit(X_train,y_train)
    train_score=classifier.score(X_train,y_train)
    test_score=classifier.score(X_test,y_test)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 2
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 2
    plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
    plot_samples(ax,X_train,y_train)
    ax.legend(loc='best',fontsize='xx-small')
    ax.set_xlabel(iris.feature_names[0])
    ax.set_ylabel(iris.feature_names[1])
    ax.set_title("layer_size:%s" % str(size))
    print("layer_size:%s;train score:%.2f;test score:%.2f"%(size,train_score,test_score))
plt.show()


fig=plt.figure()
fig.set_size_inches(16,8)
ativations=["logistic","tanh","relu"]
for itx,act in enumerate(ativations):
    ax=fig.add_subplot(1,3,itx+1)
    classifier=MLPClassifier(activation=act,max_iter=10000,hidden_layer_sizes=(30,))
    classifier.fit(X_train,y_train)
    train_score=classifier.score(X_train,y_train)
    test_score=classifier.score(X_test,y_test)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 2
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 2
    plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
    plot_samples(ax,X_train,y_train)
    ax.legend(loc='best',fontsize='xx-small')
    ax.set_xlabel(iris.feature_names[0])
    ax.set_ylabel(iris.feature_names[1])
    ax.set_title("activation:%s" % act)
    print("activation:%s;train score:%.2f;test score:%.2f"%(act,train_score,test_score))
plt.show()


fig=plt.figure()
fig.set_size_inches(16,8)
solvers=["lbfgs","sgd","adam"]
for itx,solver in enumerate(solvers):
    ax=fig.add_subplot(1,3,itx+1)
    classifier=MLPClassifier(activation="tanh",max_iter=10000,hidden_layer_sizes=(30,),solver=solver)
    classifier.fit(X_train,y_train)
    train_score=classifier.score(X_train,y_train)
    test_score=classifier.score(X_test,y_test)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 2
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 2
    plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
    plot_samples(ax,X_train,y_train)
    ax.legend(loc='best',fontsize='xx-small')
    ax.set_xlabel(iris.feature_names[0])
    ax.set_ylabel(iris.feature_names[1])
    ax.set_title("solvers:%s" % solver)
    print("solvers:%s;train score:%.2f;test score:%.2f"%(solver,train_score,test_score))
plt.show()

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
import matplotlib.pyplot as plt

def plot_samples(ax,x,y):
    n_classes = 3
    plot_colors = "bry"
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        #ax.scatter(x[idx, 0], x[idx, 1], c=color,label=iris.target_names[i], cmap=plt.cm.Paired)
        ax.scatter(x[idx, 0], x[idx, 1], c=color,label=iris.target_names[i])

def plot_classifier_predict_meshgrid(ax,clf,x_min,x_max,y_min,y_max):
    plot_step = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)

iris = datasets.load_iris()
X = iris.data[:,0:2]
Y = iris.target
data = np.hstack((X,Y.reshape(Y.size,1)))
np.random.seed(0)
np.random.shuffle(data)
X = data[:,0:2]
Y = data[:,-1]
X_train = X[0:-30]
X_test = X[-30:]
y_train = Y[0:-30]
y_test = Y[-30:]

fig=plt.figure()
etas=[0.1,0.01,0.001,0.0001]
for itx,eta in enumerate(etas):
    ax=fig.add_subplot(2,2,itx+1)
    classifier=MLPClassifier(activation="tanh",max_iter=1000000,hidden_layer_sizes=(30,),solver='sgd',learning_rate_init=eta)
    classifier.fit(X_train,y_train)
    train_score=classifier.score(X_train,y_train)
    test_score=classifier.score(X_test,y_test)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 2
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 2
    plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
    plot_samples(ax,X_train,y_train)
    ax.legend(loc='best',fontsize='xx-small')
    ax.set_xlabel(iris.feature_names[0])
    ax.set_ylabel(iris.feature_names[1])
    ax.set_title("etas:%s" % eta)
    print("etas:%s;train score:%.2f;test score:%.2f"%(etas,train_score,test_score))
plt.show()
