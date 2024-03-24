import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from Files.lab_utils_multi import load_house_data
from Files.lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')
# 数据导入
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']
# 归一化
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
# 回归模型创建和拟合
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
# 得出权重和偏差
b_norm = sgdr.intercept_
w_norm = sgdr.coef_

# np.dot 和 sk api的预测数据对比
# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b.
y_pred = np.dot(X_norm, w_norm) + b_norm
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")
print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

# 预测与原始训练集对比
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
