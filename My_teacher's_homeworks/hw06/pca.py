from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

data=load_iris()
X=data['data']
y=data['target']
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_std=sc.fit_transform(X)
#print(X_std)

cov_mat=np.cov(X.T)#计算协方差矩阵
print(cov_mat)

eig_vals,eig_vecs=np.linalg.eig(cov_mat)# 求特征值和特征向量
print('eig_vals:',eig_vals)
print('eig_valuse:',eig_vecs)#特征值对应的特征向量为一列
eig_pairs=[(np.abs(eig_vals[i]),eig_vecs[:,i])for i in range(len(eig_vals))]
print(eig_pairs)
eig_pairs.sort(key=lambda x:x[0],reverse=True)
for i in eig_pairs:
    print(i[0])
#特征值越大对应的特征向量越重要

matrix_w=np.hstack((eig_pairs[0][1].reshape(4,1),
                    eig_pairs[1][1].reshape(4,1)))
print('Matrix W:',matrix_w)
Y=X_std.dot(matrix_w)
#print(Y)
ex_plained_variance_retio=eig_vals/eig_vals.sum()
print(ex_plained_variance_retio)

print(matrix_w.shape)

Y=X_std.dot(matrix_w)
print(Y)
ex_plained_variance_retio=eig_vals/eig_vals.sum()
print(ex_plained_variance_retio)
#碎石图
import matplotlib.pyplot as plt
plt.plot(np.cumsum(ex_plained_variance_retio))
plt.xlabel('principal Component(k)')
plt.ylabel('%of variance Explained<=k')
plt.show()

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(X)
X_dr = pca.transform(X)  # 获取新矩阵
print(X_dr)
#explained_variation_ratio_，表示降维后各主成分的方差值与总方差值的比值。比值越大，主成分越重要
print(pca.explained_variance_ratio_)
#explained_variance，表示降维后主成分的方差。方差值越大，主成分越重要。
print(pca.explained_variance_)#属性explained_variance_，查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
print(pca.explained_variance_ratio_.sum())#得到了比设定2个特征时更高的信息含量，

