import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.tree import plot_tree

breast_cancer_data = load_breast_cancer()
X = breast_cancer_data['data']
y = breast_cancer_data['target']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
leaf_node = clf.apply(X)
fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
plot_tree(clf, filled=True, feature_names = breast_cancer_data.feature_names, class_names = breast_cancer_data.target_names)
depth = clf.get_depth()
plt.show()

df = pd.DataFrame({'leaf_node': leaf_node, 'num':np.ones(len(leaf_node)).astype((int))})
print(df)
df= df.groupby(["leaf_node"]).sum().reset_index(drop=False)
print()
print(depth)
print(df)

#-------用新调整的参数训练模型------------------
clf = tree.DecisionTreeClassifier(random_state=0,max_depth=4,min_samples_leaf=3)
clf = clf.fit(X, y)
plt.figure(figsize=(12,7))
plot_tree(clf,filled=True,feature_names=breast_cancer_data.feature_names, class_names=breast_cancer_data.target_names)
depth = clf.get_depth()
leaf_node = clf.apply(X)
#-----观察各个叶子节点上的样本个数---------
df  = pd.DataFrame({"leaf_node":leaf_node,"num":np.ones(len(leaf_node)).astype(int)})
df  = df.groupby(["leaf_node"]).sum().reset_index(drop=False)
df  = df.sort_values(by='num').reset_index(drop=True)
print("\n==== 树深度：",depth," ============")
print("==各个叶子节点上的样本个数：==")
print(df)
plt.show()

#-------用新调整的参数训练模型------------------
clf = tree.DecisionTreeClassifier(random_state=0,max_depth=4,min_samples_leaf=10)
clf = clf.fit(X, y)
plt.figure(figsize=(12,7))
plot_tree(clf,filled=True,feature_names=breast_cancer_data.feature_names, class_names=breast_cancer_data.target_names)
depth = clf.get_depth()
leaf_node = clf.apply(X)
#-----观察各个叶子节点上的样本个数---------
df  = pd.DataFrame({"leaf_node":leaf_node,"num":np.ones(len(leaf_node)).astype(int)})
df  = df.groupby(["leaf_node"]).sum().reset_index(drop=False)
df  = df.sort_values(by='num').reset_index(drop=True)
print("\n==== 树深度：",depth," ============")
print("==各个叶子节点上的样本个数：==")
print(df)
plt.show()