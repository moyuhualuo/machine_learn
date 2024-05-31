import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import plot_tree

datas = load_iris()
X = datas['data']
y = datas['target']
print(X.shape, datas.feature_names)

clf = tree.DecisionTreeClassifier(min_samples_split=10, ccp_alpha = 0.125)
clf.fit(X, y)

pruning_path = clf.cost_complexity_pruning_path(X, y)
print(pruning_path)
plt.plot(range(len(pruning_path['impurities'])), pruning_path['impurities'], color='blue', marker='o', markerfacecolor='r', linestyle='-')
plt.grid(True)
plt.xlabel('$alpha_i$')
plt.ylabel('impurities')
plt.show()
plt.close('all')

plot_tree(clf, filled=True, feature_names=datas.feature_names, class_names=datas.target_names)
plt.show()

is_leaf = clf.tree_.children_left == -1
tree_impurities = (clf.tree_.impurity[is_leaf]* clf.tree_.n_node_samples[is_leaf]/len(y)).sum()
print(tree_impurities)
