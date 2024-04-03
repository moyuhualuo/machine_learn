from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建决策树模型并进行训练
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 输出原始模型的准确率
print("Original model accuracy:", clf.score(X_test, y_test))

# 使用成本复杂度剪枝，通过调整ccp_alpha参数
ccp_alphas = np.arange(0.0, 0.5, 0.05)  # 定义一系列ccp_alpha值
clfs = []  # 存储剪枝后的模型
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# 绘制不同ccp_alpha值下的模型性能
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

plt.plot(ccp_alphas, train_scores, marker='o', label='Train', drawstyle='steps-post')
plt.plot(ccp_alphas, test_scores, marker='o', label='Test', drawstyle='steps-post')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy vs ccp_alpha')
plt.legend()
plt.show()
plt.close('all')

ccp_alpha = 0.01
clf_pruned = tree.DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
clf_pruned.fit(X_train, y_train)
plt.figure(figsize=(10, 10))
plot_tree(clf_pruned, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

print(clf_pruned.score(X_test, y_test))
print(clf.score(X_test, y_test))
