from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

"""breast cancer datas from sklearn 14.2v

1. `'data'`: 这个键对应的值是一个二维数组，包含了数据集中的样本特征数据。每一行代表一个样本，每一列代表一个特征。
2. `'target'`: 这个键对应的值是一个一维数组，包含了数据集中每个样本的标签。在癌细胞数据集中，这些标签可能表示样本的类别，比如良性或恶性。
3. `'frame'`: 这个键对应的值通常是一个 Pandas DataFrame 对象，可能包含了整个数据集的数据和标签。
4. `'target_names'`: 这个键对应的值是一个数组或列表，包含了数据集中每个类别的名称。在癌细胞数据集中，可能包含良性和恶性两个类别。
5. `'DESCR'`: 这个键对应的值是一个字符串，包含了关于数据集的描述信息，如数据集的来源、特征的含义等。
6. `'feature_names'`: 这个键对应的值是一个数组或列表，包含了数据集中每个特征的名称，描述了每个特征所代表的含义。
7. `'filename'`: 这个键对应的值是一个字符串，表示数据集文件的路径或文件名。
8. `'data_module'`: 这个键对应的值是一个模块对象，包含了加载数据集所用的模块的信息。
"""
breast_cancer_datas = load_breast_cancer()
breast_cancer_data = breast_cancer_datas['data']
benign_as_0_or_malignant_as_1 = breast_cancer_datas['target']

'''打印数组，二维用大写，一维用小写'''
print(breast_cancer_data.shape, benign_as_0_or_malignant_as_1.shape)
"""
X：特征数据，是一个二维数组或 DataFrame，每行代表一个样本，每列代表一个特征。
y：标签数据，是一个一维数组，每个元素代表对应样本的标签或目标值。
test_size：测试集的大小，可以是一个浮点数（表示测试集占总数据集的比例）或整数（表示测试集的样本数量）。
random_state：随机种子，用于控制数据的随机划分，保证可复现性。

该函数会返回四个数组或 DataFrame：
>>> X_train：训练集特征数据
>>> X_test：测试集特征数据
>>> y_train：训练集标签数据
>>> y_test：测试集标签数据
>>>训练集（Training Set）：训练集用于训练模型。我们将模型拟合到训练集上，使其学习特征和标签之间的关系。
>>>测试集（Test Set）：测试集用于评估模型的性能。一旦模型在训练集上训练完成，我们使用测试集来测试模型对新样本的泛化能力。换句话说，测试集提供了一个真实环境下的评估标准。"""
X, y = breast_cancer_data, benign_as_0_or_malignant_as_1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''该模型会考虑每个样本的最近 8 个邻居的标签'''
knn = KNeighborsClassifier(n_neighbors=8)

'''使用模型'''
knn_model = knn.fit(X_train, y_train)

X_test_predict = knn.predict(X_test)
print('模型预测结果：', X_test_predict)
print('真实分类结果', y_test)

accuracy = knn_model.score(X_test, y_test)
print(accuracy)

accuracies = []
ks = []
for i in range(2, 100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    accuracies.append(accuracy)
    ks.append(i)

plt.plot(ks, accuracies)
plt.xlabel('K value')
plt.ylabel('accuracy')
plt.show()

max_accuracy_index = np.array(accuracies).argmax()
print('最大值下标：', max_accuracy_index)
print('最大值下标对应的K值：', ks[max_accuracy_index])

# Created by moyuhualuo, if u like, star my GitHub.
