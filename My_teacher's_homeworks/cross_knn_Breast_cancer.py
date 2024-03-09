from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

breast_cancer_datas = load_breast_cancer()
breast_cancer_data = breast_cancer_datas['data']
benign_as_0_or_malignant_as_1 = breast_cancer_datas['target']

X, y = breast_cancer_data, benign_as_0_or_malignant_as_1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)

scores = []
ks = []
for k in range(2, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=7).mean()
    scores.append(score)
    ks.append(k)

max_accuracy_index = np.array(scores).argmax()
print('最大值下标：', max_accuracy_index, ' value: ', scores[max_accuracy_index])
print('最大值下标对应的K值：', ks[max_accuracy_index])

plt.plot(ks, scores)
plt.xlabel('k value')
plt.ylabel('score')
plt.show()

# Created by moyuhualuo, if u like, start my GitHub.
