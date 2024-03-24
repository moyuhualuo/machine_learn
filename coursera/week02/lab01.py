import numpy as np
import matplotlib.pyplot as plt
from Files.lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays
'''
np.set_printoptions(precision=2)是一个NumPy函数，
用于设置打印浮点数数组时的显示精度。
在这个例子中，通过将精度设置为2，可以使得打印出来的浮点数数组只保留两位小数，以提高可读性。
'''
# create target data
def draw_p1():
    x = np.arange(0, 20, 1)
    y = 1 + x**2
    X = x.reshape(-1, 1)

    model_w,model_b = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)

    plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
    plt.plot(x,X@model_w + model_b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); plt.legend();plt.show()


def draw_p2():
    # create target data
    x = np.arange(0, 20, 1)
    y = 1 + x**2

    # Engineer features
    X = x**2      #<-- added engineered feature
    X = X.reshape(-1, 1)  #X should be a 2-D Matrix
    model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)

    plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
    plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

def draw_p3():
    # create target data
    x = np.arange(0, 20, 1)
    y = x ** 2

    # engineer features .
    X = np.c_[x, x ** 2, x ** 3]  # <-- added engineered feature
    print(X)
    model_w, model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)

    plt.scatter(x, y, marker='x', c='r', label="Actual Value");
    plt.title("x, x**2, x**3 features")
    plt.plot(x, X @ model_w + model_b, label="Predicted Value");
    plt.xlabel("x");
    plt.ylabel("y");
    plt.legend();
    plt.show()

def draw_p4():
    # create target data
    x = np.arange(0, 20, 1)
    y = x ** 2

    # engineer features .
    X = np.c_[x, x ** 2, x ** 3]  # <-- added engineered feature
    X_features = ['x', 'x^2', 'x^3']
    fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X[:, i], y)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("y")
    plt.show()

def draw_p5_Znorm():
    # create target data
    x = np.arange(0, 20, 1)
    X = np.c_[x, x ** 2, x ** 3]
    print(f"Peak to Peak range by column in Raw        X:{np.ptp(X, axis=0)}")

    # add mean_normalization
    X = zscore_normalize_features(X)
    print(f"Peak to Peak range by column in Normalized X:{np.ptp(X, axis=0)}")
    x = np.arange(0, 20, 1)
    y = x ** 2

    X = np.c_[x, x ** 2, x ** 3]
    X = zscore_normalize_features(X)

    model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)

    plt.scatter(x, y, marker='x', c='r', label="Actual Value");
    plt.title("Normalized x x**2, x**3 feature")
    plt.plot(x, X @ model_w + model_b, label="Predicted Value");
    plt.xlabel("x");
    plt.ylabel("y");
    plt.legend();
    plt.show()

def draw_p6():
    x = np.arange(0, 20, 1)
    y = np.cos(x / 2)

    X = np.c_[x, x ** 2, x ** 3, x ** 4]
    X = zscore_normalize_features(X)

    model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha=1e-1)

    plt.scatter(x, y, marker='x', c='r', label="Actual Value");
    plt.title("Normalized x x**2, x**3 feature")
    plt.plot(x, X @ model_w + model_b, label="Predicted Value");
    plt.xlabel("x");
    plt.ylabel("y");
    plt.legend();
    plt.show()

draw_p6()