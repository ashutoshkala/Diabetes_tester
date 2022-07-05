import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
x = pd.read_csv('Training Data/Diabetes_XTrain.csv')
y = pd.read_csv('Training Data/Diabetes_YTrain.csv')
x_test = pd.read_csv('Test Cases/Diabetes_Xtest.csv')
y_test = pd.read_csv('Test Cases/sample_submission.csv')


def dist(x1, x2):
    return np.sqrt(sum((x1-x2)**2))


def knn(X, Y, queryPoint, k=5):
    vals = []
    m = X. shape[0]
    for i in range(m):
        d = dist(queryPoint, X[i])
        vals.append((d, Y[i]))
    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]
    vals = np. array(vals)
    # print(vals)
    new_vals = np.unique(vals[:, 1], return_counts=True)
    # print(new vals)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    return int(pred)


x_train = x.values
y_train = y.values
x_te = x_test.values
y_te = x_test.values
# y_train = y_train[:, 0]
# a = knn(x_train, y_train, x_te[10], k=5)
# print(a)
# print(x_te.shape)
count = 0
for i in range(192):
    a = knn(x_train, y_train, x_te[i], k=5)
    if(a == 1):
        print("Diabetes Positive")
    else:
        print("Diabetes Negative")
