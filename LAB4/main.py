import numpy as np


f = open("data-csv/pima-indians-diabetes.csv")
data = np.genfromtxt(fname=f, delimiter=',')
X = data[:, :-1]
y = data[:, -1]

num_training = int(0.8*len(X))
num_test = len(X) - num_training

X_train, y_train = X[:num_training], y[:num_training]

X_test, y_test = X[:num_test], y[:num_test]


if __name__ == "__main__":
    print(X_test)
