import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


class pimaIndiansDiabetesClassifier():

    def __init__(self):
        """
        Support vector classification model Teaching and training set
        """
        X, y = self.dataSet()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        svc = svm.SVC(kernel='rbf').fit(X, y)

        # Train the model using the training sets
        svc.fit(X_train, y_train)

        self.svc = svc

    def dataSet(self):
        """
        Loads pima data to be useful for SVM
        """
        file = open("data-csv/pima-indians-diabetes.csv")
        data = np.genfromtxt(fname=file, delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]
        return X, y

    def pred(self, data):
        # Predict the output
        return int(self.svc.predict([data])[0])
