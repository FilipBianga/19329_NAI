import numpy as np
from sklearn import svm


class transfusionClassifier():

    def __init__(self):
        """
        Support vector classification model Teaching and training set
        """
        X, y = self.dataSet()
        num_training = int(0.8 * len(X))
        X_train, y_train = X[:num_training], y[:num_training]

        svc = svm.SVC(kernel='rbf').fit(X, y)

        # Train the model using the training sets
        svc.fit(X_train, y_train)

        self.svc = svc

    def dataSet(self):
        """
        Loads transfusion data to be useful for SVM
        """
        file = open("data-csv/transfusion.csv")
        data = np.genfromtxt(fname=file, delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]
        return X, y

    def pred(self, data):
        # Predict the output
        return int(self.svc.predict([data])[0])