from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
import numpy as np
import cv2


class Sgd_Ensemble(BaseEstimator):
    def __init__(self):
        self.SGD_Ensemble = []
        self.ensemble_leader = SGDClassifier(random_state=7)


    def fit(self, X, Y, OVO=False):
        """
        Train the model against the X and Y values.
         - default is One-V-All, this results in an ensemble containing a classifier for each digit.
           if One-V-One is selected then there will be a classifier for every combination of two digits. This results in
           45 classifiers and an improved accuracy.

        :param X: Features
        :param Y: Answers
        :param OVO: If our model will be One-V-One or One-V-All
        :return: Void, but the model will be trained.
        """
        if OVO:
            # for each combination of two digit, create a model that compares weather the example is one of those two digits.
            for i in range(9):
                for j in range(i+1, 10):
                    binary = (Y == i) + (Y == j)

                    ### Train a classifier for each digit
                    sgd_c = SGDClassifier(random_state=7)
                    sgd_c.fit(X, binary)
                    self.SGD_Ensemble.append(sgd_c)
        else:
            # Train ten classifiers for each digit.
            for digit in range(10):
                binary = (Y == digit)

                ### Train a classifier for each digit
                sgd_c = SGDClassifier(random_state=7)
                sgd_c.fit(X, binary)
                self.SGD_Ensemble.append(sgd_c)

        # This get's the output of our ensemble
        ensemble_prediction = self.predict_digits(X)

        # Train a classifier to manage all our classifiers
        self.ensemble_leader.fit(ensemble_prediction, Y)

    def predict_digits(self, X):
        """
        Get the ensembles predictions.

        :param X: Examples to predict
        :return: toRet, an array of predictions from each model in the ensemble.
        """
        toRet = []
        for digit in range(len(self.SGD_Ensemble)):
            toRet.append(self.SGD_Ensemble[digit].decision_function(X))

        # also append a column (that will be an additional feature) that counts how many holes each digit has.
        toRet.append(self.get_euler_numbers(X))
        return np.array(toRet).T

    def get_euler_numbers(self, X):
        """
        Gets the number of holes in the digit presented.

        :param X:
        :return: toRet, an array where each element is the number of holes of that example.
        """
        toRet = []
        for x in X:
            _, bin = cv2.threshold(x.reshape(28, 28, 1).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
            _, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            toRet.append(len(hierarchy[0]))
        return toRet

    def predict(self, X):
        """
        The final prediction of the model, or the prediction of our ensemble leader.

        :param X: Examples to predict.
        :return: An array of each prediction in X.
        """
        return self.ensemble_leader.predict(self.predict_digits(X))
