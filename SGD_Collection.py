from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
import numpy as np



class Sgd_collection(BaseEstimator):
    def __init__(self):
        self.SGD_Ensemble = []

    def fit(self, X, Y):
        """
        Train the data on X and Y.

        :param X: Training data.
        :param Y: Training answers.
        :return: Void, but the model is trained.
        """
        for digit in range(10):
            binary = (Y == digit)

            ### Train a classifier for each digit
            sgd_c = SGDClassifier(random_state=7)
            sgd_c.fit(X, binary)
            self.SGD_Ensemble.append(sgd_c)

    def predict(self, X):
        """
        Predict on the examples.

        :param X: Predicition data
        :return: toRet, an array of predictions.
        """
        toRet = []
        predictions = []
        for digit in range(10):
            predictions.append(self.SGD_Ensemble[digit].predict(X))

        multiple_positives = 0

        for row_of_predictions in np.array(predictions).T:
            if np.sum(row_of_predictions) > 1:
                multiple_positives += 1

            ### if we have no positive predictions, return 0
            ### if we do have positive predictions, return the first one.
            if np.sum(row_of_predictions) < 1:
                toRet.append(0)
            else:
                qq = row_of_predictions.nonzero()[0]
                toRet.append(qq[0])

        print('number of multiple positives', multiple_positives)
        return toRet

