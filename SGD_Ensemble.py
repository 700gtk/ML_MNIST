from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
import numpy as np

class Sgd_ensemble(BaseEstimator):
    def __init__(self):
        self.SGD_Ensemble = []

    def fit(self, X, Y=None):
        for digit in range(10):
            binary = (Y == digit)

            ### Train a classifier for each digit
            sgd_c = SGDClassifier(random_state=7)
            sgd_c.fit(X, binary)
            self.SGD_Ensemble.append(sgd_c)


    def predict(self, X):
        toRet = []
        predictions = []
        for digit in range(10):
            predictions.append(self.SGD_Ensemble[digit].predict(X))

        multiple_positives = 0

        for row_of_predictions in np.array(predictions).T:
            if np.sum(row_of_predictions) > 1:
                multiple_positives += 1


            if np.sum(row_of_predictions) < 1:
                toRet.append(0)
            else:
               qq = row_of_predictions.nonzero()[0]
               toRet.append(qq[0])

        print('number of multiple positives', multiple_positives)
        return toRet


        # for x in X:
        #     highest_confidence = 0
        #     highest_confidence_classifier = 0
        #     for digit in range(10):
        #         pred = self.SGD_Ensemble[digit].predict(x)
        #         if pred > highest_confidence:
        #             highest_confidence = pred
