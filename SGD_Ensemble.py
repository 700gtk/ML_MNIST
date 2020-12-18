from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
import numpy as np
import cv2


### 83.7% We have a SGD model that pulls the outputs of ten other SGD models, each of the ten trained on one digit, and the euiler number of the image(how many holes the digit has).
class Sgd_Ensemble(BaseEstimator):
    def __init__(self):
        self.SGD_Ensemble = []
        self.ensemble_leader = SGDClassifier(random_state=7)

    def fit(self, X, Y=None):
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
        toRet = []
        for digit in range(10):
            toRet.append(self.SGD_Ensemble[digit].decision_function(X))

        # also append a column that counts how many holes each digit has.
        toRet.append(self.get_euler_numbers(X))
        return np.array(toRet).T

    def get_euler_numbers(self, X):
        toRet = []
        for x in X:
            _, bin = cv2.threshold(x.reshape(28, 28, 1).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
            _, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            toRet.append(len(hierarchy[0]))
        return toRet

    def predict(self, X):
        return self.ensemble_leader.predict(self.predict_digits(X))
