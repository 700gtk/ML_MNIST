import SGD_Ensemble
import support as sp
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
import cv2
import numpy as np


### Get data
X, Y = sp.Read_in_data('Data/train.csv')
X_test, _ = sp.Read_in_data('Data/test.csv', test=True)

# sgd_e = SGD_Collection.Sgd_collection()
# sgd_e.fit(X, Y)
# predictions = sgd_e.predict(X_test)
# sp.predictions_to_submission('SGD_ensemble_predicitons', predictions)

sgd_e = SGD_Ensemble.Sgd_Ensemble()
sgd_e.fit(X, Y, OVO=True)

score = sp.Test(sgd_e.predict(X), Y)
print('accuracy on this thing:', score)

predictions = sgd_e.predict(X_test)
sp.predictions_to_submission('SGD_predictions_with_each_model_being_positive_if_it_sees_one_of_two_digits', predictions)


# score_f1 = f1_score(binary_5, sgd_c.predict(X))
# print('accuracy on this thing:', score_f1)


