from sklearn.linear_model import SGDClassifier
import SGD_Ensemble
from sklearn.metrics import f1_score
import support as sp
import matplotlib.pyplot as plt
import numpy as np

### Get data
X, Y = sp.Read_in_data('Data/train.csv')
X_test, _ = sp.Read_in_data('Data/test.csv', test=True)


sgd_e = SGD_Ensemble.Sgd_ensemble()
sgd_e.fit(X, Y)
predictions = sgd_e.predict(X_test)

sp.predictions_to_submission('SGD_ensemble_predicitons', predictions)

# score_f1 = f1_score(binary_5, sgd_c.predict(X))
# print('accuracy on this thing:', score_f1)


