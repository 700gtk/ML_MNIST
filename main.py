import SGD_Ensemble
import support as sp
import neuralNet as nueralN
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
import cv2
import numpy as np
# 0.8711874092240292 best score

print('starting')

### Get data
X, Y = sp.Read_in_data('Data/train.csv')
X_test, _ = sp.Read_in_data('Data/test.csv', test=True)
print('read in data')

### skelefy features
# X = sp.skelefy(X)
# X_test = sp.skelefy(X_test)
# sp.join_count(X)


### SGD Collection
# sgd_e = SGD_Collection.Sgd_collection()
# sgd_e.fit(X, Y)
# predictions = sgd_e.predict(X_test)
# sp.predictions_to_submission('SGD_ensemble_predicitons', predictions)
# print('Skelify done')


### SGD Ensemble
# sgd_e = SGD_Ensemble.Sgd_Ensemble()
# sgd_e.fit(X, Y, OVO=True)

# score = sp.Test(sgd_e.predict(X), Y)
# print('accuracy on this thing:', score)


### Neural Net
nn = nueralN.neuralNet()
nn.train(X, Y)


predictions = nn.predict(X_test)
### Make a submission with this name and those predictions
sp.predictions_to_submission('single_layer_MLP', predictions)



### F1 score
# score_f1 = f1_score(binary_5, sgd_c.predict(X))
# print('accuracy on this thing:', score_f1)


