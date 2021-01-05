import SGD_Ensemble
from sklearn.model_selection import GridSearchCV
import support as sp
import neuralNet as nueralN
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
import cv2
import numpy as np

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
# nn = nueralN.neuralNet(1, 30, 40)  # 93% accuracy
# nn = nueralN.neuralNet(3, 100, 100)  # 95.5% accuracy
nn = nueralN.neuralNet()

# nn.fit(X, Y, 1, 30, 40)  # 93% accuracy
# nn.fit(X, Y, 3, 100, 100)  # 95.5% accuracy
nn.fit(X, Y)

## GridSearch
## BEST parameters layers:3, lr: 0.1,neuron_count:1000
# parameters = {
#
#   'layers': [1, 3, 5, 7, 11],
#   'neuron_count': [50, 100, 500, 1000],
#   'learning_rate': [.1, .001, .0001]
#   }
# gd = GridSearchCV(nn, parameters, scoring="accuracy")
# gd.fit(X, Y)
#
# print('grid search best parameters', gd.best_params_)

predictions = nn.predict(X_test)
### Make a submission with this name and those predictions
sp.predictions_to_submission('grid_search_best_mlp_40_epochs', predictions)



### F1 score
# score_f1 = f1_score(binary_5, sgd_c.predict(X))
# print('accuracy on this thing:', score_f1)


