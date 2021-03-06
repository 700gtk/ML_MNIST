# ML_MNIST

The MNIST data set consists of 70,000 handwritten digits. The code in this repo was used to train and build a classifier used in the public Kaggle.com 'Digit Classifier' challenge.

# SGD_Collection
### 78.4% accuracy
A collection of 10 Stochastic Gradient Descent models from Scikit learn used to classify the MNIST data set. 

# SGD_Ensemble
### OVO: 85.6% accuracy
### OVA: 83.7% accuracy
The same collection of SGD models as in SGD_Collection but with the added feature of Eulers Number.
SGD_Ensemble is a stacked ensemble with the final predictor being another SGD model.

OVO: One-V-One, a strategy to train a binary classifier to return true if one of two digits are present. This results in 45 individual classifiers and an improvement in accuracy.
OVA: One-V-All, a strategy that trains a binary classifier to classify one digit. This results in 10 classifiers, one for each digit.

# MLP Neural Net
### 97.22% accuracy after grid search
### 95.55% accuracy
A Multi layer perceptron model. This uses dense hidden layers of perceptron models culminating in a softmax final layer to accurately classify digits.
Grid search was run over different numbers of layers, neurons per layer and learning rate. The results were significant. This could be further capitalized on through even narrower gridsearches. GridsearchCV from sklearn was used and the neural net class was configured to work with it.
