from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import sklearn
import support as sp


class neuralNet(sklearn.base.BaseEstimator):
    def __init__(self, layers=3, neuron_count=1000, epochs=40, learning_rate=.1):
        self.model = None
        self.scaler = StandardScaler()
        self.layers = layers
        self.neuron_count = neuron_count
        self.epochs = epochs
        self.learning_rate = learning_rate

    def data_into_sets(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        return self.scaler.fit_transform(x_train), self.scaler.transform(x_test), y_train, y_test

    def fit(self, x, y):
        ### build net
        self.model = keras.models.Sequential()
        for iter in range(self.layers):
            self.model.add(keras.layers.Dense(self.neuron_count, activation='relu'))

        self.model.add(keras.layers.Dense(10, activation='softmax'))

        # get the data
        x_train, x_test, y_train, y_test = self.data_into_sets(x, y)

        # compile model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=self.learning_rate), metrics=["accuracy"])

        #fit the model
        print('layers:', self.layers, 'neurons', self.neuron_count, 'learning_rate', self.learning_rate)
        self.model.fit(x_train, y_train, epochs=self.epochs, validation_data=(x_test, y_test), verbose=2)

    def predict(self, x):
        # this is just for gridSearchCV
        if self.model is None:
            return [0]*len(x)

        list_of_weights_of_predictions = self.model.predict(self.scaler.transform(x))
        best_answers = []
        # iter = 0
        for prediction_set in list_of_weights_of_predictions:
            # prediction = prediction_set.tolist().index(max(prediction_set))
            best_answers.append(prediction_set.tolist().index(max(prediction_set)))
            # sp.Plot_digit(x[iter])
            # iter += 1


        return best_answers
