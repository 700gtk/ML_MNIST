from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import support as sp


class neuralNet:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def data_into_sets(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        return self.scaler.fit_transform(x_train), self.scaler.transform(x_test), y_train, y_test

    def train(self, x, y, layers, neuron_count, epochs):
        ### build net
        self.model = keras.models.Sequential()
        for iter in range(layers):
            self.model.add(keras.layers.Dense(neuron_count, activation='relu'))

        self.model.add(keras.layers.Dense(10, activation='softmax'))

        # get the data
        x_train, x_test, y_train, y_test = self.data_into_sets(x, y)

        # compile model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=.001), metrics=["accuracy"])

        #fit the model
        self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))


    def predict(self, x):
        list_of_weights_of_predictions = self.model.predict(self.scaler.transform(x))
        best_answers = []
        # iter = 0
        for prediction_set in list_of_weights_of_predictions:
            # prediction = prediction_set.tolist().index(max(prediction_set))
            best_answers.append(prediction_set.tolist().index(max(prediction_set)))
            # sp.Plot_digit(x[iter])
            # iter += 1


        return best_answers
