from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


class neuralNet:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def data_into_sets(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        return self.scaler.fit_transform(x_train), self.scaler.transform(x_test), y_train, y_test

    def train(self, x, y):
        ### build net
        input = keras.layers.Input(shape=x[1:])
        hidden_layer = keras.layers.Dense(30, activation="relu")(input)
        out_put = keras.layers.Dense(10)(hidden_layer)
        self.model = keras.Model(inputs=[input], outputs=[out_put])

        # get the data
        x_train, x_test, y_train, y_test = self.data_into_sets(x, y)

        self.model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=.001))

        history = self.model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

    def predict(self, x):
        return self.model.predict(self.scaler.transform(x))
