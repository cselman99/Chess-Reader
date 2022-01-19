import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy


import Constants as constants
print("TensorFlow version:", tf.__version__)


class ModelGenerator:

    def __init__(self):
        self.model = None

    # Build a model from initialized training data
    def buildModel(self, trainX, trainY):
        self.model = Sequential([
            Dense(units=16, input_shape=(1,), activation='relu'),
            Dense(units=32, activation='relu'),
            Dense(units=6, activation='softmax')
        ])

        print(self.model.summary())

        self.model.compile(optimizer=Adam(learning_rate=1e-4),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'],
                           run_eagerly=True)

        # Can alter input params to improve model (Ex: Increase epochs)
        fitStats = self.model.fit(trainX, trainY, batch_size=10, epochs=5, shuffle=True, verbose=1)
        print(fitStats)

    def evaluateBoard(self, testX, testY):
        return self.model.evaluate(testX, testY)

    def saveModel(self):
        self.model.save('Keras')

    def loadModel(self):
        self.model = load_model(constants.MODEL_PATH, compile=True)

    def predict(self, values):
        return self.model.predict(values)
