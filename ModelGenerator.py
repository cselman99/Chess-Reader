import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy
import Constants as constants
print("TensorFlow version:", tf.__version__)

batch_size = 32
epochs = 10
img_height = 80
img_width = 80

# Build a model from initialized training data
def buildModel(train_ds, val_ds):
    # model = Sequential([
    #     Dense(units=16, input_shape=(1,), activation='relu'),
    #     Dense(units=32, activation='relu'),
    #     Dense(units=6, activation='softmax')
    # ])
    num_classes = len(constants.PIECES)

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                       loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'],
                       run_eagerly=True)

    # Can alter input params to improve model (Ex: Increase epochs)
    fitStats = model.fit(train_ds, validation_data=val_ds, epochs=10)
    print(model.summary())
    print(fitStats)
    return model

def evaluateBoard(model, testX, testY):
    return model.evaluate(testX, testY)

def saveModel(model):
    model.save('Keras')

def loadModel(path):
    return load_model(path, compile=True)

def predict(model, value):
    return model.predict(value)

def loadDataset(path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # normalization_layer = tf.keras.layers.Rescaling(1./255)
    # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds
