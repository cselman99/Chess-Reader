# NOTE: ChessDriver is for independant training of the Computer Vision CNN Model

import sys
from numpy import int32
from sklearn.preprocessing import MinMaxScaler
import Constants as constants
from os import listdir
from os.path import isfile, join
import numpy as np
from ImageReader import ImageReader
from ModelGenerator import ModelGenerator

# Uses the consistent filename scheme to determine the piece.
# Ex: if filename is "rook9128.png" --> return "rook"
def extractPieceFromFilename(filename):
    for piece in constants.PIECES:
        if piece in filename:
            return piece
    return None

def trainingSetGeneration():
    # Format training set
    trainX, trainY = [], []
    trainingFiles = [join(constants.TRAINING_PATH, f)
                     for f in listdir(constants.TRAINING_PATH)
                     if isfile(join(constants.TRAINING_PATH, f))]

    reader = ImageReader()
    for file in trainingFiles:

        reader.setFilename(file)
        reader.loadImageGray()
        name = extractPieceFromFilename(file)

        if name is not None:
            imageData = reader.getImage().astype(int32)
            trainX.append(imageData.flatten())
            trainY.append(name)

    trainX = np.asarray(trainX)
    trainY = np.asarray(trainY)
    return trainX, trainY


if __name__ == '__main__':
    model = ModelGenerator()
    if len(sys.argv) == 3 and sys.argv[2] == '1':
        print("Constructing new model...")
        trainX, trainY = trainingSetGeneration()

        model.buildModel(trainX, trainY)
        model.saveModel()  # Save model for future iterations
    else:
        print("Loading stored model...")
        model.loadModel()  # Populates model with saved model at specified dir location

    # Read in and load image with gray-scale format
    filename = sys.argv[1]

    reader = ImageReader()
    reader.setFilename(filename)
    reader.processImage()
    testImage = reader.getImage().astype(int32).flatten()
    # reader.loadImageGray()
    # reader.showImage()

    predictions = model.predict(testImage)
    print(predictions)
    piece_index = np.argmax(predictions, axis=1)
    piece_name = constants.PIECES[piece_index]


