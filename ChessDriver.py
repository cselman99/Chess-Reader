# NOTE: ChessDriver is for independant training of the Computer Vision CNN Model

import sys

import cv2
from sklearn.preprocessing import MinMaxScaler
import Constants as constants
from os import listdir
from os.path import isfile, join
import numpy as np
from ModelGenerator import ModelGenerator
from ProcessTraining import getPiecesFromImage


# Uses the consistent filename scheme to determine the piece.
# Ex: if filename is "rook9128.png" --> return "rook"
def extractPieceFromFilename(filename):
    for piece in constants.PIECES:
        if piece in filename:
            return piece
    return None


def loadTrainingSet():
    # Format training set
    trainX, trainY = [], []
    trainingFiles = [join(constants.TRAINING_PATH, f)
                     for f in listdir(constants.TRAINING_PATH)
                     if isfile(join(constants.TRAINING_PATH, f))]

    for file in trainingFiles:
        name = extractPieceFromFilename(file)
        if name is not None:
            img = np.asarray(cv2.imread(file)).reshape(1,)  # 80 x 80 (6400 pixels)
            trainX.append(img)
            trainY.append(name)

    trainX = np.asarray(trainX)
    trainY = np.asarray(trainY)
    return trainX, trainY


def predictSquare(curCentroid, houghlines):
    return ""


def makePredictions(model, images, centroids, houghlines):
    N = len(images)
    pieces = []
    for i in range(N):
        # Prepare Image for Prediction (6400 pixels)
        curImage = np.asarray(images[i]).reshape(1,)
        # Get Related Centroid for Piece
        curCentroid = centroids[i]
        # Predict Piece Position
        squareName = predictSquare(curCentroid, houghlines)
        # Predict Type of Piece
        prediction = model.predict(curImage)
        pieceIndex = np.argmax(prediction, axis=1)
        pieceName = constants.PIECES[pieceIndex]

        pieces.append((pieceName, squareName))
    return pieces

class IllegalArgumentError(ValueError):
    pass

if __name__ == '__main__':
    model = ModelGenerator()
    if len(sys.argv) == 3 and sys.argv[2] == '1':
        print("Constructing new model...")
        trainX, trainY = loadTrainingSet()
        model.buildModel(trainX, trainY)
        print("Saving model...")
        model.saveModel()  # Save model for future iterations
    elif len(sys.argv) == 3:
        print("Loading stored model...")
        model.loadModel()  # Populates model with saved model at specified dir location
    else:
        raise IllegalArgumentError(f"Wrong number of command line arguments. Expected 3 but found {len(sys.argv)}")

    # Read in and load image with gray-scale format
    filename = sys.argv[1]
    croppedImages, centroids, houghlines = getPiecesFromImage(cv2.imread(filename))
    pieces = makePredictions(model, croppedImages, centroids, houghlines)
    print(pieces)



