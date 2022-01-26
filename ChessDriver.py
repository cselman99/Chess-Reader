# NOTE: ChessDriver is for independant training of the Computer Vision CNN Model

import sys

import cv2
from sklearn.preprocessing import MinMaxScaler
import Constants as constants
from os import listdir
from os.path import isfile, join
import numpy as np
import ModelGenerator as mg
from ProcessTraining import getPiecesFromImage


# Uses the consistent filename scheme to determine the piece.
# Ex: if filename is "rook9128.png" --> return "rook"
def extractPieceFromFilename(filename):
    for piece in constants.PIECES:
        if piece in filename:
            return piece
    return None


def isAbove(point, lines):
    pass


def isRight(point, lines):
    pass


def predictSquare(curCentroid, houghlines):
    horizontal, verticle = houghlines[0], houghlines[1]
    vSquare = 0
    hSquare = 0
    while isAbove(curCentroid, horizontal):
        hSquare += 1
    while isRight(curCentroid, verticle):
        vSquare += 1
    return hSquare, vSquare


def makePredictions(model, images, centroids, houghlines):
    N = len(images)
    pieces = []
    for i in range(N):
        # Prepare Image for Prediction (6400 pixels)
        curImage = np.asarray(images[i])
        # Get Related Centroid for Piece
        curCentroid = centroids[i]
        # Predict Piece Position
        h, v = predictSquare(curCentroid, houghlines)
        # Predict Type of Piece
        curImage = curImage / 255.
        curImage = np.reshape(curImage, (1, 80, 80, 3))
        prediction = mg.predict(model, curImage)
        pieceIndex = np.argmax(prediction, axis=1)[0]
        pieceName = constants.PIECES[pieceIndex]

        pieces.append((pieceName, (h, v)))
    return pieces

class IllegalArgumentError(ValueError):
    pass

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[2] == '1':
        print("Constructing new model...")
        train_ds, val_ds = mg.loadDataset(constants.TRAINING_PATH)
        model = mg.buildModel(train_ds, val_ds)
        print("Saving model...")
        mg.saveModel(model)  # Save model for future iterations
    elif len(sys.argv) == 3:
        print("Loading stored model...")
        model = mg.loadModel(constants.MODEL_PATH)  # Populates model with saved model at specified dir location
    else:
        raise IllegalArgumentError(f"Wrong number of command line arguments. Expected 3 but found {len(sys.argv)}")

    # Read in and load image with gray-scale format
    filename = sys.argv[1]
    croppedImages, centroids, houghlines = getPiecesFromImage(cv2.imread(filename))
    pieces = makePredictions(model, croppedImages, centroids, houghlines)
    print(pieces)



