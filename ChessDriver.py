# NOTE: ChessDriver is for independant training of the Computer Vision CNN Model

import sys

import cv2
import Constants as constants
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


def checkLocation(point, line):
    x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
    x3, y3 = point[0], point[1]
    return ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) > 0


def predictSquare(curCentroid, coords):
    horizontal, verticle = coords[0], coords[1]
    vSquare = 0
    hSquare = 0
    while checkLocation(curCentroid, horizontal[hSquare]):
        hSquare += 1
    while checkLocation(curCentroid, verticle[vSquare]):
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
    croppedImages, centroids, coords = getPiecesFromImage(cv2.imread(filename))
    pieces = makePredictions(model, croppedImages, centroids, coords)
    print(pieces)



