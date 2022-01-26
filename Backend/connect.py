from flask import Flask, request, jsonify
import os
import ModelGenerator as mg
import Constants as constants
from ChessDriver import getPiecesFromImage, makePredictions
import cv2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = constants.APP_UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
IMG_UPLOAD_SUCESS = CLASSIFY_IMAGE_SUCCESS = 1
IMG_UPLOAD_FAILURE = CLASSIFY_IMAGE_FAILURE = 0
FILENAME = 'board.jpeg'
CHESS_BOARD = 64
BOARD_LEN = 8


@app.route('/', methods=['GET'])
def test():
    return jsonify({"hello": "world"})


def checkExtension(filename):
    for ext in ALLOWED_EXTENSIONS:
        if ext in filename:
            return True
    return False


@app.route('/submitImage', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        print("/submitImage reached!")
        # check if the post request has the file part
        if 'file' not in request.files:
            return IMG_UPLOAD_FAILURE
        file = request.files['file']

        if file.filename == '':
            return IMG_UPLOAD_FAILURE
        if file and checkExtension(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], FILENAME))
            return IMG_UPLOAD_SUCESS
    return IMG_UPLOAD_FAILURE


@app.route('/classify', methods=['GET'])
def classifyImage():
    if request.method == 'GET':
        print("/classify reached!")
        # Confirm image upload has taken place
        if len(os.listdir(app.config['UPLOAD_FOLDER'])) == 0:
            return CLASSIFY_IMAGE_FAILURE
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], FILENAME)
        # Load CNN Model
        model = mg.loadModel(constants.MODEL_PATH)
        # Parse Image File for Pieces
        croppedImages, centroids, coords = getPiecesFromImage(cv2.imread(filepath))
        # Predict pieces and their locations
        pieces = makePredictions(model, croppedImages, centroids, coords)
        return pieces


if __name__ == '__main__':
    app.run(debug=True)
