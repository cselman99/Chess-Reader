from flask import Flask, request, jsonify
import os

import chess
from ImageReader import ImageReader
from ModelGenerator import ModelGenerator

UPLOAD_FOLDER = '../IMGs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
IMG_UPLOAD_SUCESS = CLASSIFY_IMAGE_SUCCESS = 1
IMG_UPLOAD_FAILURE = CLASSIFY_IMAGE_FAILURE = 0
FILENAME = 'board.jpeg'
CHESS_BOARD = 64
BOARD_LEN = 8

board = ChessBoard()

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
        return processImage()

def processImage():
    # Confirm image upload has taken place
    if len(os.listdir(app.config['UPLOAD_FOLDER'])) == 0:
        return CLASSIFY_IMAGE_FAILURE

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], FILENAME)

    # Image Processing Steps
    reader = ImageReader()
    reader.setFilename(filepath)
    reader.processImage()
    squares = reader.getSplitImages()

    # Load CNN Model
    model = ModelGenerator()
    model.loadModel()

    # Classify each board piece
    for i in range(CHESS_BOARD):
        prediction = model.predict(reader.getImage())
        board.setSpace(i // BOARD_LEN, i % BOARD_LEN, prediction)

@app.route('/getMoves', methods=['GET'])
def getMoves():
    if request.method == 'GET':
        return findMoves()

def findMoves():
    return board.getBestMoves()

if __name__ == '__main__':
    app.run(debug=True)
