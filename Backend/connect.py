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
FILENAME = 'boardTest.jpeg'
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
        # if 'file' not in request.files:
        #     return {'status': IMG_UPLOAD_FAILURE, 'message': 'File not found in request'}
        # file = request.files['file']

        bytesOfImage = request.get_data()
        with open('/'.join([constants.APP_UPLOAD_FOLDER, FILENAME]), 'wb') as out:
            out.write(bytesOfImage)
        return {'status': IMG_UPLOAD_SUCESS}
    return {'status': IMG_UPLOAD_FAILURE, 'message': 'Invalid method: ' + request.method}


@app.route('/classify', methods=['GET'])
def classifyImage():
    if request.method == 'GET':
        print("/classify reached!")
        # Confirm image upload has taken place
        if len(os.listdir(app.config['UPLOAD_FOLDER'])) == 0:
            return {}
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], FILENAME)
        # Load CNN Model
        model = mg.loadModel('../Keras')
        # Parse Image File for Pieces
        croppedImages, centroids, coords = getPiecesFromImage(cv2.imread(filepath))
        # Predict pieces and their locations
        pieces = makePredictions(model, croppedImages, centroids, coords)
        # pieces = [
        #         [
        #             "r",
        #             [0, 0]
        #         ],
        #         [
        #             "k",
        #             [0, 2]
        #         ],
        #         [
        #             "r",
        #             [0, 7]
        #         ],
        #         [
        #             "wp",
        #             [1, 0]
        #         ],
        #         [
        #             "n",
        #             [1, 3]
        #         ],
        #         [
        #             "b",
        #             [1, 4]
        #         ],
        #         [
        #             "wp",
        #             [1, 6]
        #         ],
        #         [
        #             "b",
        #             [2, 2]
        #         ],
        #         [
        #             "wp",
        #             [2, 7]
        #         ],
        #         [
        #             "wp",
        #             [3, 1]
        #         ],
        #         [
        #             "n",
        #             [3, 3]
        #         ],
        #         [
        #             "wp",
        #             [3, 5]
        #         ],
        #         [
        #             "R",
        #             [7, 0]
        #         ],
        #         [
        #             "K",
        #             [7, 6]
        #         ],
        #         [
        #             "P",
        #             [6, 1]
        #         ],
        #         [
        #             "P",
        #             [6, 5]
        #         ],
        #         [
        #             "P",
        #             [6, 6]
        #         ],
        #         [
        #             "P",
        #             [6, 7]
        #         ],
        #         [
        #             "Q",
        #             [5, 3]
        #         ],
        #         [
        #             "N",
        #             [5, 5]
        #         ],
        #         [
        #             "B",
        #             [5, 6]
        #         ],
        #         [
        #             "P",
        #             [4, 2]
        #         ],
        #         [
        #             "P",
        #             [4, 3]
        #         ],
        #     ]
        return {"pieces": pieces}
    return {"pieces": {}}


if __name__ == '__main__':
    app.run(host='192.168.1.224', port=3000, debug=True)
