import cv2
from flask import Flask, request, jsonify
import os
import socket
import tensorflow as tf
# import ModelGenerator as mg
import workspace.Constants as Constants
# from ChessDriver import getPiecesFromImage, makePredictions
from workspace.Object_Detection.ImageDetection import detect
from workspace.Computer_Vision.ProcessTraining import borderCalculator, warpImagePerspective

import chess
import chess.engine
import asyncio

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Constants.APP_UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
OPERATION_SUCCESS = "SUCCESS"
OPERATION_FAILURE = "FAILURE"
FILENAME = 'board.jpeg'
FILENAME_WARPED = 'board_warped.jpeg'
FILENAME_TEMP = 'board_temp.jpg'
CHESS_BOARD = 64
BOARD_LEN = 8

# Load the TFLite model
model_path = 'C:/Users/Carter/Desktop/Classes/Chess-Reader/workspace/Object_Detection/model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
# ------------------------------------------------ #


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
        with open('/'.join([Constants.APP_UPLOAD_FOLDER, FILENAME]), 'wb') as out:
            out.write(bytesOfImage)
        return {'status': OPERATION_SUCCESS}
    return {'status': OPERATION_FAILURE, 'message': 'Invalid method: ' + request.method}


@app.route('/bound', methods=['GET'])
def bound():
    if request.method == 'GET':
        print("/bound reached!")
        # Confirm image upload has taken place
        if len(os.listdir(app.config['UPLOAD_FOLDER'])) == 0:
            return {}
        filepath = "/".join([app.config['UPLOAD_FOLDER'], FILENAME_TEMP]) # FILENAME

        frame = cv2.imread(filepath)
        borderPoints = borderCalculator(frame)
        print(borderPoints)
        # Warp frame and write to upload directory
        warpFrame = warpImagePerspective(frame, borderPoints)
        cv2.imwrite("/".join([Constants.APP_UPLOAD_FOLDER, FILENAME_WARPED]), warpFrame)

        return {"status": OPERATION_SUCCESS}

    return {"status": OPERATION_FAILURE}


@app.route('/stockfish', methods=['POST'])
def stockfish_setup():
    if request.method == 'POST':
        print("/stockfish reached!")
        if request.json['move'] is None:
            return {"status": OPERATION_FAILURE}

        board = chess.Board(request.json['move'])
        if request.json['time'] is None:
            time = 5
        else:
            time = float(request.json['time'])
        print(board)
        asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
        result = asyncio.run(stockfish(board, time))
        piece = board.piece_at(result.move.from_square).symbol()
        dataPackage = {"status": OPERATION_SUCCESS,
                "from": chess.square_name(result.move.from_square),
                "to": chess.square_name(result.move.to_square),
                "piece": piece}
        print(dataPackage)
        return dataPackage
    return {"status": OPERATION_FAILURE}


async def stockfish(board, time):
    transport, engine = await chess.engine.popen_uci(
        r"D:\Carter\downloads\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe")
    result = await engine.play(board, chess.engine.Limit(time=time))
    return result

# @app.route('/warp', methods=['POST'])
# def warp():
#     if request.method == 'POST':
#         print("/warp reached!")
#
#         boardBL = request.json["boardBL"]
#         boardTL = request.json["boardTL"]
#         boardTR = request.json["boardTR"]
#         boardBR = request.json["boardBR"]
#
#         if boardTR is None or boardBR is None or boardBL is None or boardTL is None:
#             return {'status': IMG_UPLOAD_FAILURE, 'message': 'Unable to gather boarder points'}
#
#         # Update image
#         if len(os.listdir(app.config['UPLOAD_FOLDER'])) == 0:
#             return {'status': IMG_UPLOAD_FAILURE, 'message': 'No image uploaded'}
#         fp = os.path.join(app.config['UPLOAD_FOLDER'], FILENAME)
#         frame = imread(fp)
#
#
#
#         return {'status': IMG_UPLOAD_SUCESS}
#     return {'status': IMG_UPLOAD_FAILURE, 'message': 'Invalid method: ' + request.method}


@app.route('/classify', methods=['GET'])
def classify():
    if request.method == 'GET':
        print("/classify reached!")

        """
        # Load CNN Model
        model = mg.loadModel('../Keras')
        # Parse Image File for Pieces
        croppedImages, centroids, coords = getPiecesFromImage(cv2.imread(filepath))
        # Predict pieces and their locations
        pieces = makePredictions(model, croppedImages, centroids, coords)
        """
        # Confirm image upload has taken place
        if len(os.listdir(app.config['UPLOAD_FOLDER'])) != 3:
            print("Image uploaded incorrectly.")
            return {'status': OPERATION_FAILURE, 'message': 'Image uploaded incorrectly.'}
        filepath = app.config['UPLOAD_FOLDER'] + '/' + FILENAME_WARPED
        print('Detecting pieces from ' + filepath)

        pieces = detect(filepath, interpreter)
        return {"status": OPERATION_SUCCESS, "pieces": pieces}
    return {"status": OPERATION_FAILURE, "pieces": {}}


if __name__ == '__main__':
    print("Running Flask Server on: " + socket.gethostbyname(socket.gethostname()))
    app.run(host=socket.gethostbyname(socket.gethostname()), port=3000, debug=True)
