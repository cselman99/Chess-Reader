import cv2
from flask import Flask, request, jsonify
import os
import socket
import tensorflow as tf
import workspace.Constants as Constants
from workspace.Object_Detection.ImageDetection import detect
from workspace.Computer_Vision.ChessboardDetection import borderCalculator, warpImagePerspective

import chess
import chess.engine
import asyncio

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
model_path = 'C:/Users/Carter/Desktop/Classes/Chess-Reader/workspace/Object_Detection/models/model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
# ------------------------------------------------ #


@app.route('/submitImage', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        print("/submitImage reached!")

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
        filepath = "/".join([app.config['UPLOAD_FOLDER'], FILENAME])

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
