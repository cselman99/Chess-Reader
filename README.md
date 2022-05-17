# Chess-Reader

## Environment Setup
_Note: Project requires the use of GPUs. Not suitable for some laptops._

_Also make sure to change any path variables in **ImageDetection.py** and **connect.py**_

* Setup Anaconda Environement
* Install Pycocotools
* Install Keras + TensorFlow
* Install stockfish_15_win_x64_avx2

Useful Links:

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#set-env

To run project:

1. `python3 workspace/Object_Detection/ImageDetection.py 0`  Note: 0 = Build new model, 1 = Load existing model

Make sure the exported model.tflite file is properly located in the **workspace/Object_Detection/models** folder

2. `python3 Chess-Reader/workspace/Backend/connect.py`

Active Files:
* connect.py
* ChessboardDetection.py
* ImageDetection.py
* Constants.py

Deprecated Files:
* AugmentTraining.py
* ChessDriver.py
* ModelGenerator.py