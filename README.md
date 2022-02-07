# Chess-Reader

## Build Training Set
1) Run ProcessTraining.py to get cropped pieces from specified image.
2) Sort cropped images into classified folders in Training folder
3) Run AugmentTraining.py on Training folder to expand training set. (Optional)
4) Run ChessDriver.py to build Keras Model


## Run backend
``python3 connect.py``