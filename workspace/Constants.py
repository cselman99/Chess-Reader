from os.path import sep

TRAINING_PATH = '.' + sep + 'Training' + sep
ROBOFLOW_PATH = '.' + sep + 'roboflow_imgs_coco' + sep

PIECES = ['black-bishop',
          'black-king',
          'black-knight',
          'black-pawn',
          'black-queen',
          'black-rook',
          'pieces',
          'bishop',
          'white-bishop',
          'white-king',
          'white-knight',
          'white-pawn',
          'white-queen',
          'white-rook']

label_map = {
        1: 'white-king',
        2: 'white-queen',
        3: 'white-bishop',
        4: 'white-knight',
        5: 'white-rook',
        6: 'white-pawn',
        7: 'black-king',
        8: 'black-queen',
        9: 'black-bishop',
        10: 'black-knight',
        11: 'black-rook',
        12: "black-pawn"
    }

MODEL_PATH = '.' + sep + 'Keras'
AUGMENTATION_LIMIT = 4
APP_UPLOAD_FOLDER = './img_uploads'
