# --- DEPRECATED FILE --- #

import workspace.Constants as constants
from os import listdir
from os.path import isfile, join, isdir
from workspace.Constants import AUGMENTATION_LIMIT
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import cv2


def augmentImage(source, destination):
    # Image Augmentation to increase training set size
    onlyfiles = [join(source, f) for f in listdir(source) if isfile(join(source, f))]
    for file in onlyfiles:
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        frame = cv2.imread(file)
        frame = img_to_array(frame)
        frame = frame.reshape((1,) + frame.shape)
        j = 0
        for _ in datagen.flow(frame, batch_size=1,
                              save_to_dir=destination, save_prefix='piece_aug', save_format='jpeg'):
            j += 1
            if j > AUGMENTATION_LIMIT:
                break


def augmentImages():
    onlydirs = [join(constants.TRAINING_PATH, d) for d in listdir(constants.TRAINING_PATH) if isdir(join(constants.TRAINING_PATH, d))]
    print(onlydirs)
    for d in onlydirs:
        augmentImage(d, d)
    print("Augmentation Process Complete")


if __name__ == '__main__':
    augmentImages()
