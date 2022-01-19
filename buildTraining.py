import math
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from numpy import pi, cos, sin

AUGMENTATION_LIMIT = 4
CANNY_LOW = 50
CANNY_HIGH = 255

def pol2cart(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def polSlope(theta):
    cosTheta = np.cos(theta)
    secSquaredTheta = (1 / (cosTheta * cosTheta))
    return np.tan(theta) + np.tan(theta) * secSquaredTheta

def trimLines(lines):
    angle_thresh = 10  # measured in degrees
    point_thresh = 40  # hough measured distance between points of two lines
    strongLines = [lines[0]]
    for line in lines:
        for sline in strongLines:
            rho1, theta1 = line[0], line[1]
            rho2, theta2 = sline[0], sline[1]
            if abs(rho1 - rho2) > point_thresh:
                continue
            deg1 = theta1 * 180 / pi
            deg2 = theta2 * 180 / pi
            if abs(deg1 - deg2) <= angle_thresh \
                or abs((deg1 - 360) - deg2) <= angle_thresh \
                    or abs((deg2 - 360) - deg1) <= angle_thresh:
                pass
    # redlines = np.delete(lines, toRemove).reshape(-1, 2)
    # return redlines


def processAndSaveImage(frame, augment, filename='./Training/board.jpeg'): # test
    # Convert to Grayscale
    mod_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    mod_frame = cv2.GaussianBlur(mod_frame, (5, 5), 0)  # Alternative: cv2.medianBlur(img,5)
    # Apply Canny Edge Detection
    mod_frame = cv2.Canny(mod_frame, CANNY_LOW, CANNY_HIGH, L2gradient=True)
    # Get thresholds from Image
    ret, thresh = cv2.threshold(mod_frame, 0, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU
    # Use thresholds to find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Add Contours to image
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Image Augmentation to increase training set size
    if augment:
        print(f'Writing image {filename} plus {AUGMENTATION_LIMIT} augmentations')
        cv2.imwrite(filename, frame)
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        frame = img_to_array(frame)
        frame = frame.reshape((1,) + frame.shape)
        j = 0
        for _ in datagen.flow(frame, batch_size=1,
                                  save_to_dir='./Training', save_prefix='board', save_format='jpeg'):
            j += 1
            if j > AUGMENTATION_LIMIT:
                break
    else:
        print('Writing image ' + filename)
        cv2.imwrite(filename, frame)

if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)
    # i = 1
    #
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     cv2.imshow('Chess Board Input', frame)
    #
    #     c = cv2.waitKey(1)
    #     if c == 27: # Escape Key terminates program
    #         break
    #     elif c == 32: # Space bar captures photo
    #         # 0 = Process single frame
    #         # 1 = Batch augmented frames
    #         processAndSaveImage(frame, 0, './Training/board' + str(i) + '.jpeg')
    #         i += 1
    #
    # cap.release()
    # cv2.destroyAllWindows()
    frame = cv2.imread('./ChessImages/board2.jpeg')
    processAndSaveImage(frame, 0, './Training/board1.jpeg')