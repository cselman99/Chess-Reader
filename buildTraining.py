import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from numpy import pi

AUGMENTATION_LIMIT = 4
CANNY_LOW = 50
CANNY_HIGH = 255

def trimLines(lines):
    strong_lines = np.array([]).reshape(-1, 2)
    for i, n1 in enumerate(lines):
        for rho, theta in n1:
            if i == 0:
                strong_lines = np.append(strong_lines, n1, axis=0)
                continue
            if rho < 0:
                rho *= -1
                theta -= pi
            closeness_rho = np.isclose(rho, strong_lines[:, 0], atol=20)
            closeness_theta = np.isclose(theta, strong_lines[:, 1], atol=pi/10)
            closeness = np.all([closeness_rho, closeness_theta], axis=0)
            if not any(closeness) and len(strong_lines) <= 8:
                strong_lines = np.append(strong_lines, n1, axis=0)
    return strong_lines


def addLines(frame, lines, color):
    coords = []
    for line in lines:
        rho, theta = line[0], line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        coords.append((x1, y1, x2, y2))
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)  # COLOR: BGR Format
    return coords


def processAndSaveImage(frame, augment, filename='./Training/board.jpeg'): # test
    # Convert to Grayscale
    mod_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    mod_frame = cv2.GaussianBlur(mod_frame, (5, 5), 0)  # Alternative: cv2.medianBlur(img,5)
    # Apply Canny Edge Detection
    mod_frame = cv2.Canny(mod_frame, CANNY_LOW, CANNY_HIGH, L2gradient=True)
    # Calculate Contour of board for Perspective Transform
    ret, thresh = cv2.threshold(mod_frame, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxContour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(frame, maxContour, -1, (0, 255, 0), 2)
    # Calculate Hough Lines for Board Division
    horizontal = cv2.HoughLines(mod_frame, 1, pi / 180, 100, min_theta=0, max_theta=pi/4)
    vertical = cv2.HoughLines(mod_frame, 1, pi / 180, 100, min_theta=pi/4, max_theta=3*pi/4)
    if horizontal is not None and vertical is not None:  # Was able to find gridlines
        horizontal = trimLines(horizontal)
        horizontal = sorted(list(horizontal), key=lambda x: x[0])
        coordsH = addLines(frame, horizontal, (255, 0, 0))

        vertical = trimLines(vertical)
        vertical = sorted(list(vertical), key=lambda x: x[0])
        coordsV = addLines(frame, vertical, (0, 0, 255))

        board_left = int(vertical[0][0])
        board_right = int(vertical[-1][0])
        board_bottom = int(horizontal[0][0])
        board_top = int(horizontal[-1][0])
        # print(board_left, board_right, board_bottom, board_top)
        # print(frame.shape)
        mask = np.zeros(frame.shape, np.uint8)
        mask[board_bottom:board_top, board_left:board_right] = frame[board_bottom:board_top, board_left:board_right]
        cv2.imwrite('./ChessImages/testCrop.jpeg', mask)



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
