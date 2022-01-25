import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from numpy import pi
from os import listdir
from os.path import isfile, join

AUGMENTATION_LIMIT = 4
CANNY_LOW = 0  # 100
CANNY_HIGH = 155 # 255

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


def augmentImage(source, destination):
    # Image Augmentation to increase training set size
    onlyfiles = [f for f in listdir(source) if isfile(join(source, f))]
    for file in onlyfiles:
        print(f'Writing image {file} plus {AUGMENTATION_LIMIT} augmentations')
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
                              save_to_dir=destination, save_prefix=file, save_format='jpeg'):
            j += 1
            if j > AUGMENTATION_LIMIT:
                break


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


def getBoardPerspective(frame, corners, tcorners):
    h, status = cv2.findHomography(corners, tcorners)
    pframe = cv2.warpPerspective(frame, h, (frame.shape[1], frame.shape[0]))
    return pframe


def processFrame(frame, tresholding):
    # Convert to Grayscale
    mod_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    mod_frame = cv2.GaussianBlur(mod_frame, (5, 5), 0)  # Alternative: cv2.medianBlur(img,5)
    if tresholding:
        # Apply Canny Edge Detection
        mod_frame = cv2.Canny(mod_frame, CANNY_LOW, CANNY_HIGH, L2gradient=True)
    else:
        mod_frame = cv2.threshold(mod_frame, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return mod_frame


# Calculate Contour of board for Perspective Transform
def getContours(contour_frame):
    ret, thresh = cv2.threshold(contour_frame, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def processAndSaveImage(frame, augment, filename='./Training/board.jpeg'):
    # Apply Grayscale, GBlur, and Canny
    contour_frame = processFrame(frame, 1)
    contours = getContours(contour_frame)
    maxContour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    maxCArea = cv2.contourArea(maxContour)

    # Get Board Corners
    perim = cv2.arcLength(maxContour, True)
    epsilon = 0.02*perim
    approxCorners = np.asarray(cv2.approxPolyDP(maxContour, epsilon, True)).reshape(-1, 2)
    pts_dst = np.array([[3024, 0.0], [3024, 4032], [0, 4032], [0.0, 0.0]])

    # Warps Perspective and Resizes to (640, 640)
    perspective_frame = getBoardPerspective(frame, approxCorners, pts_dst)
    perspective_frame = cv2.resize(perspective_frame, (640, 640), interpolation=cv2.INTER_AREA)
    cv2.imwrite("./Training/WarpedImage.jpg", perspective_frame)

    contour_frame = processFrame(perspective_frame, 0)  # Uses Binary instead of Canny Edge Detection
    cv2.imwrite("./Training/Warped&Processed.jpg", contour_frame)
    contours = getContours(contour_frame)
    # rect = cv2.boundingRect(maxContour)
    # maxX, maxY, maxW, maxH = rect
    imageCounter = 0
    croppedImages = []
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        centerCoord = (int(x+(w/2)), int(y+(h/2)))
        cArea = cv2.contourArea(cnt)

        if np.isclose(cArea, maxCArea, atol=100):  # or x < maxX or x > (maxX + maxW) or y < maxY or y > (maxY + maxH)
            print("A")
            continue
        elif w*h >= 6400 and w*h <= 12800:  # Greater than the size of an empty square and less than 2 squares
            print("B")
            # Create cropped image
            mask = perspective_frame[y:y+h, x:x+w]
            croppedImages.append(mask)
            # Draw bounding box and centroid to image
            cv2.rectangle(perspective_frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.circle(perspective_frame, centerCoord, 4, (255, 0, 255), thickness=2)
        else:
            print(w*h)

    print('Writing image ' + filename)
    cv2.imwrite(filename, perspective_frame)

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
