import math
import cv2
import numpy as np
from numpy import pi


def trimLines(lines):
    """
    :param lines: Takes in an array of lines represented in polar coordinates [rho, theta]
    :return: Filters out lines from the list that are considered to be "close", meaning lines that are slight
    variations of each other.
    """
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


def addLines(lines):
    """
    :param lines: Takes in an array of lines represented in polar coordinates [rho, theta]
    :return: Returns the euclidean equivalent of each line
    """
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
    return coords


def getBoardPerspective(frame, corners, tcorners):
    """
    :param frame: A numpy representation of an image
    :param corners: 4 coordinates of the points in the original plane
    :param tcorners: 4 coordinates of the points in the target plane
    :return:
    """
    h, status = cv2.findHomography(corners, tcorners)
    pframe = cv2.warpPerspective(frame, h, (frame.shape[1], frame.shape[0]))
    return pframe


def processFrame(frame, thresholding=0):
    """
    :param frame: A numpy representation of an image
    :param thresholding: Indicator for method of thresholding. 0 = Canny. 1 = Binary
    :return: Returns the modified frame
    """
    # Convert to Grayscale
    mod_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    mod_frame = cv2.GaussianBlur(mod_frame, (5, 5), 1)  # Alternative: cv2.medianBlur(img,5)
    if thresholding:
        # Apply Canny Edge Detection
        mod_frame = cv2.Canny(mod_frame, 40, 140, edges=1, L2gradient=True)  # Original: 30, 90
    else:
        mod_frame = cv2.threshold(mod_frame, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return mod_frame


# Calculate Contour of board for Perspective Transform
def getContours(frame):
    """
    :param frame: A numpy representation of a contour image
    :return: A list of the contours detected in the image
    """
    ret, thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def dist(var1, var2):
    """
    :param var1: Coordinate 1
    :param var2: Coordinate 2
    :return: Returns the distance between coordinates
    """
    t1 = math.pow(var1[0] - var2[0], 2)
    t2 = math.pow(var1[1] - var2[1], 2)
    return math.sqrt(t1 + t2)


def warpImagePerspective(frame, corners):
    """
    :param frame: A numpy image
    :param corners: Border coordinates of chess board
    :return: Returns a warped and resized version of the original image
    """
    pts_dst = np.array([[0.0, 0.0], [frame.shape[1], 0.0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]])  # FOR NO ROTATION
    perspective_frame = getBoardPerspective(frame, corners, pts_dst)
    perspective_frame = cv2.resize(perspective_frame, (640, 640), interpolation=cv2.INTER_AREA)
    return perspective_frame


def getHoughLines(frame):
    """
    :param frame: A numpy image
    :return: Returns the vertical and horizontal Hough Lines detected in the input image.
    """
    hough_frame = processFrame(frame, 1)
    horizontal = cv2.HoughLines(hough_frame, 1, pi / 180, 100, min_theta=0, max_theta=pi / 4)
    vertical = cv2.HoughLines(hough_frame, 1, pi / 180, 100, min_theta=pi / 4, max_theta=3 * pi / 4)
    coordsHorizontal = []
    coordsVerticle = []
    if horizontal is not None and vertical is not None:
        horizontal = trimLines(horizontal)
        horizontal = sorted(list(horizontal), key=lambda vx: vx[0])
        coordsHorizontal = addLines(horizontal)

        vertical = trimLines(vertical)
        vertical = sorted(list(vertical), key=lambda vx: vx[0])
        coordsVerticle = addLines(vertical)
    return [coordsHorizontal, coordsVerticle]


def borderCalculator(frame):
    """
    :param frame: A numpy image
    :return: Returns the approximated corners of the chessboard in the image
    """
    # ---------- Corner identification & Perspective Warping --------- #
    contour_frame = processFrame(frame, 1)
    # cv2.imwrite("contour_frame.jpeg", contour_frame)
    contours = getContours(contour_frame)
    maxContour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Get Board Corners
    perim = cv2.arcLength(maxContour, True)
    epsilon = 0.02 * perim
    approxCorners = np.asarray(cv2.approxPolyDP(maxContour, epsilon, True)).reshape(-1, 2).tolist()
    bottomLeft = [0, 0]
    bottomRight = [0, frame.shape[1]]
    topRight = [frame.shape[0], frame.shape[1]]
    topLeft = [frame.shape[0], 0]

    # GET FOUR FARTHEST CORNERS OF CONTOUR
    boardBR, boardBL, boardTR, boardTL = None, None, None, None
    print(approxCorners)
    for ac in approxCorners:
        cv2.circle(frame, ac, 11, (255, 255, 255), thickness=4)
        if boardBR is None or dist(boardBR, bottomRight) > dist(ac, bottomRight):
            boardBR = ac
        if boardBL is None or dist(boardBL, bottomLeft) > dist(ac, bottomLeft):
            boardBL = ac
        if boardTR is None or dist(boardTR, topRight) > dist(ac, topRight):
            boardTR = ac
        if boardTL is None or dist(boardTL, topLeft) > dist(ac, topLeft):
            boardTL = ac
    for corner in [boardBL, boardTL, boardTR, boardBR]:
        cv2.circle(frame, corner, 12, (255, 0, 0), thickness=2)
    return np.asarray([boardBL, boardTL, boardTR, boardBR])


if __name__ == "__main__":
    # fp = 'C:/Users/Carter/Desktop/Classes/Chess-Reader/workspace/Object_Detection/imgs/test/img/7a34d8620235048917b28bcfd3b5572b_jpg.rf.71653deb6fe88ad472dabea12353373d.jpg'
    fp = 'C:/Users/Carter/Desktop/Classes/Chess-Reader/workspace/Object_Detection/imgs/test/img/e4583d082076b2b549b3736ad1b193c9_jpg.rf.be7ed36bb2bee36cf4edad46fdd4ec75.jpg'
    frame = cv2.imread(fp)
    # croppedImages, _, _ = getPiecesFromImage(frame)
    # for i, img in enumerate(croppedImages):
    #     cv2.imwrite(f'./Training/piece{i}.jpeg', img)
