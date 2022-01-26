import cv2
import numpy as np
from numpy import pi

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


def getPiecesFromImage(frame):
    # Apply Grayscale, GBlur, and Canny
    contour_frame = processFrame(frame, 1)
    contours = getContours(contour_frame)
    maxContour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Get Board Corners
    perim = cv2.arcLength(maxContour, True)
    epsilon = 0.02*perim
    approxCorners = np.asarray(cv2.approxPolyDP(maxContour, epsilon, True)).reshape(-1, 2)
    pts_dst = np.array([[3024, 0.0], [3024, 4032], [0, 4032], [0.0, 0.0]])
    # pts_dst = np.array([[0.0, 0.0], [3024, 0.0], [3024, 4032], [0, 4032]]) FOR NO ROTATION

    # Warps Perspective and Resizes to (640, 640)
    perspective_frame = getBoardPerspective(frame, approxCorners, pts_dst)
    perspective_frame = cv2.resize(perspective_frame, (640, 640), interpolation=cv2.INTER_AREA)
    # cv2.imwrite("./Training/WarpedImage.jpg", perspective_frame)

    contour_frame = processFrame(perspective_frame, 0)  # Uses Binary instead of Canny Edge Detection
    # cv2.imwrite("./Training/Warped&Processed.jpg", contour_frame)
    contours = getContours(contour_frame)
    # rect = cv2.boundingRect(maxContour)
    # maxX, maxY, maxW, maxH = rect
    croppedImages = []
    centroids = []

    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect

        if 6000 <= w*h <= 12800:  # Greater than the size of an empty square and less than 2 squares
            # Create cropped image
            mask = perspective_frame[y:y+h, x:x+w]
            mask = cv2.resize(mask, (80, 80), interpolation=cv2.INTER_AREA)  # Create Uniform image size for TF input
            croppedImages.append(mask)
            # Draw bounding box and centroid to image
            # cv2.rectangle(perspective_frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            # Draw center coordinate in bounding box
            centerCoord = (int(x+(w/2)), int(y+(h/2)))
            centroids.append(centerCoord)
            # cv2.circle(perspective_frame, centerCoord, 4, (255, 0, 255), thickness=2)
        # print('Writing image ' + filename)
        # cv2.imwrite(filename, perspective_frame)

    hough_frame = processFrame(perspective_frame, 1)
    horizontal = cv2.HoughLines(hough_frame, 1, pi / 180, 100, min_theta=0, max_theta=pi/4)
    vertical = cv2.HoughLines(hough_frame, 1, pi / 180, 100, min_theta=pi/4, max_theta=3*pi/4)
    coordsHorizontal = []
    coordsVerticle = []
    if horizontal is not None and vertical is not None:
        horizontal = trimLines(horizontal)
        horizontal = sorted(list(horizontal), key=lambda vx: vx[0])
        coordsHorizontal = addLines(perspective_frame, horizontal, (255, 0, 0))

        vertical = trimLines(vertical)
        vertical = sorted(list(vertical), key=lambda vx: vx[0])
        coordsVerticle = addLines(perspective_frame, vertical, (0, 0, 255))

    # cv2.imwrite('./Training/testhough.jpeg', perspective_frame)
    return croppedImages, centroids, [coordsHorizontal, coordsVerticle]


def processAndSaveImage(frame):
    croppedImages, _, _ = getPiecesFromImage(frame)

    for i, img in enumerate(croppedImages):
        cv2.imwrite(f'./Training/piece{i}.jpeg', img)

if __name__ == "__main__":
    frame = cv2.imread('./ChessImages/board2.jpeg')
    processAndSaveImage(frame, 'ChessImages/bounding_img.jpeg')
