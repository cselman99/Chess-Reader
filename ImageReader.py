import cv2

CANNY_LOW = 20
CANNY_HIGH = 200

class ImageReader:

    def __init__(self, filename=None):
        self.__filename = filename
        self.__image = None

    # Use OpenCV library to load file at specified path in BGR Format
    def loadImageBGR(self):
        self.__image = cv2.imread(self.__filename)

    # Use OpenCV library to load file at specified path in GRAY Format
    def processImage(self):
        self.__image = self.__convertToGrayScale(cv2.imread(self.__filename))
        self.__image = cv2.GaussianBlur(self.__image, (self.__image.shape[0] + 1, self.__image.shape[1] + 1), cv2.BORDER_DEFAULT)
        self.__image = cv2.Canny(self.__image, CANNY_LOW, CANNY_HIGH)

    # BGR --> GRAY
    def __convertToGrayScale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Show the image
    def showImage(self):
        cv2.imshow('image', self.__image)
        cv2.waitKey(0)

    # Get Image
    def getImage(self):
        return self.__image

    # Set Filename
    def setFilename(self, filename):
        self.__filename = filename

    # Return 64 cropped images of board spaces
    def getSplitImages(self):
        return self.__image
