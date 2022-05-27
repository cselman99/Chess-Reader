import cv2
import numpy as np
from workspace.Computer_Vision.SLID import slid, pSLID, slid_tendency
from workspace.Computer_Vision.LAPS import LAPS
from workspace.Computer_Vision.LLR import LLR, llr_pad
import workspace.Computer_Vision.ImageModifier as im

FRAME_SIZE = 640
na = np.array
save = cv2.imwrite
load = cv2.imread

# distCache = dict()
#
# def merge_line_segments(line_i, line_j):
#     # Conditional formatting
#     if len(line_i) == 4:
#         line_i = [[line_i[0], line_i[1]], [line_i[2], line_i[3]]]
#     if len(line_j) == 4:
#         line_j = [[line_j[0], line_j[1]], [line_j[2], line_j[3]]]
#     # print(line_i, line_j)
#
#     # line distance
#     line_i_length = math.hypot(line_i[1][0] - line_i[0][0], line_i[1][1] - line_i[0][1])
#     line_j_length = math.hypot(line_j[1][0] - line_j[0][0], line_j[1][1] - line_j[0][1])
#
#     # centroids
#     Xg = line_i_length * (line_i[0][0] + line_i[1][0]) + line_j_length * (line_j[0][0] + line_j[1][0])
#     Xg /= 2 * (line_i_length + line_j_length)
#
#     Yg = line_i_length * (line_i[0][1] + line_i[1][1]) + line_j_length * (line_j[0][1] + line_j[1][1])
#     Yg /= 2 * (line_i_length + line_j_length)
#
#     # orientation
#     orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))
#     orientation_j = math.atan2((line_j[0][1] - line_j[1][1]), (line_j[0][0] - line_j[1][0]))
#
#     if (abs(orientation_i - orientation_j) <= math.pi / 2):
#         orientation_r = line_i_length * orientation_i + line_j_length * orientation_j
#         orientation_r /= line_i_length + line_j_length
#     else:
#         orientation_r = line_i_length * orientation_i + line_j_length * (
#                     orientation_j - math.pi * orientation_j / abs(orientation_j))
#         orientation_r /= line_i_length + line_j_length
#
#     # coordinate transformation
#     a_x_g = (line_i[0][1] - Yg) * math.sin(orientation_r) + (line_i[0][0] - Xg) * math.cos(orientation_r)
#     b_x_g = (line_i[1][1] - Yg) * math.sin(orientation_r) + (line_i[1][0] - Xg) * math.cos(orientation_r)
#     c_x_g = (line_j[0][1] - Yg) * math.sin(orientation_r) + (line_j[0][0] - Xg) * math.cos(orientation_r)
#     d_x_g = (line_j[1][1] - Yg) * math.sin(orientation_r) + (line_j[1][0] - Xg) * math.cos(orientation_r)
#
#     # orthogonal projections over the axis X
#     start_f = min(a_x_g, b_x_g, c_x_g, d_x_g)
#     end_f = max(a_x_g, b_x_g, c_x_g, d_x_g)
#
#     start_x = int(Xg - start_f * math.cos(orientation_r))
#     start_y = int(Yg - start_f * math.sin(orientation_r))
#     end_x = int(Xg - end_f * math.cos(orientation_r))
#     end_y = int(Yg - end_f * math.sin(orientation_r))
#
#     return [[start_x, start_y], [end_x, end_y]]
#
#
# def mergeLines(groups):
#     new_group = []
#     for group in groups:
#         merged_line = np.asarray(group[0]).reshape((2, 2))
#         for j in range(1, len(group)):
#             merged_line = merge_line_segments(merged_line, group[j])
#         new_group.append(merged_line)
#     print(len(new_group))
#     return np.asarray(new_group)
#
#
# def n1n(line, point, mag):
#     cross = np.cross(na(line[1]) - na(line[0]),
#                      na(line[0]) - na(point))
#     return np.linalg.norm(cross) / mag
#
#
# def measureCloseness(l1, l2):
#     x1, y1, x2, y2 = l1[0], l1[1], l1[2], l1[3]  # Line AB
#     x3, y3, x4, y4 = l2[0], l2[1], l2[2], l2[3]  # LINE CD
#     a = normdist((x1, y1), (x2, y2))  # Magnitude of AB
#     b = normdist((x3, y3), (x4, y4))  # Magnitude of CD
#
#     l1_rep = [l1[:2], l1[2:]]
#     l2_rep = [l2[:2], l2[2:]]
#     # Perpendicular Intersection
#     # Line AB
#     seg_x1, seg_y1 = n1n(l1_rep, na(x3, y3), a), n1n(l1_rep, na(x4, y4), a)
#     # Line CD
#     seg_x2, seg_y2 = n1n(l2_rep, na(x1, y1), b), n1n(l2_rep, na(x2, y2), b)
#
#     p = 0.4  # Hyper Parameter - Degree of Similarity
#     omega = np.pi / (2 * np.power(FRAME_SIZE * FRAME_SIZE, 0.25))
#     gamma = 0.25 * (seg_x1 + seg_y1 + seg_x2 + seg_y2) + 0.00001
#     t_delta = p * omega
#     delta = (a + b)
#
#     return a / gamma > delta and b / gamma > delta
#
#
# def normdist(a, b):
#     h = hash(str(a) + str(b))
#     if distCache.get(h) is not None:
#         return distCache[h]
#     distCache[h] = np.linalg.norm(na(a) - na(b))
#     return distCache[h]
#
#
# def groupLines(lines):
#     """
#     :param lines: Takes in an array of lines represented in polar coordinates [rho, theta]
#     :return: Filters out lines from the list that are considered to be "close", meaning lines that are slight
#     variations of each other.
#     """
#     print("Running groupLines...")
#     lines = [line[0] for line in lines]
#     grouped_lines = set()
#     groups = []
#     closeDict = dict()
#
#     for i, n1 in enumerate(lines):
#         # Confirm evaluation of new line
#         if i not in grouped_lines:
#             # Create new group with this line. Add group to seen lines
#             new_group = [n1]
#             grouped_lines.add(i)
#             # Compare to all other lines in list that are not grouped
#             for j in range(i + 1, len(lines)):
#                 n2 = lines[j]
#                 if j not in grouped_lines:
#                     # Memoized version to prevent re-calculated distance between lines
#                     tupleHash = hash((i, j)) % 100000
#                     if closeDict.get(tupleHash) is not None:
#                         close = closeDict[tupleHash]
#                     else:
#                         close = measureCloseness(n1, n2)
#                         closeDict[tupleHash] = close
#                     if close:
#                         grouped_lines.add(j)
#                         new_group.append(n2)
#             groups.append(new_group)
#
#     print("Line len before grouping: ", len(grouped_lines))
#     print("Line len after grouping: ", len(groups))
#     return groups
#
#
# def trimLines(lines):
#     """
#     :param lines: Takes in an array of lines represented in polar coordinates [rho, theta]
#     :return: Filters out lines from the list that are considered to be "close", meaning lines that are slight
#     variations of each other.
#     """
#     strong_lines = np.array([]).reshape(-1, 2)
#     for i, n1 in enumerate(lines):
#         for rho, theta in n1:
#             if i == 0:
#                 strong_lines = np.append(strong_lines, n1, axis=0)
#                 continue
#             if rho < 0:
#                 rho *= -1
#                 theta -= pi
#             closeness_rho = np.isclose(rho, strong_lines[:, 0], atol=20)
#             closeness_theta = np.isclose(theta, strong_lines[:, 1], atol=pi/10)
#             closeness = np.all([closeness_rho, closeness_theta], axis=0)
#             if not any(closeness) and len(strong_lines) <= 8:
#                 strong_lines = np.append(strong_lines, n1, axis=0)
#     return strong_lines
#
#
# def polarToCartesian(lines):
#     """
#     :param lines: Takes in an array of lines represented in polar coordinates [rho, theta]
#     :return: Returns the Cartesian equivalent of each line
#     """
#     coords = []
#     for line in lines:
#         rho, theta = line[0], line[1]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))
#         coords.append((x1, y1, x2, y2))
#     return coords
#
#
# def getBoardPerspective(frame, corners, tcorners):
#     """
#     :param frame: A numpy representation of an image
#     :param corners: 4 coordinates of the points in the original plane
#     :param tcorners: 4 coordinates of the points in the target plane
#     :return:
#     """
#     h, status = cv2.findHomography(corners, tcorners)
#     pframe = cv2.warpPerspective(frame, h, (frame.shape[1], frame.shape[0]))
#     return pframe
#
#
# def processFrame(frame, thresholding=0):
#     """
#     :param frame: A numpy representation of an image
#     :param thresholding: Indicator for method of thresholding. 0 = Canny. 1 = Binary
#     :return: Returns the modified frame
#     """
#     # Convert to Grayscale
#     mod_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Apply Gaussian Blur to reduce noise
#     mod_frame = cv2.GaussianBlur(mod_frame, (5, 5), 1)  # Alternative: cv2.medianBlur(img,5)
#     if thresholding:
#         # Apply Canny Edge Detection
#         mod_frame = cv2.Canny(mod_frame, 40, 140, edges=1, L2gradient=True)  # Original: 30, 90
#     else:
#         mod_frame = cv2.threshold(mod_frame, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     return mod_frame
#
#
# # Calculate Contour of board for Perspective Transform
# def getContours(frame):
#     """
#     :param frame: A numpy representation of a contour image
#     :return: A list of the contours detected in the image
#     """
#     ret, thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours
#
#
# def dist(var1, var2):
#     """
#     :param var1: Coordinate 1
#     :param var2: Coordinate 2
#     :return: Returns the distance between coordinates
#     """
#     return math.sqrt(np.power(var1[0] - var2[0], 2) + np.power(var1[1] - var2[1], 2))
#
#
# def warpImagePerspective(frame, corners):
#     """
#     :param frame: A numpy image
#     :param corners: Border coordinates of chess board
#     :return: Returns a warped and resized version of the original image
#     """
#     pts_dst = np.array([[0.0, 0.0], [frame.shape[1], 0.0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]])  # FOR NO ROTATION
#     perspective_frame = getBoardPerspective(frame, corners, pts_dst)
#     perspective_frame = cv2.resize(perspective_frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
#     return perspective_frame
#
#
# def getHoughLines(frame):
#     """
#     :param frame: A numpy image
#     :return: Returns the vertical and horizontal Hough Lines detected in the input image.
#     """
#     hough_frame = processFrame(frame, 1)
#     horizontal = cv2.HoughLines(hough_frame, 1, pi / 180, 100, min_theta=0, max_theta=pi / 4)
#     vertical = cv2.HoughLines(hough_frame, 1, pi / 180, 100, min_theta=pi / 4, max_theta=3 * pi / 4)
#     coordsHorizontal = []
#     coordsVertical = []
#
#     coords1 = polarToCartesian(horizontal.reshape(-1, 2))
#     coords2 = polarToCartesian(vertical.reshape(-1, 2))
#     cframe = np.copy(frame)
#     cframe2 = np.copy(frame)
#     if horizontal is not None and vertical is not None:
#         horizontal = trimLines(horizontal)
#         horizontal = sorted(list(horizontal), key=lambda vx: vx[0])
#         coordsHorizontal = polarToCartesian(horizontal)
#
#         vertical = trimLines(vertical)
#         vertical = sorted(list(vertical), key=lambda vx: vx[0])
#         coordsVertical = polarToCartesian(vertical)
#
#         for i in range(len(coords1)):
#             cs = [coords1[i][:2], coords1[i][2:]]
#             cv2.line(cframe, cs[0], cs[1], (255, 0, 0), 2)  # COLOR: BGR Format
#         for i in range(len(coords2)):
#             cs = [coords2[i][:2], coords2[i][2:]]
#             cv2.line(cframe, cs[0], cs[1], (0, 0, 255), 2)  # COLOR: BGR Format
#         for i in range(len(coordsHorizontal)):
#             cs = [coordsHorizontal[i][:2], coordsHorizontal[i][2:]]
#             cv2.line(cframe2, cs[0], cs[1], (255, 0, 0), 2)  # COLOR: BGR Format
#         for i in range(len(coordsVertical)):
#             cs = [coordsVertical[i][:2], coordsVertical[i][2:]]
#             cv2.line(cframe2, cs[0], cs[1], (0, 0, 255), 2)  # COLOR: BGR Format
#
#     save('trash_imgs/trimmedhough.jpg', cframe2)
#     save('trash_imgs/untrimmed_hough.jpg', cframe)
#     return [coordsHorizontal, coordsVertical]
#
#
# def borderCalculator(frame):
#     """
#     :param frame: A numpy image
#     :return: Returns the approximated corners of the chessboard in the image
#     """
#     # ---------- Corner identification & Perspective Warping --------- #
#     contour_frame = processFrame(frame, 1)
#     contours = getContours(contour_frame)
#     maxContour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
#
#     # Get Board Corners
#     perim = cv2.arcLength(maxContour, True)
#     epsilon = 0.02 * perim
#     approxCorners = np.asarray(cv2.approxPolyDP(maxContour, epsilon, True)).reshape(-1, 2).tolist()
#     bottomLeft = [0, 0]
#     bottomRight = [0, frame.shape[1]]
#     topRight = [frame.shape[0], frame.shape[1]]
#     topLeft = [frame.shape[0], 0]
#
#     # GET FOUR FARTHEST CORNERS OF CONTOUR
#     boardBR, boardBL, boardTR, boardTL = None, None, None, None
#     print(approxCorners)
#     for ac in approxCorners:
#         # cv2.circle(frame, ac, 11, (255, 255, 255), thickness=4)
#         if boardBR is None or dist(boardBR, bottomRight) > dist(ac, bottomRight):
#             boardBR = ac
#         if boardBL is None or dist(boardBL, bottomLeft) > dist(ac, bottomLeft):
#             boardBL = ac
#         if boardTR is None or dist(boardTR, topRight) > dist(ac, topRight):
#             boardTR = ac
#         if boardTL is None or dist(boardTL, topLeft) > dist(ac, topLeft):
#             boardTL = ac
#     # for corner in [boardBL, boardTL, boardTR, boardBR]:
#     #     cv2.circle(frame, corner, 12, (255, 0, 0), thickness=2)
#     return np.asarray([boardBL, boardTL, boardTR, boardBR])
#
#
# def drawLines(linesP, frame, color):
#     if linesP is not None:
#         for i in range(0, len(linesP)):
#             line = linesP[i][0]
#             cv2.line(frame, (line[0], line[1]), (line[2], line[3]), color, 3, cv2.LINE_AA)
#
#
# def boardExtractor(frame):
#     # Probabilistic Hough Line Transform
#     cframe = np.copy(frame)
#     gframe = np.copy(frame)
#     pf = processFrame(frame, 1)
#     linesP = cv2.HoughLinesP(pf, rho=1, theta=np.pi/360*2,
#                              threshold=40, minLineLength=50, maxLineGap=15)
#     drawLines(linesP, frame, (0, 0, 255))
#     # linesP = [line[0] for line in linesP]
#     # print(linesP)
#     save('trash_imgs/houghlinesP.jpeg', frame)
#
#     colors = [(16 * i, 16 * i, 16 * i) for i in range(20)]
#     groups = groupLines(linesP)
#     [drawLines(np.asarray(group).reshape((len(group), 1, 4)), gframe, colors[i % 20]) for i, group in enumerate(groups)]
#     save('trash_imgs/houghlinesP_grouped.jpeg', gframe)
#
#     merged = mergeLines(groups)
#     merged = merged.reshape((len(merged), 1, 4))
#     drawLines(merged, cframe, (255, 0, 0))
#     save('trash_imgs/houghlinesP_merged.jpeg', cframe)
#
#     # # Group Hough Lines based on colinearity
#     # grouped_lines =  groupLines(linesP)
#     # # Merge colinear Hough Lines
#     # merged_lines = mergeLines(grouped_lines)
#     # # Geometry detector (Verify Latice Points)
#     # result = geometryDetector(merged_lines)
#     return None


def detectBoard(frame):
    frame, _, _ = im.image_resize(frame)
    copy_frame = np.copy(frame)
    colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255)]

    for i in range(3):
        # --- 1 step --- find all possible lines (that makes sense) ----------------
        segments = pSLID(frame)
        raw_lines = slid(segments)
        lines = slid_tendency(raw_lines)

        # --- 2 step --- find interesting intersections (potentially a mesh grid) --
        points = LAPS(frame, lines)

        # --- 3 step --- last layer reproduction (for chessboard corners) ----------
        inner_points = LLR(frame, points, lines)
        four_points = llr_pad(inner_points, frame)
        print("Four_points: ", four_points)
        for p in four_points:
            cv2.circle(copy_frame, p, 10, colors[i], thickness=8)

        # --- 4 step --- preparation for next layer (deep analysis) ----------------
        try:
            frame = im.crop(frame, four_points)
        except Exception as e:
            print(e)
            frame = im.crop(frame, inner_points)
        print("\n")
    return frame


if __name__ == "__main__":
    fp = 'C:/Users/Carter/Desktop/Classes/Chess-Reader/workspace/Object_Detection/imgs/test/img/7a34d8620235048917b28bcfd3b5572b_jpg.rf.71653deb6fe88ad472dabea12353373d.jpg'
    # fp = 'C:/Users/Carter/Desktop/Classes/Chess-Reader/workspace/Object_Detection/imgs/test/img/e4583d082076b2b549b3736ad1b193c9_jpg.rf.be7ed36bb2bee36cf4edad46fdd4ec75.jpg'
    # fp = './warped_test.jpeg'
    # corners = borderCalculator(frame)
    # warpFrame = warpImagePerspective(frame, corners)
    # save('./warped_test.jpeg', warpFrame)
    # ch, cv = getHoughLines(frame)
    # boardExtractor(frame)
    frame = load(fp)
    detectBoard(frame)
    # if len(four_points) == 4:
    #     print(four_points)
    #     perspective_frame = warpImagePerspective(copy_frame, four_points)
    #     save("persp_warp_frame.jpeg", perspective_frame)
