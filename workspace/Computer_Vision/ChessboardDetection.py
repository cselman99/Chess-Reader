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


# Code adopted from: https://github.com/maciejczyzewski/neural-chessboard/tree/b4a8906059fd61f1962828602cc7253a9881cd7a #
# This includes helper files:
# * geometry.py
# * LAPS.py
# * LLR.py
# * SLID.py
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
    frame = load(fp)
    detectBoard(frame)

