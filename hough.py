import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def findlines(img, judge):
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    dst = cv2.Canny(img, 250, 450, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 70, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 500*(-b)), int(y0 + 500*(a)))
            pt2 = (int(x0 - 500*(-b)), int(y0 - 500*(a)))
            if judge:
                cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    return lines,cdst
