import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

from hough import findlines

def findhelper(bases, imges, num, windows,length, judge):
    base = 0
    for i in range(num):
        if length+base < num:
            lines, cdst = findlines(windows[0 + base:length + base, :, :], judge)
        else:
            break;
        base += length
        lines = np.array(lines)
        # print(lines.size)
        if lines.size > 3:
            bases.append(base)
            imges.append(cdst)

def findwindows(windows, tempwindow,tempwindow2, tempwindow3,length, judge):
    x1, y1, z1 = windows.shape
    x2, y2, z2 = tempwindow.shape
    x3, y3, z3 = tempwindow2.shape
    x4, y4, z4 = tempwindow3.shape
    bases = []
    imges = []
    findhelper(bases, imges, x1, windows, length, judge)
    findhelper(bases, imges, x2, tempwindow, length, judge)
    findhelper(bases, imges, x3, tempwindow2, length, judge)
    findhelper(bases, imges, x4, tempwindow3, length, judge)
    imges = np.array(imges)
    print(imges.shape)

    return imges