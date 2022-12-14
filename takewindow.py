import cv2
import matplotlib.pyplot as plt
import numpy as np


def getwindows(img, width, length):
    # width and length are the size of the window
    windows = np.zeros((width, length,3))
    rows, column, color = img.shape
    i=0
    j=0
    tempwindow = []
    tempwindow2 = []
    tempwindow3 = []
    judge = 1
    judge1 = 1
    judge2 = 1
    judge3 = 1
    for j in range(0, rows, 20):
        for i in range(0, column, 20):
            if abs(i-column)< width:
                break
            window = img[j : j+width, i: i + length,:]
            if judge == 1:
                windows = window
            windows = np.r_[windows, window]
            judge +=1
        if i < column:
            window = img[j : j+width, i-20: column - 1,:]
            if judge1 == 1:
                tempwindow2 = window
            tempwindow2 = np.r_[tempwindow2, window]
            judge1 +=1
    if j < rows:
        for i in range(0, column, 20):
            if abs(i-column) < width:
                break
            window = img[j-20: rows - 1, i: i + length,:]
            if judge3 == 1:
                tempwindow = window
            tempwindow = np.r_[tempwindow, window]
            judge3 += 1
        if i < column:
            window = img[j-20: rows - 1, i-20: column - 1,:]
            if judge2 == 1:
                tempwindow3 = window
            tempwindow3 = np.r_[tempwindow3, window]
            judge2 += 1

    windows = windows[1:len(windows)]
    tempwindow = tempwindow[1:len(tempwindow)]
    tempwindow2 = tempwindow2[1:len(tempwindow2)]
    tempwindow3 = tempwindow3[1:len(tempwindow3)]
    print(windows.shape)
    print(tempwindow.shape)
    print(tempwindow2.shape)
    print(tempwindow3.shape)
    return windows, tempwindow,tempwindow2, tempwindow3
