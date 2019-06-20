from PyQt5.QtGui import *
import numpy as np
import cv2
import os
import time

def npy2qpm(opencv_img):
    img = opencv_img.copy()
    #img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    #img = opencv_img.copy()[:,:,::-1]
    showImage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
    return QPixmap.fromImage(showImage)

def fit_to_480x640(img, PIC_WIDTH, PIC_HEIGHT):
    if img.shape[0] < PIC_HEIGHT:
        delta1 = int((PIC_HEIGHT - img.shape[0]) / 2)
        delta2 = PIC_HEIGHT - img.shape[0] - delta1
        img = np.concatenate((np.zeros((delta1, img.shape[1], 3)).astype("uint8"), img, np.zeros((delta2, img.shape[1], 3)).astype("uint8")), axis=0)
    if img.shape[1] < PIC_WIDTH:
        delta1 = int((PIC_WIDTH - img.shape[1]) / 2)
        delta2 = PIC_WIDTH - img.shape[1] - delta1
        img = np.concatenate((np.zeros((img.shape[0], delta1, 3)).astype("uint8"), img, np.zeros((img.shape[0], delta2, 3)).astype("uint8")), axis=1)
    return img

def save_image(data):
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    name = time.strftime("%Y%m%d%H%M%S", time.localtime())
    cur = 0
    while True:
        if not os.path.exists("./captured/{0}{1}.png".format(name, cur)):
            cv2.imwrite("./captured/{0}{1}.png".format(name, cur), data)
            break
        cur += 1

