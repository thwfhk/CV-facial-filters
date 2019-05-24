import sys
import cv2
import qdarkstyle
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np

import a_mobilenetv2 as twh
import filters

import time


from ui import Ui_Dialog


selectedFilters = {"nose": None, "eye": None, "ear": None}


def npy2qpm(opencv_img):
    img = opencv_img.copy()
    #img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    #img = opencv_img.copy()[:,:,::-1]
    showImage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
    return QPixmap.fromImage(showImage)

class FilterClass(QListWidgetItem):
    def __init__(self, name, typ, text, img_path):
        super(FilterClass, self).__init__(text)
        # print(name, typ, text, img)
        self.setIcon(QIcon(img_path))
        self.setText(text)
        self.name = name
        self.typ = typ

def fit_to_480x640(img):
    if img.shape[0] < 480:
        delta1 = int((480 - img.shape[0]) / 2)
        delta2 = 480 - img.shape[0] - delta1
        img = np.concatenate((np.zeros((delta1, img.shape[1], 3)).astype("uint8"), img, np.zeros((delta2, img.shape[1], 3)).astype("uint8")), axis=0)
    if img.shape[1] < 640:
        delta1 = int((640 - img.shape[1]) / 2)
        delta2 = 640 - img.shape[1] - delta1
        img = np.concatenate((np.zeros((img.shape[0], delta1, 3)).astype("uint8"), img, np.zeros((img.shape[0], delta2, 3)).astype("uint8")), axis=1)
    return img



class Worker(QThread):
    cap = cv2.VideoCapture(0)
    sinOut = pyqtSignal()
    data = None

    def __init__(self, typ):
        super(Worker, self).__init__()
        self.typ = typ
        self.raw_image = None
        self.keep_running = False
        self.file_name = ""
        self.qinding = False


    def __del__(self):
        self.keep_running = False
        self.wait()

    def run(self):
        if self.typ == "camera":
            while self.keep_running:
                _, self.raw_image = self.cap.read()
                self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
                self.data = twh.addFilters(self.raw_image.copy(), selectedFilters)
                self.sinOut.emit()
        elif self.typ == "photo":
            if not self.qinding:
                self.raw_image = cv2.imread(self.file_name)
                self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
                self.raw_image = self.raw_image[:,::-1,:]

                h, w = self.raw_image.shape[:2]
                if h/w >= 480/640:
                    new_h = 480
                    new_w = int(new_h * (w/h))
                else:
                    new_w = 640
                    new_h = int(new_w * (h/w))
                print((new_h, new_w))

                self.raw_image = cv2.resize(self.raw_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                self.raw_image = fit_to_480x640(self.raw_image)

            self.data = twh.addFilters(self.raw_image.copy(), selectedFilters)
            self.sinOut.emit()


class Main_Form(QDialog):
    # def __del__(self):
    #     super(Main_Form, self).__del__()
    #     self.cameraThread.keep_running = False

    def __init__(self):
        super(Main_Form, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.picture.installEventFilter(self)
        self.video = False
        self.photo = False

        self.play_green = QIcon('play-green.png')
        self.play_gray = QIcon('play-gray.png')

        self.ui.cameraButton.setIcon(self.play_gray)

        self.photoThread = Worker("photo")
        self.photoThread.sinOut.connect(self.photoCallback)

        self.cameraThread = Worker("camera")
        self.cameraThread.sinOut.connect(self.cameraCallback)
        self.ui.cameraButton.clicked.connect(self.cameraButtonClicked)

        self.ui.noseFilters.itemSelectionChanged.connect(self.noseFiltersItemSelectionChanged)
        self.ui.eyeFilters.itemSelectionChanged.connect(self.eyeFiltersItemSelectionChanged)
        self.ui.earFilters.itemSelectionChanged.connect(self.earFiltersItemSelectionChanged)

    def photoCallback(self):
        self.updatePicture(self.photoThread.data)

    def cameraCallback(self):
        self.updatePicture(self.cameraThread.data)

    def cameraButtonClicked(self):
        if self.video:
            self.ui.cameraButton.setIcon(self.play_gray)
            self.cameraThread.keep_running = False
            self.photo = True
            self.photoThread.raw_image = self.cameraThread.raw_image
            self.photoThread.qinding = True
        else:
            self.ui.cameraButton.setIcon(self.play_green)
            self.photo = False
            self.cameraThread.keep_running = True
            self.cameraThread.start()
        self.video = not self.video

    def noseFiltersItemSelectionChanged(self):
        if len(self.ui.noseFilters.selectedItems()) == 0:
            selectedFilters["nose"] = None
        else:
            selectedFilters["nose"] = self.ui.noseFilters.selectedItems()[0].name
        if self.photo:
            self.photoThread.start()

    def earFiltersItemSelectionChanged(self):
        if len(self.ui.earFilters.selectedItems()) == 0:
            selectedFilters["ear"] = None
        else:
            selectedFilters["ear"] = self.ui.earFilters.selectedItems()[0].name
        if self.photo:
            self.photoThread.start()

    def eyeFiltersItemSelectionChanged(self):
        if len(self.ui.eyeFilters.selectedItems()) == 0:
            selectedFilters["eye"] = None
        else:
            selectedFilters["eye"] = self.ui.eyeFilters.selectedItems()[0].name
        if self.photo:
            self.photoThread.start()

    def eventFilter(self, source, e):
        if source is self.ui.picture:
            if e.type() == QEvent.DragEnter:
                if self.video:
                    e.ignore()
                if e.mimeData().hasUrls():
                    if len(e.mimeData().urls()) == 1:
                        e.ignore()
                e.accept()
                return True

            if e.type() == QEvent.Drop:
                self.photoThread.qinding = False
                self.video = False
                self.ui.cameraButton.setIcon(self.play_gray)
                self.photoThread.file_name = e.mimeData().urls()[0].toLocalFile()
                print(self.photoThread.file_name)
                self.photo = True
                self.photoThread.run()
                return True

        return QDialog.eventFilter(self, source, e)

    def updatePicture(self, opencv_img):
        self.ui.picture.setPixmap(npy2qpm(opencv_img))


import matplotlib.pyplot as plt

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Main_Form()
    allFilters = filters.getAllFilters()
    for i in allFilters:
        if i.type == "nose":
            #form.ui.noseFilters.addItem(FilterClass(text="", name=i.name, img=i.image, typ=i.type))
            form.ui.noseFilters.addItem(FilterClass(text="", name=i.name, img_path=i.image_path, typ=i.type))
        elif i.type == "eye":
            #form.ui.eyeFilters.addItem(FilterClass(text="", name=i.name, img=i.image, typ=i.type))
            form.ui.eyeFilters.addItem(FilterClass(text="", name=i.name, img_path=i.image_path, typ=i.type))
        elif i.type == "ear":
            #form.ui.earFilters.addItem(FilterClass(text="", name=i.name, img=i.image, typ=i.type))
            form.ui.earFilters.addItem(FilterClass(text="", name=i.name, img_path=i.image_path, typ=i.type))


    """
    print(img)

    form.ui.noseFilters.addItem(FilterClass(text="nose", name="nose", img=img, typ="nose"))
    form.ui.noseFilters.addItem(FilterClass(text="nose1", name="nose1", img=img, typ="nose1"))
    form.ui.eyeFilters.addItem(FilterClass(text="eye", name="eye", img=img, typ="eye"))
    form.ui.earFilters.addItem(FilterClass(text="ear", name="ear", img=img, typ="ear"))
    """
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    form.show()
sys.exit(app.exec_())
