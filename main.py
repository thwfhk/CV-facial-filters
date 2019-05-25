import sys
# import cv2
import qdarkstyle
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# import numpy as np

import a_mobilenetv2 as twh
import filters

# import time
import filetype

from GUIutils import *


from ui import Ui_Dialog

CAMERA_ID = 1

selectedFilters = {"nose": None, "eye": None, "ear": None}

class FilterClass(QListWidgetItem):
    def __init__(self, name, typ, text, img_path):
        super(FilterClass, self).__init__(text)
        self.setIcon(QIcon(img_path))
        self.setText(text)
        self.name = name
        self.typ = typ

class Worker(QThread):
    cap = cv2.VideoCapture(CAMERA_ID)
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
                # self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2HSV)
                # self.raw_image[2] = cv2.equalizeHist(self.raw_image[2])
                # self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_HSV2RGB)
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

                self.raw_image = cv2.resize(self.raw_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                self.raw_image = fit_to_480x640(self.raw_image)

            self.data = twh.addFilters(self.raw_image.copy(), selectedFilters)
            self.sinOut.emit()


class Whiter(QThread):
    def __init__(self, wdg):
        super(Whiter, self).__init__()
        self.wdg = wdg
        self.pxm = QPixmap(640, 480)


    def __del__(self):
        self.wait()

    def run(self):
        delta = 0.1 / 255.0
        cur = 255
        for _ in range(255):
            self.pxm.fill(QColor(255, 255, 255, cur))
            self.wdg.setPixmap(self.pxm)
            cur -= 1
            time.sleep(delta)
        self.wdg.clear()


class Main_Form(QDialog):
    # def __del__(self):
    #     super(Main_Form, self).__del__()
    #     self.cameraThread.keep_running = False

    def __init__(self):
        super(Main_Form, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.white.installEventFilter(self)
        self.video = False
        self.photo = False
        self.capture = False

        self.play_green = QIcon()
        self.play_green.addPixmap(QPixmap("play-green.svg"), QIcon.Normal, QIcon.Off)
        self.play_gray = QIcon()
        self.play_gray.addPixmap(QPixmap("play-gray.svg"), QIcon.Normal, QIcon.Off)

        self.ui.cameraButton.setIcon(self.play_gray)

        self.photoThread = Worker("photo")
        self.photoThread.sinOut.connect(self.photoCallback)

        self.cameraThread = Worker("camera")
        self.cameraThread.sinOut.connect(self.cameraCallback)
        self.ui.cameraButton.clicked.connect(self.cameraButtonClicked)

        self.ui.noseFilters.itemSelectionChanged.connect(self.noseFiltersItemSelectionChanged)
        self.ui.eyeFilters.itemSelectionChanged.connect(self.eyeFiltersItemSelectionChanged)
        self.ui.earFilters.itemSelectionChanged.connect(self.earFiltersItemSelectionChanged)

        self.whiterThread = Whiter(self.ui.white)
        self.ui.captureButton.clicked.connect(self.captureButtonClicked)
        self.ui.white.setAttribute(Qt.WA_TranslucentBackground)


    def captureButtonClicked(self):
        self.whiterThread.start()
        if self.video:
            self.capture = True
        elif self.photo:
            save_image(self.photoThread.data)

    def photoCallback(self):
        self.updatePicture(self.photoThread.data)

    def cameraCallback(self):
        self.updatePicture(self.cameraThread.data)
        if self.capture:
            save_image(self.cameraThread.data)
            self.capture = False

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
        if source is self.ui.white:
            if e.type() == QEvent.DragEnter:
                if self.video:
                    e.ignore()
                else:
                    if e.mimeData().hasUrls():
                        if len(e.mimeData().urls()) != 1:
                            e.ignore()
                        else:
                            kind = filetype.guess(e.mimeData().urls()[0].toLocalFile())
                            if kind is None:
                                e.ignore()
                            elif kind.mime[:5] == 'image':
                                e.accept()
                            else:
                                e.ignore()
                    else:
                        e.ignore()
                return True

            if e.type() == QEvent.Drop:
                self.photoThread.qinding = False
                self.video = False
                self.ui.cameraButton.setIcon(self.play_gray)
                self.photoThread.file_name = e.mimeData().urls()[0].toLocalFile()
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
