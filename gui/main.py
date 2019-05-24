import sys
import cv2
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# import twh

import time


from ui import Ui_Dialog


"""
allFilters = twh.getAllFilters()
allFilters[0].image =  numpyarray
allFilters[0].name = "string"
allFilters[0].typ = "nose" || "eye" || "ear"
"""


selectedFilters = {"nose": None, "eye": None, "ear": None}


def npy2qpm(opencv_img):
    img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    showImage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
    return QPixmap.fromImage(showImage)

class FilterClass(QListWidgetItem):
    def __init__(self, name, typ, text="", img=None):
        super(FilterClass, self).__init__(text)
        if img is not None:
            #icon =
            #icon.addPixmap()
            self.setIcon(QIcon(npy2qpm(img)))
        self.setText(text)
        self.name = name
        self.typ = typ


class Worker(QThread):
    cap = cv2.VideoCapture(0)
    sinOut = pyqtSignal()
    data = None

    def __init__(self, typ):
        super(Worker, self).__init__()
        self.typ = typ
        self.raw_image = None


    def __del__(self):
        self.wait()

    def run(self):
        if self.typ == "camera":
            while True:
                _, frame = self.cap.read()
                # self.data = twh.addFilters(frame, selectedFilters)
                self.data = frame
                self.sinOut.emit()
        elif self.typ == "photo":
            _, frame = self.cap.read()
            # self.data = twh.addFilters(frame, selectedFilters)
            self.data = frame
            self.sinOut.emit()


class Main_Form(QDialog):
    def __init__(self):
        super(Main_Form, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.picture.installEventFilter(self)
        self.video = False

        self.photoThread = Worker("photo")
        self.photoThread.sinOut.connect(self.photoCallback)
        self.ui.photoButton.clicked.connect(self.photoButtonClicked)

        self.cameraThread = Worker("camera")
        self.cameraThread.sinOut.connect(self.cameraCallback)
        self.ui.cameraButton.clicked.connect(self.cameraButtonClicked)

        self.ui.noseFilters.itemSelectionChanged.connect(self.noseFiltersItemSelectionChanged)
        self.ui.eyeFilters.itemSelectionChanged.connect(self.eyeFiltersItemSelectionChanged)
        self.ui.earFilters.itemSelectionChanged.connect(self.earFiltersItemSelectionChanged)

    def photoCallback(self):
        self.updatePicture(self.photoThread.data)

    def photoButtonClicked(self):
        self.photoThread.start()

    def cameraCallback(self):
        self.updatePicture(self.cameraThread.data)

    def cameraButtonClicked(self):
        if self.video:
            self.cameraThread.terminate()
        else:
            self.cameraThread.start()
        self.video = not self.video

    def noseFiltersItemSelectionChanged(self):
        if len(self.ui.noseFilters.selectedItems()) == 0:
            selectedFilters["nose"] = None
        else:
            selectedFilters["nose"] = self.ui.noseFilters.selectedItems()[0].name
        print(selectedFilters)

    def earFiltersItemSelectionChanged(self):
        if len(self.ui.earFilters.selectedItems()) == 0:
            selectedFilters["ear"] = None
        else:
            selectedFilters["ear"] = self.ui.earFilters.selectedItems()[0].name
        print(selectedFilters)

    def eyeFiltersItemSelectionChanged(self):
        if len(self.ui.eyeFilters.selectedItems()) == 0:
            selectedFilters["eye"] = None
        else:
            selectedFilters["eye"] = self.ui.eyeFilters.selectedItems()[0].name
        print(selectedFilters)

    def eventFilter(self, source, e):
        if source is self.ui.picture:
            if e.type() == QEvent.DragEnter:
                if e.mimeData().hasUrls():
                    if len(e.mimeData().urls()) == 1:
                        e.ignore()
                e.accept()
                return True

            if e.type() == QEvent.Drop:
                print(e.mimeData().urls()[0].toLocalFile())
                opencv_img = cv2.imread(e.mimeData().urls()[0].toLocalFile())
                opencv_img = cv2.resize(opencv_img, (640, 480), interpolation=cv2.INTER_CUBIC)
                self.updatePicture(opencv_img)
                return True

        return QDialog.eventFilter(self, source, e)

    def updatePicture(self, opencv_img):
        self.ui.picture.setPixmap(npy2qpm(opencv_img))




if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Main_Form()

    img = cv2.imread('test.jpg')
    print(img)

    form.ui.noseFilters.addItem(FilterClass(text="nose", name="nose", img=img, typ="nose"))
    form.ui.noseFilters.addItem(FilterClass(text="nose1", name="nose1", img=img, typ="nose1"))
    form.ui.eyeFilters.addItem(FilterClass(text="eye", name="eye", img=img, typ="eye"))
    form.ui.earFilters.addItem(FilterClass(text="ear", name="ear", img=img, typ="ear"))

    """
    tmp = QListWidgetItem()
    tmp.setIcon(QIcon('test.jpg'))
    tmp.setText('')
    form.ui.earFilters.addItem(tmp)
    """

    img = cv2.imread('test.jpg')
    form.updatePicture(img)

    form.show()
sys.exit(app.exec_())
