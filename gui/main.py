import sys
import cv2
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from ui import Ui_Dialog



selectedFilters = []


class Main_Form(QDialog):
    def __init__(self):
        super(Main_Form, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.showCameraPic)
        self.ui.filtersBox.itemClicked.connect(self.filtersBoxClick)
        self.ui.captureButton.clicked.connect(self.capButtonClick)
        self.ui.videoButton.clicked.connect(self.videoButtonClicked)
        self.ui.picture.installEventFilter(self)
        self.video = False

    def videoButtonClicked(self):
        if self.video:
            self.timer.stop()
        else:
            self.timer.start()

        self.video = not self.video

    def filtersBoxClick(self, item):
        print("You chose" + item.text())
        self.ui.listWidget.clear()
        for i in self.ui.filtersBox.selectedItems():
            self.ui.listWidget.addItem(i.clone())

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
        img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
        #img = opencv_img[:,:,::-1]
        showImage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.ui.picture.setPixmap(QPixmap.fromImage(showImage))
    
    def showCameraPic(self):
        _, img = self.cap.read()
        self.updatePicture(img)

    def capButtonClick(self):
        self.showCameraPic()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Main_Form()
    form.ui.filtersBox.addItem("Hello")
    form.ui.filtersBox.addItem("World")
    tmp = QListWidgetItem()
    tmp.setIcon(QIcon('test.jpg'))
    tmp.setText('')
    form.ui.filtersBox.addItem(tmp)

    img = cv2.imread('test.jpg')
    form.updatePicture(img)

    form.show()
sys.exit(app.exec_())
