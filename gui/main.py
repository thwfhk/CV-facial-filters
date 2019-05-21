import sys
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import cv2


selectedFilters = []


class Main_Form(QDialog):
    def __init__(self):
        super(Main_Form, self).__init__()
        self.ui = uic.loadUi('untitled.ui', self)
        self.cap = cv2.VideoCapture(0)

    def filtersBoxClick(self, item):
        print("You chose" + item.text())
        self.ui.listWidget.clear()
        for i in self.ui.filtersBox.selectedItems():
            self.ui.listWidget.addItem(i.clone())


    def updatePicture(self, opencv_img):
        img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
        showImage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.ui.picture.setPixmap(QPixmap.fromImage(showImage))
    
    def resizeEvent(self, event):
        self.ui.filtersBox.setGeometry(60, self.height() - self.ui.filtersBox.height() - 16, self.ui.filtersBox.width(), self.ui.filtersBox.height())
    
    def capButtonClick(self):
        _, img = self.cap.read()
        self.updatePicture(img)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Main_Form()
    form.setWindowTitle('baka')
    form.ui.filtersBox.addItem("Hello")
    form.ui.filtersBox.addItem("World")
    tmp = QListWidgetItem()
    tmp.setIcon(QIcon('test.jpg'))
    tmp.setText('')
    form.ui.filtersBox.addItem(tmp)
    form.ui.filtersBox.itemClicked.connect(form.filtersBoxClick)
    form.ui.captureButton.clicked.connect(form.capButtonClick)

    img = cv2.imread('test.jpg')
    form.updatePicture(img)

    form.show()
sys.exit(app.exec_())
