# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(675, 839)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setAcceptDrops(False)
        self.filtersBox = QtWidgets.QListWidget(Dialog)
        self.filtersBox.setGeometry(QtCore.QRect(10, 520, 651, 91))
        self.filtersBox.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.filtersBox.setIconSize(QtCore.QSize(100, 100))
        self.filtersBox.setViewMode(QtWidgets.QListView.IconMode)
        self.filtersBox.setObjectName("filtersBox")
        self.captureButton = QtWidgets.QPushButton(Dialog)
        self.captureButton.setGeometry(QtCore.QRect(360, 630, 93, 28))
        self.captureButton.setObjectName("captureButton")
        self.picture = QtWidgets.QLabel(Dialog)
        self.picture.setGeometry(QtCore.QRect(20, 20, 640, 480))
        self.picture.setAcceptDrops(True)
        self.picture.setAutoFillBackground(True)
        self.picture.setText("")
        self.picture.setAlignment(QtCore.Qt.AlignCenter)
        self.picture.setObjectName("picture")
        self.listWidget = QtWidgets.QListWidget(Dialog)
        self.listWidget.setGeometry(QtCore.QRect(10, 630, 331, 191))
        self.listWidget.setObjectName("listWidget")
        self.videoButton = QtWidgets.QPushButton(Dialog)
        self.videoButton.setGeometry(QtCore.QRect(360, 670, 93, 28))
        self.videoButton.setObjectName("videoButton")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Pyxie"))
        self.captureButton.setText(_translate("Dialog", "capture!"))
        self.videoButton.setText(_translate("Dialog", "OpenCamera"))


