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
        Dialog.resize(1153, 524)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setAcceptDrops(False)
        self.eyeFilters = QtWidgets.QListWidget(Dialog)
        self.eyeFilters.setGeometry(QtCore.QRect(670, 170, 461, 100))
        self.eyeFilters.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.eyeFilters.setIconSize(QtCore.QSize(90, 90))
        self.eyeFilters.setGridSize(QtCore.QSize(90, 90))
        self.eyeFilters.setViewMode(QtWidgets.QListView.IconMode)
        self.eyeFilters.setObjectName("eyeFilters")
        self.photoButton = QtWidgets.QPushButton(Dialog)
        self.photoButton.setGeometry(QtCore.QRect(680, 450, 93, 28))
        self.photoButton.setObjectName("photoButton")
        self.picture = QtWidgets.QLabel(Dialog)
        self.picture.setGeometry(QtCore.QRect(20, 20, 640, 480))
        self.picture.setAcceptDrops(True)
        self.picture.setAutoFillBackground(True)
        self.picture.setText("")
        self.picture.setAlignment(QtCore.Qt.AlignCenter)
        self.picture.setObjectName("picture")
        self.cameraButton = QtWidgets.QPushButton(Dialog)
        self.cameraButton.setGeometry(QtCore.QRect(790, 450, 93, 28))
        self.cameraButton.setObjectName("cameraButton")
        self.noseFilters = QtWidgets.QListWidget(Dialog)
        self.noseFilters.setGeometry(QtCore.QRect(670, 310, 461, 100))
        self.noseFilters.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.noseFilters.setIconSize(QtCore.QSize(90, 90))
        self.noseFilters.setGridSize(QtCore.QSize(90, 90))
        self.noseFilters.setViewMode(QtWidgets.QListView.IconMode)
        self.noseFilters.setObjectName("noseFilters")
        self.earFilters = QtWidgets.QListWidget(Dialog)
        self.earFilters.setGeometry(QtCore.QRect(670, 30, 461, 100))
        self.earFilters.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.earFilters.setIconSize(QtCore.QSize(90, 90))
        self.earFilters.setGridSize(QtCore.QSize(90, 90))
        self.earFilters.setViewMode(QtWidgets.QListView.IconMode)
        self.earFilters.setObjectName("earFilters")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(670, 10, 101, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(670, 150, 101, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(670, 290, 101, 16))
        self.label_3.setObjectName("label_3")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Pyxie"))
        self.photoButton.setText(_translate("Dialog", "capture!"))
        self.cameraButton.setText(_translate("Dialog", "OpenCamera"))
        self.label.setText(_translate("Dialog", "Ear Filters:"))
        self.label_2.setText(_translate("Dialog", "Eye Filters:"))
        self.label_3.setText(_translate("Dialog", "Nose Filters:"))


