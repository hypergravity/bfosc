# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bfosc.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1430, 566)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.toolButton = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton.setGeometry(QtCore.QRect(20, 20, 26, 22))
        self.toolButton.setObjectName("toolButton")
        self.lineEdit_wd = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_wd.setEnabled(True)
        self.lineEdit_wd.setGeometry(QtCore.QRect(60, 20, 461, 21))
        self.lineEdit_wd.setObjectName("lineEdit_wd")
        self.tableWidget_files = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget_files.setGeometry(QtCore.QRect(10, 70, 571, 451))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tableWidget_files.setFont(font)
        self.tableWidget_files.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidget_files.setAlternatingRowColors(True)
        self.tableWidget_files.setObjectName("tableWidget_files")
        self.tableWidget_files.setColumnCount(0)
        self.tableWidget_files.setRowCount(0)
        self.tableWidget_files.horizontalHeader().setCascadingSectionResizes(True)
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(580, 110, 134, 331))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_update_table = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(8)
        sizePolicy.setHeightForWidth(self.pushButton_update_table.sizePolicy().hasHeightForWidth())
        self.pushButton_update_table.setSizePolicy(sizePolicy)
        self.pushButton_update_table.setObjectName("pushButton_update_table")
        self.verticalLayout.addWidget(self.pushButton_update_table)
        self.pushButton_proc_bias = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_proc_bias.setObjectName("pushButton_proc_bias")
        self.verticalLayout.addWidget(self.pushButton_proc_bias)
        self.pushButton_proc_flat = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_proc_flat.setObjectName("pushButton_proc_flat")
        self.verticalLayout.addWidget(self.pushButton_proc_flat)
        self.pushButton_clear_aperture = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_clear_aperture.setObjectName("pushButton_clear_aperture")
        self.verticalLayout.addWidget(self.pushButton_clear_aperture)
        self.pushButton_add_aperture = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_add_aperture.setObjectName("pushButton_add_aperture")
        self.verticalLayout.addWidget(self.pushButton_add_aperture)
        self.pushButton_del_aperture = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_del_aperture.setObjectName("pushButton_del_aperture")
        self.verticalLayout.addWidget(self.pushButton_del_aperture)
        self.pushButton_save_aperture = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_save_aperture.setObjectName("pushButton_save_aperture")
        self.verticalLayout.addWidget(self.pushButton_save_aperture)
        self.pushButton_proc_all = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_proc_all.setObjectName("pushButton_proc_all")
        self.verticalLayout.addWidget(self.pushButton_proc_all)
        self.lineEdit_nap = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_nap.setEnabled(True)
        self.lineEdit_nap.setGeometry(QtCore.QRect(590, 470, 113, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_nap.setFont(font)
        self.lineEdit_nap.setObjectName("lineEdit_nap")
        self.lineEdit_fear = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_fear.setGeometry(QtCore.QRect(60, 45, 461, 21))
        self.lineEdit_fear.setObjectName("lineEdit_fear")
        self.toolButton_load_fear = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_load_fear.setGeometry(QtCore.QRect(20, 45, 26, 22))
        self.toolButton_load_fear.setObjectName("toolButton_load_fear")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1430, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "BFOSC E9+G10 Pipeline"))
        self.toolButton.setText(_translate("MainWindow", "..."))
        self.pushButton_update_table.setText(_translate("MainWindow", "update table"))
        self.pushButton_proc_bias.setText(_translate("MainWindow", "process bias"))
        self.pushButton_proc_flat.setText(_translate("MainWindow", "process flat"))
        self.pushButton_clear_aperture.setText(_translate("MainWindow", "clear apertures"))
        self.pushButton_add_aperture.setText(_translate("MainWindow", "add aperture"))
        self.pushButton_del_aperture.setText(_translate("MainWindow", "del aperture"))
        self.pushButton_save_aperture.setText(_translate("MainWindow", "save apertures"))
        self.pushButton_proc_all.setText(_translate("MainWindow", "process all"))
        self.toolButton_load_fear.setText(_translate("MainWindow", "..."))
