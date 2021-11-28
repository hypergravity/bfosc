import sys, os, glob
from PyQt5 import QtCore, QtGui, QtWidgets
from bfosc import Ui_MainWindow
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from astropy import table
from astropy.io import fits
from collections import OrderedDict

matplotlib.use('Qt5Agg')
matplotlib.rcParams["font.size"] = 5


class UiBfosc(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(UiBfosc, self).__init__(parent)

        self.setupUi(self)
        self.add_canvas()
        self.initUi()
        self.add_canvas()

        self._wd = ""

        self.assumption()
        self.datatable = None
        self.pos = []
        self.pos_temp = []

    def add_canvas(self):
        self.widget2 = QtWidgets.QWidget(self.centralwidget)
        self.widget2.setGeometry(QtCore.QRect(560, 70, 400, 450))
        self.widget2.setObjectName("widget")
        self.verticalLayout2 = QtWidgets.QVBoxLayout(self.widget2)
        self.verticalLayout2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout2.setObjectName("verticalLayout")

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        # self.canvas.setGeometry(QtCore.QRect(350, 110, 371, 311))
        # self.canvas.setObjectName("canvas")

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)
        # self.toolbar.setGeometry(QtCore.QRect(370, 70, 371, 41))

        # Just some button connected to `plot` method
        # self.pushButton_showimage.clicked.connect(self.plot)

        # set the layout
        # layout = QtWidgets.QVBoxLayout()
        self.verticalLayout2.addWidget(self.toolbar)
        self.verticalLayout2.addWidget(self.canvas)
        # layout.addWidget(self.button)
        # self.setLayout(layout)

    def assumption(self):
        test_dir = "/Users/cham/projects/bfosc/20200915_bfosc"
        self._wd = test_dir
        self.lineEdit_wd.setText(test_dir)

    def initUi(self):
        self.toolButton.clicked.connect(self._select_wd)
        self.lineEdit_wd.textChanged.connect(self._get_file_list)
        self.tableWidget_files.itemSelectionChanged.connect(self._show_img)
        self.pushButton_update_table.clicked.connect(self._update_datatable)
        # self.listWidget_files.currentItemChanged.connect(self._show_img)

    def _select_wd(self):
        print("clicked")
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.lineEdit_wd.setText(directory)

    def _set_wd(self):
        self._wd = self.lineEdit_wd.text()

    def _make_datatable(self):
        # get file list
        fps_full = glob.glob(self.lineEdit_wd.text() + "/*.fit")
        fps_full.sort()
        self.fps_full = fps_full
        fps = [os.path.basename(_) for _ in fps_full]
        self.nfp = len(fps)

        imgtype = np.asarray([fits.getheader(fp)["IMAGETYP"] for fp in fps_full])
        exptime = np.asarray([fits.getheader(fp)["EXPTIME"] for fp in fps_full])
        types = np.zeros_like(imgtype)
        self.type_dict = OrderedDict(drop=0, bias=1, flat=2, fear=3, star=4)
        self.type_list = list(self.type_dict.keys())
        self.color_list = [[255, 255, 255],
                           [211, 211, 211],
                           [255, 182, 193],
                           [255, 228, 181],
                           [173, 216, 230], ]
        # initial guess for types:
        for i in range(self.nfp):
            if "bias" in imgtype[i].lower():
                types[i] = "bias"
            elif "flat" in imgtype[i].lower():
                types[i] = "flat"
            elif "light" in imgtype[i].lower():
                if exptime[i] == 300:
                    types[i] = "fear"
                else:
                    types[i] = "star"

        self.datatable = table.Table(
            data=[fps, imgtype, exptime, types],
            names=["filename", "imagetype", "exptime", "type"])

    def _update_datatable(self):
        self.datatable["type"] = [self.type_list[self.tableWidget_files.cellWidget(irow, 3).currentIndex()] for irow in range(self.nfp)]
        self._refresh_datatable()

    def _get_file_list(self):
        self._make_datatable()
        self._refresh_datatable()

    def _refresh_datatable(self):
        # change to Table Widget
        self.tableWidget_files.clear()
        self.tableWidget_files.verticalHeader().setVisible(False)
        self.tableWidget_files.setRowCount(self.nfp)
        self.tableWidget_files.setColumnCount(4)
        self.tableWidget_files.setHorizontalHeaderLabels(self.datatable.colnames)
        for irow in range(self.nfp):
            self.tableWidget_files.setItem(irow, 0, QtWidgets.QTableWidgetItem(str(self.datatable["filename"][irow])))
            self.tableWidget_files.setItem(irow, 1, QtWidgets.QTableWidgetItem(str(self.datatable["imagetype"][irow])))
            self.tableWidget_files.setItem(irow, 2, QtWidgets.QTableWidgetItem("{:.0f}".format(self.datatable["exptime"][irow])))

            comboBoxItem = QtWidgets.QComboBox()
            comboBoxItem.addItems(self.type_dict.keys())
            # print(self.type_dict[self.datatable["type"][irow]])
            this_type_index = self.type_dict[self.datatable["type"][irow]]
            comboBoxItem.setCurrentIndex(this_type_index)
            self.tableWidget_files.setCellWidget(irow, 3, comboBoxItem)

            for icol in range(3):
                self.tableWidget_files.item(irow, icol).setBackground(
                    QtGui.QBrush(QtGui.QColor(*self.color_list[this_type_index])))

        self.tableWidget_files.resizeColumnsToContents()
        self.tableWidget_files.resizeRowsToContents()

    def _show_img(self):
        # print("-------------------------------")
        # print(self.tableWidget_files.currentRow(),self.tableWidget_files.currentColumn(),)
        ind_elected = self.tableWidget_files.currentRow()
        fp_selected = self.fps_full[ind_elected]
        print("Show file {}: {}".format(ind_elected, fp_selected))
        # try to draw it
        try:
            img = fits.getdata(fp_selected)
        except IsADirectoryError:
            print("Not sure about what you are doing ...")
            return
        # draw
        self.figure.clear()
        ax = self.figure.add_axes([0, 0, 1, 1])
        ax.imshow(img, cmap=plt.cm.jet, origin="lower", vmin=np.percentile(img, 5), vmax=np.percentile(img, 95))
        ax.set_xticks([])
        ax.set_yticks([])
        # refresh canvas
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.draw()

    def onclick(self, event):
        self.pos_temp = event.xdata, event.ydata
        # and trace this aperture
        print(self.pos_temp)

# capture cursor position ===============

# ref:
# https://matplotlib.org/stable/users/event_handling.html
#
# import matplotlib.pylab as plt
# import numpy as np
#
# f,a = plt.subplots()
# x = np.linspace(1,10,100)
# y = np.sin(x)
# a.plot(x,y)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # mainWindow = QtWidgets.QMainWindow()
    bfosc = UiBfosc()
    # ui.setupUi(mainWindow)
    # ui.initUi(mainWindow)
    bfosc.show()
    sys.exit(app.exec_())