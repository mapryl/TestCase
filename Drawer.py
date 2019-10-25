from PyQt5.QtWidgets import (QWidget, QFileDialog, QApplication, QLayout, QLabel, QHBoxLayout,
                             QVBoxLayout, QTableWidget, QTableWidgetItem, QAbstractItemView, QPushButton, QFileDialog,
                             QLineEdit, )
from PyQt5.QtCore import *
import pyqtgraph as pg
import numpy as np
import tables
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui
import datetime
import math


def initTableWidget(tableWidget):
    for i in range(0, tableWidget.rowCount()):
        tableWidget.setRowHeight(i, 10)
    for j in range(0, tableWidget.columnCount()):
        tableWidget.setColumnWidth(j, 10)
    tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)


class BarPlotWidget(pg.PlotWidget):
    def __init__(self, parent, size_p):
        super(self.__class__, self).__init__()
        self.x = None
        self.bg = None
        self.init_plots(size_p)

    def init_plots(self, size_p):
        self.x = np.arange(size_p)
        self.bg = pg.BarGraphItem(x=self.x, y0=0, y1=0, width=0.6)
        self.addItem(self.bg)

    def set_data(self, low, height):
        self.bg.setOpts(y0=low, y1=height, brush='b')


'''
class StackedWidget(QStackedWidget):
    def __init__(self):
        super(self.__class__, self).__init__(None)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)'''


class HistogramWidget(pg.PlotWidget):
    def __init__(self):
        super(self.__class__, self).__init__()

    def draw(self, data, step=10):
        y, x = np.histogram(data, bins=step)
        self.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))


class HistogramDrawer(QWidget):
    def __init__(self, M):
        super(self.__class__, self).__init__(None)

        layout = QHBoxLayout()
        self.setLayout(layout)

        leftLayout = QVBoxLayout()
        layout.addLayout(leftLayout)
        self.L1All = HistogramWidget()
        leftLayout.addWidget(self.L1All)
        self.L2All = HistogramWidget()
        leftLayout.addWidget(self.L2All)

        middleLayout = QVBoxLayout()
        layout.addLayout(middleLayout)
        self.middleHists = []
        for i in range(0, M):
            self.middleHists.append(HistogramWidget())
            middleLayout.addWidget(self.middleHists[i])

        rightLayout = QVBoxLayout()
        layout.addLayout(rightLayout)
        self.rightHists = []
        for i in range(0, M):
            self.rightHists.append(HistogramWidget())
            rightLayout.addWidget(self.rightHists[i])
        self.setWindowTitle('Histograms')
        self.setFixedSize(1500, 950)

    def draw(self, L1All, L2All, L1Row, L2Row):
        self.L1All.draw(L1All)
        self.L2All.draw(L2All)

        for i in range(0, len(L1Row)):
            self.middleHists[i].draw(L1Row[i])
            self.rightHists[i].draw(L2Row[i])

    def saveToImage(self, name):
        pixmap = QPixmap(self.size())
        self.render(pixmap)
        pixmap.save(name)


class MainWindow(QWidget):
    def __init__(self):
        super(self.__class__, self).__init__(None)

        self.M = 10
        self.N = 13
        self.P = 60

        self.fileName = None
        self.fileNameTick = None
        self.graphNum = 0
        self.initGui()

        self.startIndex = 0
        self.timeArr = None
        self.timeArrTick = None

        # self.resize(1910, 920)

    def initGui(self):
        self.setWindowTitle('TestCase')

        mainLayout = QVBoxLayout()
        self.setLayout(mainLayout)

        firstBarLayout = QHBoxLayout()
        mainLayout.addLayout(firstBarLayout)

        self.barPlotWidget = BarPlotWidget(self, self.P)
        firstBarLayout.addWidget(self.barPlotWidget)

        self.tableWidget = QTableWidget(self.M, self.N)
        firstBarLayout.addWidget(self.tableWidget)
        initTableWidget(self.tableWidget)

        self.pixelWidget = QLabel()
        canvas = QtGui.QPixmap(300, 300)
        self.pixelWidget.setPixmap(canvas)
        firstBarLayout.addWidget(self.pixelWidget)

        secondBarLayout = QHBoxLayout()
        mainLayout.addLayout(secondBarLayout)

        self.barPlotWidgetTick = BarPlotWidget(self, self.P)
        secondBarLayout.addWidget(self.barPlotWidgetTick)

        self.tableWidgetTick = QTableWidget(self.M, self.N)
        secondBarLayout.addWidget(self.tableWidgetTick)
        initTableWidget(self.tableWidgetTick)

        self.pixelWidgetTick = QLabel()
        self.pixelWidgetTick.setPixmap(canvas)
        secondBarLayout.addWidget(self.pixelWidgetTick)

        thirdBarLayout = QHBoxLayout()
        mainLayout.addLayout(thirdBarLayout)

        self.barPlotWidgetDif = BarPlotWidget(self, self.P)
        thirdBarLayout.addWidget(self.barPlotWidgetDif)
        self.tableWidgetDif = QTableWidget(self.M, self.N)
        thirdBarLayout.addWidget(self.tableWidgetDif)
        initTableWidget(self.tableWidgetDif)

        self.pixelWidgetDif = QLabel()
        self.pixelWidgetDif.setPixmap(canvas)
        thirdBarLayout.addWidget(self.pixelWidgetDif)

        secondLayout = QHBoxLayout()
        mainLayout.addLayout(secondLayout)

        loadButton = QPushButton('Выбрать файл')
        loadButton.clicked.connect(self.open_file)
        secondLayout.addWidget(loadButton)

        thirdLayout = QHBoxLayout()
        mainLayout.addLayout(thirdLayout)

        maxTime = 2359

        self.startTimeEdit = QLineEdit('0000')
        thirdLayout.addWidget(QLabel('Начало промежутка времени (формат ЧЧСС):'))
        self.startTimeEdit.setValidator(QIntValidator(0, maxTime))
        thirdLayout.addWidget(self.startTimeEdit)

        self.endTimeEdit = QLineEdit('0001')
        thirdLayout.addWidget(QLabel('Конец промежутка времени (формат ЧЧСС):'))
        self.endTimeEdit.setValidator(QIntValidator(0, maxTime))
        thirdLayout.addWidget(self.endTimeEdit)

        run_button = QPushButton('Применить интервал datetime')
        mainLayout.addWidget(run_button)
        run_button.clicked.connect(self.apply_time_interval)

        fourth_layout = QHBoxLayout()
        mainLayout.addLayout(fourth_layout)
        left_button = QPushButton('Назад')
        right_button = QPushButton('Вперед')
        left_button.clicked.connect(self.left)
        right_button.clicked.connect(self.right)
        fourth_layout.addWidget(left_button)
        fourth_layout.addWidget(right_button)

    def reset_interval(self):
        self.startIndex = 0
        self.graphNum = 0

    def open_file(self):
        self.fileName = QFileDialog.getOpenFileName(None, "Open data file", QDir(".").canonicalPath(), "HDF5(*.h5)")
        self.fileNameTick = QFileDialog.getOpenFileName(None, "Open data file", QDir(".").canonicalPath(), "HDF5(*.h5)")

        self.run()

    def left(self):
        if self.graphNum == 0:
            return
        self.graphNum -= 1
        self.startIndex -= 1
        start_time = self.timeArr[self.startIndex]
        self.startTimeEdit.setText(str(start_time))
        self.run()

    def right(self):
        start_time = int(self.startTimeEdit.text())
        end_time = int(self.endTimeEdit.text())
        if end_time <= start_time or self.timeArr is None:
            return
        self.graphNum += 1
        self.startIndex += 1
        start_time = self.timeArr[self.startIndex]
        self.startTimeEdit.setText(str(start_time))
        self.run()

    def apply_time_interval(self):
        self.reset_interval()
        self.run()

    def synchronize(self, fileRead, fileReadTick):
        timeArr = self.timeArr
        timeArrTick = self.timeArrTick
        data = np.array(fileRead.root['data'])
        dataTick = np.array(fileReadTick.root['data'])
        # close file in read mode
        fileRead.close()
        fileReadTick.close()

        for i in range(0, max(len(timeArr), len(timeArr))):
            if timeArrTick[i] != timeArr[i]:
                if timeArrTick[i] not in timeArr and timeArr[i] not in timeArrTick:
                    if timeArrTick[i] < timeArr[i]:
                        timeArr.insert(i, timeArrTick[i])
                        timeArrTick.insert(i + 1, timeArr[i + 1])
                        data = np.insert(data, i, dataTick[i], axis=0)
                        dataTick = np.insert(dataTick, i + 1, dataTick[i + 1], axis=0)
                    else:
                        timeArrTick.insert(i, timeArr[i])
                        timeArr.insert(i + 1, timeArrTick[i + 1])
                        dataTick = np.insert(dataTick, i, dataTick[i], axis=0)
                        data = np.insert(data, i + 1, dataTick[i + 1], axis=0)
                    print(timeArr[i], "not in tick file and ", timeArrTick[i], "not in bar file")
                elif timeArrTick[i] not in timeArr:
                    timeArr.insert(i, timeArrTick[i])
                    data = np.insert(data, i, dataTick[i], axis=0)
                    print(timeArrTick[i], "not in bar file")
                elif timeArr[i] not in timeArrTick:
                    timeArrTick.insert(i, timeArr[i])
                    dataTick = np.insert(dataTick, i, data[i], axis=0)
                    print(timeArr[i], " not in tick file")
        if len(timeArr) > len(timeArrTick):
            for i in range(len(timeArrTick), len(timeArr)):
                timeArrTick.append(timeArr[i])
                dataTick = np.append(dataTick, data[i], axis=0)
                print(timeArr[i], "not in tick file")
        elif len(timeArrTick) > len(timeArr):
            for i in range(len(timeArr), len(timeArrTick)):
                timeArr.append(timeArrTick[i])
                data = np.append(data, dataTick[i], axis=0)
                print(timeArrTick[i], "not in bar file")
        # open file in write mode
        fileRead = tables.open_file(self.fileName[0], mode='w')
        datetimeAtom = tables.Int64Atom()
        datetimeArray = fileRead.create_earray(fileRead.root, 'datetime', datetimeAtom, (0,))
        dataAtom = tables.Float64Atom()
        dataArray = fileRead.create_earray(fileRead.root, 'data', dataAtom, (0, self.M * self.N + self.P * 2))
        datetimeArray.append(timeArr)
        dataArray.append(data)
        fileRead.close()

        fileReadTick = tables.open_file(self.fileNameTick[0], mode='w')
        datetimeAtom = tables.Int64Atom()
        datetimeArrayTick = fileReadTick.create_earray(fileReadTick.root, 'datetime', datetimeAtom, (0,))
        dataAtom = tables.Float64Atom()
        dataArrayTick = fileReadTick.create_earray(fileReadTick.root, 'data', dataAtom,
                                                   (0, self.M * self.N + self.P * 2))
        datetimeArrayTick.append(timeArrTick)
        dataArrayTick.append(dataTick)
        # close file in write mode
        fileReadTick.close()
        print("Synchronization done")

    def run(self):
        start_time = int(self.startTimeEdit.text())
        end_time = int(self.endTimeEdit.text())

        if self.fileName is None or self.fileNameTick is None or not self.fileName[0] or not self.fileNameTick[0] \
                or end_time < start_time:
            return

        fileRead = tables.open_file(self.fileName[0], mode='r')
        fileReadTick = tables.open_file(self.fileNameTick[0], mode='r')
        self.timeArr = list(fileRead.root['datetime'])
        self.timeArrTick = list(fileReadTick.root['datetime'])

        if start_time not in self.timeArr or end_time not in self.timeArr:
            self.startTimeEdit.setText(str(self.timeArr[0]))
            self.endTimeEdit.setText(str(self.timeArr[-1]))
            if self.timeArr != self.timeArrTick:
                self.synchronize(fileRead, fileReadTick)
            self.calculate(fileRead, fileReadTick)
            fileRead.close()
            fileReadTick.close()
            return

        self.startIndex = self.timeArr.index(start_time)
        lowData = fileRead.root['data'][self.startIndex][:self.P]
        high_data = fileRead.root['data'][self.startIndex][self.P:self.P * 2]
        table_data = np.array(fileRead.root['data'][self.startIndex][self.P * 2:])
        lowDataTick = fileReadTick.root['data'][self.startIndex][:self.P]
        highDataTick = fileReadTick.root['data'][self.startIndex][self.P:self.P * 2]
        tableDataTick = np.array(fileReadTick.root['data'][self.startIndex][self.P * 2:])
        tableDataDif = abs(table_data - tableDataTick)

        fileRead.close()
        fileReadTick.close()

        self.barPlotWidget.set_data(lowData, high_data)
        self.barPlotWidgetTick.set_data(lowDataTick, highDataTick)
        self.setTableData(table_data, self.tableWidget)
        self.setTableData(tableDataTick, self.tableWidgetTick)
        self.setTableData(tableDataDif, self.tableWidgetDif)
        self.setPixelWidget(table_data, self.pixelWidget)
        self.setPixelWidget(tableDataTick, self.pixelWidgetTick)
        self.setPixelWidget(tableDataDif, self.pixelWidgetDif)

    def calculate(self, fileRead, fileReadTick):
        L1All = []
        L2All = []
        L1Row = []
        L2Row = []
        for i in range(0, self.M):
            L1Row.append([])
            L2Row.append([])
        for i in range(0, len(self.timeArr)):
            lowData = fileRead.root['data'][i][:self.P]
            highData = fileRead.root['data'][i][self.P:self.P * 2]
            tableData = np.array(fileRead.root['data'][i][self.P * 2:])
            lowDataTick = fileReadTick.root['data'][i][:self.P]
            highDataTick = fileReadTick.root['data'][i][self.P:self.P * 2]
            tableDataTick = np.array(fileReadTick.root['data'][i][self.P * 2:])
            tableDataDif = abs(tableData - tableDataTick)
            L1Allts, L2Allts, L1Rowts, L2Rowts = self.metrics(tableDataDif)
            L1All.append(L1Allts)
            L2All.append(L2Allts)
            for j in range(0, self.M):
                L1Row[j].append(L1Rowts[j])
                L2Row[j].append(L2Rowts[j])
            if i % 60 == 0:
                self.highLowDif(self.timeArr[i], lowData, lowDataTick, highData, highDataTick)
        L1AllMean = np.mean(L1All)
        L2AllMean = np.mean(L2All)
        L1RowMean = np.mean(L1Row, axis=1)
        L2RowMean = np.mean(L2Row, axis=1)
        L1AllStd = np.std(L1All)
        L2AllStd = np.std(L2All)
        L1RowStd = np.std(L1Row, axis=1)
        L2RowStd = np.std(L2Row, axis=1)
        L1AllPercentile = np.percentile(L1All, (5, 50, 95))
        L2AllPercentile = np.percentile(L2All, (5, 50, 95))
        # rewrite this later
        L1RowPercentile = np.percentile(L1Row, (5, 50, 95), axis=1).transpose()
        L2RowPercentile = np.percentile(L2Row, (5, 50, 95), axis=1).transpose()
        # slice data from file name
        fileName = self.fileName[0][-11:-3] + "Metrics.txt"
        with open(fileName, "w") as file:
            file.write("L1 mean = " + str(L1AllMean) + "\n")
            file.write("L2 mean = " + str(L2AllMean) + "\n")
            file.write("L1 std = " + str(L1AllStd) + "\n")
            file.write("L2 std = " + str(L2AllStd) + "\n")
            file.write("L1 5/50/95 percentile = " + str(L1AllPercentile) + "\n")
            file.write("L2 5/50/95 percentile = " + str(L2AllPercentile) + "\n")

            for i in range(0, self.M):
                file.write("L1 mean for " + str(i) + " row = " + str(L1RowMean[i]) + "\n")
                file.write("L2 mean for " + str(i) + " row = " + str(L2RowMean[i]) + "\n")
                file.write("L1 std for " + str(i) + " row = " + str(L1RowStd[i]) + "\n")
                file.write("L2 std for " + str(i) + " row = " + str(L2RowStd[i]) + "\n")
                file.write("L1 5/50/95 percentile for " + str(i) + " row = " + str(L1RowPercentile[i]) + "\n")
                file.write("L2 5/50/95 percentile for " + str(i) + " row = " + str(L2RowPercentile[i]) + "\n")

        self.histogramDrawer = HistogramDrawer(self.M)

        self.histogramDrawer.show()
        self.histogramDrawer.draw(L1All, L2All, L1Row, L2Row)
        self.histogramDrawer.saveToImage(str(self.fileName[0][-11:-3]) + ".png")

    def metrics(self, tableDataDif):
        L1All = sum(tableDataDif) / len(tableDataDif)
        L2ALL = sum(tableDataDif ** 2) / len(tableDataDif)
        L1Row = []
        L2Row = []
        for i in range(0, self.M):
            L1Row.append(sum(tableDataDif[i * self.N:(i + 1) * self.N]) / self.N)
            L2Row.append(sum(tableDataDif[i * self.N:(i + 1) * self.N] ** 2) / self.N)
        return L1All, L2ALL, L1Row, L2Row

    def setTableData(self, table_data, tableWidget):
        for i in range(0, self.N):
            for j in range(0, self.M):
                item = QTableWidgetItem(str(round(table_data[j + i * self.M], 2)))
                tableWidget.setItem(self.M - 1 - j, i, item)

    def highLowDif(self, time, lowData, lowDataTick, highData, highDataTick):
        if np.all(lowData != lowDataTick) or np.all(highData != highDataTick):
            for i in range(0, len(lowData)):
                if lowData[i] != lowDataTick[i]:
                    time = datetime.timedelta(hours=int(time // 100),
                                              minutes=int(time % 100 - self.P + i))
                    print("time =", time,
                          "bar`s low level is equal to =", lowData[i],
                          "but tick`s one is equal to =", lowDataTick[i],
                          "difference is =", abs(lowDataTick[i] - lowData[i]))
                if highData[i] != highDataTick[i]:
                    time = datetime.timedelta(hours=int(time // 100),
                                              minutes=int(time % 100 - self.P + i))
                    print("time = ", time,
                          "bar`s high level is equal to =", lowData[i],
                          "but tick`s one is equal to =", lowDataTick[i],
                          "difference is =", abs(lowDataTick[i] - lowData[i]))

    def setPixelWidget(self, tableData, pixelWidget):
        xsize = 325
        ysize = 300
        wx = math.ceil(xsize / self.N)
        wy = math.ceil(ysize / self.M)

        canvas = QtGui.QPixmap(xsize, ysize)
        pixelWidget.setPixmap(canvas)

        self.painter = QtGui.QPainter(pixelWidget.pixmap())
        self.painter_pen = QtGui.QPen()
        self.painter_brush = QtGui.QBrush(Qt.SolidPattern)
        # self.painter_pen.setWidth(max(wx, wy))
        self.painter.setPen(self.painter_pen)

        mval = tableData.max()

        for i in range(0, self.N):
            for j in range(0, self.M):
                if mval == 0:
                    color = 0
                else:
                    color = int(255 * tableData[j + i * self.M] / mval)
                qcolor = QtGui.QColor(color, color, color, 255)
                self.painter_pen.setColor(qcolor)
                self.painter_brush.setColor(qcolor)
                self.painter.setPen(self.painter_pen)
                self.painter.setBrush(self.painter_brush)
                self.painter.drawRect(i * wx, ysize - wy * (j + 1), wx, wy)
        self.painter.end()


app = QApplication([])

pg.setConfigOption('background', 'w')
pg.setConfigOptions(antialias=True)

mainWindow = MainWindow()
mainWindow.showMaximized()

app.exec_()
