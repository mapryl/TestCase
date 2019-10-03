from PyQt5.QtWidgets import (QWidget, QFileDialog, QApplication, QLayout, QLabel, QHBoxLayout,
                             QVBoxLayout, QTableWidget, QTableWidgetItem, QAbstractItemView, QPushButton, QFileDialog, QLineEdit)
from PyQt5.QtCore import *
import pyqtgraph as pg
import numpy as np
import tables
from PyQt5.QtGui import *


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


class MainWindow(QWidget):
    def __init__(self):
        super(self.__class__, self).__init__(None)

        self.M = 10
        self.N = 13
        self.P = 60

        self.fileName = None
        self.graphNum = 0
        self.init_gui()

        self.startIndex = 0
        self.timeArr = None

        self.resize(1650, 650)

    def init_gui(self):
        self.setWindowTitle('TestCase')

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        first_layout = QHBoxLayout()
        main_layout.addLayout(first_layout)

        self.bar_plot_widget = BarPlotWidget(self, self.P)
        first_layout.addWidget(self.bar_plot_widget)

        self.table_widget = QTableWidget(self.M, self.N)
        first_layout.addWidget(self.table_widget)
        self.init_table_widget()

        second_layout = QHBoxLayout()
        main_layout.addLayout(second_layout)

        load_button = QPushButton('Выбрать файл')
        load_button.clicked.connect(self.open_file)
        second_layout.addWidget(load_button)

        third_layout = QHBoxLayout()
        main_layout.addLayout(third_layout)

        max_time = 2359

        self.start_time_edit = QLineEdit('0000')
        third_layout.addWidget(QLabel('Начало промежутка времени (формат ЧЧСС):'))
        self.start_time_edit.setValidator(QIntValidator(0, max_time))
        third_layout.addWidget(self.start_time_edit)

        self.end_time_edit = QLineEdit('0001')
        third_layout.addWidget(QLabel('Конец промежутка времени (формат ЧЧСС):'))
        self.end_time_edit.setValidator(QIntValidator(0, max_time))
        third_layout.addWidget(self.end_time_edit)

        run_button = QPushButton('Применить интервал datetime')
        main_layout.addWidget(run_button)
        run_button.clicked.connect(self.apply_time_interval)

        fourth_layout = QHBoxLayout()
        main_layout.addLayout(fourth_layout)
        left_button = QPushButton('Назад')
        right_button = QPushButton('Вперед')
        left_button.clicked.connect(self.left)
        right_button.clicked.connect(self.right)
        fourth_layout.addWidget(left_button)
        fourth_layout.addWidget(right_button)

    def init_table_widget(self):
        for i in range(0, self.table_widget.rowCount()):
            self.table_widget.setRowHeight(i, 5)
        for j in range(0, self.table_widget.columnCount()):
            self.table_widget.setColumnWidth(j, 5)
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def reset_interval(self):
        self.startIndex = 0
        self.graphNum = 0

    def open_file(self):
        self.fileName = QFileDialog.getOpenFileName(None, "Open data file", QDir(".").canonicalPath(),
                                                    "HDF5(*.h5)")
        self.run()

    def left(self):
        if self.graphNum == 0:
            return
        self.graphNum -= 1
        self.startIndex -= 1
        start_time = self.timeArr[self.startIndex]
        self.start_time_edit.setText(str(start_time))
        self.run()

    def right(self):
        start_time = int(self.start_time_edit.text())
        end_time = int(self.end_time_edit.text())
        if end_time <= start_time or self.timeArr is None:
            return
        self.graphNum += 1
        self.startIndex += 1
        start_time = self.timeArr[self.startIndex]
        self.start_time_edit.setText(str(start_time))
        self.run()

    def apply_time_interval(self):
        self.reset_interval()
        self.run()

    def run(self):
        start_time = int(self.start_time_edit.text())
        end_time = int(self.end_time_edit.text())

        if self.fileName is None or not self.fileName[0] or end_time < start_time:
            return

        file_read = tables.open_file(self.fileName[0], mode='r')
        self.timeArr = list(file_read.root['datetime'])
        if start_time not in self.timeArr or end_time not in self.timeArr:
            self.start_time_edit.setText(str(self.timeArr[0]))
            self.end_time_edit.setText(str(self.timeArr[-1]))
            return

        self.startIndex = self.timeArr.index(start_time)
        low_data = file_read.root['data'][self.startIndex][:self.P]
        high_data = file_read.root['data'][self.startIndex][self.P:self.P * 2]
        table_data = np.array(file_read.root['data'][self.startIndex][self.P * 2:])
        print(sum(table_data))
        file_read.close()

        self.bar_plot_widget.set_data(low_data, high_data)
        self.set_table_data(table_data)

    def set_table_data(self, table_data):
        for i in range(0, self.N):
            for j in range(0, self.M):
                item = QTableWidgetItem(str(round(table_data[j + i * self.M], 2)))
                self.table_widget.setItem(self.M - 1 - j, i, item)


app = QApplication([])

pg.setConfigOption('background', 'w')
pg.setConfigOptions(antialias=True)

mainWindow = MainWindow()
mainWindow.show()

app.exec_()
