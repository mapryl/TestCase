import pandas as pd
import tables
import numpy as np
import time


def loadData(file_name):
    cols = ['<DATE>', '<TIME>', '<HIGH>', '<LOW>']
    data = pd.read_csv(file_name, usecols=cols)
    data.rename(columns={'<DATE>': 'DATE', '<TIME>': 'TIME', '<HIGH>': 'HIGH', '<LOW>': 'LOW'}, inplace=True)
    data.dropna(inplace=True)
    data = data[(data['DATE'].str.isdigit()) & (data['TIME'].str.isdigit())].reset_index(drop=True)
    data = data.astype(float).astype(int)

    return data


def createOutputStorage(filename):
    file = tables.open_file(filename, mode='w')
    datetimeAtom = tables.Int64Atom()
    datetimeArray = file.create_earray(file.root, 'datetime', datetimeAtom, (0,))

    dataAtom = tables.Float64Atom()
    dataArray = file.create_earray(file.root, 'data', dataAtom, (0, 250))

    return file, datetimeArray, dataArray


def priceCounter(priceMin, priceMax, low, high):
    if low >= priceMin and high < priceMax:
        return 1
    elif low >= priceMax or high <= priceMin:
        if low == high and priceMax == 1 and high == 1:
            return 1
        return 0
    elif low >= priceMin:
        return (priceMax - low) / (high - low)
    elif high <= priceMax:
        return (high - priceMin) / (high - low)
    else:
        return (priceMax - priceMin) / (high - low)


class Solver:
    def __init__(self, file_name):
        self.data = loadData(file_name)
        self.M = 10
        self.N = 13
        self.P = 60
        self.timeStep = self.P / self.N
        self.priceStep = 1 / self.M
        self.lowDataNorm = []
        self.highDataNorm = []
        self.maxPrice = 0
        self.minPrice = 0
        self.maxIndex = pd.Series([])
        self.minIndex = pd.Series([])
        self.table = np.zeros((self.M * self.N))

    def run(self, startTime, endTime):
        outputFilename = str(self.data['DATE'][startTime]) + '.h5'
        file, datetimeArray, dataArray = createOutputStorage(outputFilename)

        self.processData(startTime, endTime, datetimeArray, dataArray)

        file.close()

    def normalize(self, highData, lowData):
        self.maxPrice = highData.max()
        self.minPrice = lowData.min()
        self.highDataNorm = highData.apply(
            lambda x: (x - self.minPrice) / (self.maxPrice - self.minPrice)).reset_index(drop=True)
        self.lowDataNorm = lowData.apply(
            lambda x: (x - self.minPrice) / (self.maxPrice - self.minPrice)).reset_index(drop=True)
        self.maxIndex = highData[highData == self.maxPrice]
        self.minIndex = lowData[lowData == self.minPrice]

    def createMatrix(self):
        for n in range(0, self.N):
            nMax = self.timeStep * (n + 1)
            nMin = self.timeStep * n
            nMaxI = int(nMax)
            for m in range(0, self.M):
                nMinI = int(nMin)
                result = (1 - (nMin % 1)) * priceCounter(self.priceStep * m, self.priceStep * (m + 1),
                                                         self.lowDataNorm[nMinI],
                                                         self.highDataNorm[nMinI])
                nMinI += 1
                while nMinI < nMaxI:
                    result += 1 * priceCounter(self.priceStep * m, self.priceStep * (m + 1),
                                               self.lowDataNorm[nMinI],
                                               self.highDataNorm[nMinI])
                    nMinI += 1
                result += (nMax % 1) * priceCounter(self.priceStep * m, self.priceStep * (m + 1),
                                                    self.lowDataNorm[nMinI],
                                                    self.highDataNorm[nMinI])
                self.table[n * self.M + m] = result

    def processData(self, startTime, endTime, datetimeArray, dataArray):
        highData = self.data['HIGH'][startTime: self.P + startTime]
        lowData = self.data['LOW'][startTime: self.P + startTime]
        self.normalize(highData, lowData)
        for i in range(self.P + startTime, endTime):
            highData = self.data['HIGH'][i - self.P: i]
            lowData = self.data['LOW'][i - self.P: i]
            if (highData.iloc[-1] > self.maxPrice or lowData.iloc[-1] < self.minPrice or self.maxIndex.iloc[-1] < 0
                    or self.minIndex.iloc[-1] < 0):
                self.normalize(highData, lowData)
            else:
                self.highDataNorm.drop(index=0, inplace=True)
                self.lowDataNorm.drop(index=0, inplace=True)
                self.highDataNorm = self.highDataNorm.append(
                    pd.Series((highData.iloc[-1] - self.minPrice) / (self.maxPrice - self.minPrice)))
                self.lowDataNorm = self.lowDataNorm.append(
                    pd.Series((lowData.iloc[-1] - self.minPrice) / (self.maxPrice - self.minPrice)))
                self.highDataNorm.reset_index(drop=True, inplace=True)
                self.lowDataNorm.reset_index(drop=True, inplace=True)
            self.createMatrix()
            datetimeInt = self.data['TIME'][i] // 100
            datetimeArray.append([datetimeInt])
            toSave = list(lowData) + list(highData) + list(self.table)
            toSave = np.array([toSave])
            toSaveShaped = toSave.reshape(1, 250)
            dataArray.append(toSaveShaped)
            self.maxIndex -= 1
            self.minIndex -= 1


start = time.time()
solver = Solver('./2_5219905305105663004')
solver.run(797, 1500)
print("--- %s seconds ---" % (time.time() - start))
