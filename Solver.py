import pandas as pd
import tables
import numpy as np
import time


def loadData(file_name):
    cols = ['<DATE>', '<TIME>', '<HIGH>', '<LOW>']
    data = pd.read_csv(file_name, usecols=cols)
    data.rename(columns={'<DATE>': 'DATE', '<TIME>': 'TIME', '<HIGH>': 'HIGH', '<LOW>': 'LOW'}, inplace=True)
    data.dropna(inplace=True)
    return np.array(data['DATE']), np.array(data['TIME']), np.array(data['HIGH']), np.array(data['LOW'])


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
        self.date, self.time, self.highData, self.lowData = loadData(file_name)
        self.M = 10
        self.N = 13
        self.P = 60
        self.timeStep = self.P / self.N
        self.priceStep = 1 / self.M
        self.lowDataNorm = []
        self.highDataNorm = []
        self.maxPrice = 0
        self.minPrice = 0
        self.maxIndex = pd.Series([-1])
        self.minIndex = pd.Series([-1])
        self.table = np.zeros((self.M * self.N))

    def createOutputStorage(self, filename):
        file = tables.open_file(filename, mode='w')
        datetimeAtom = tables.Int64Atom()
        datetimeArray = file.create_earray(file.root, 'datetime', datetimeAtom, (0,))

        dataAtom = tables.Float64Atom()
        # M * N = size of table, P = size of bar`s high and low level
        dataArray = file.create_earray(file.root, 'data', dataAtom, (0, self.M * self.N + self.P * 2))

        return file, datetimeArray, dataArray

    def run(self, startTime, endTime):
        outputFilename = str(self.date[startTime]) + '.h5'
        file, datetimeArray, dataArray = self.createOutputStorage(outputFilename)

        self.processData(startTime, endTime, datetimeArray, dataArray)

        file.close()

    def normalize(self, highData, lowData):
        self.maxPrice = max(highData)
        self.minPrice = min(lowData)
        self.highDataNorm = np.fromiter(((x - self.minPrice) / (self.maxPrice - self.minPrice) for x in highData),
                                        float)
        self.lowDataNorm = np.fromiter(((x - self.minPrice) / (self.maxPrice - self.minPrice) for x in lowData), float)

        self.maxIndex = np.where(highData == self.maxPrice)[0]
        self.minIndex = np.where(lowData == self.minPrice)[0]

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
        for i in range(self.P + startTime, endTime):
            highData = self.highData[i - self.P: i]
            lowData = self.lowData[i - self.P: i]
            if (highData[-1] > self.maxPrice or lowData[-1] < self.minPrice or self.maxIndex[-1] < 0
                    or self.minIndex[-1] < 0):
                self.normalize(highData, lowData)
            else:
                self.highDataNorm = np.delete(self.highDataNorm, 0)
                self.lowDataNorm = np.delete(self.lowDataNorm, 0)
                self.highDataNorm = np.append(self.highDataNorm,
                                              (highData[-1] - self.minPrice) / (self.maxPrice - self.minPrice))
                self.lowDataNorm = np.append(self.lowDataNorm,
                                             (lowData[-1] - self.minPrice) / (self.maxPrice - self.minPrice))
            self.createMatrix()
            datetimeInt = self.time[i] // 100
            datetimeArray.append([datetimeInt])
            toSave = list(self.lowDataNorm) + list(self.highDataNorm) + list(self.table)
            toSave = np.array([toSave])
            toSaveShaped = toSave.reshape(1, 250)
            dataArray.append(toSaveShaped)
            self.maxIndex -= 1
            self.minIndex -= 1
            print(round(sum(self.table), 2))


start = time.time()
solver = Solver('./USD000000TOD_1M_131204_131204.txt')
solver.run(0, 423)
print("--- %s seconds ---" % (time.time() - start))
