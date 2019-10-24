import pandas as pd
import tables
import numpy as np
import time
import datetime

fileName = './USD000000TOD_TICK_131204_131204.txt'
cols = ['<DATE>', '<TIME>', '<LAST>']
data = pd.read_csv(fileName, usecols=cols)
data.rename(columns={'<DATE>': 'DATE', '<TIME>': 'TIME', '<LAST>': 'PRICE'}, inplace=True)
data.dropna(inplace=True)
data['MINUTE'] = data['TIME'] // 100
price = np.array(data['PRICE'])
minute = np.array(data['MINUTE'])

arr = []
i = 0
j = 0

while i < len(minute):
    minuteI = minute[i]
    priceArray = np.array(data[(data['MINUTE'] == minuteI)]['PRICE'])
    timeArray = np.array(data[(data['MINUTE'] == minuteI)]['TIME'])
    i += len(priceArray)
    arr.append([])
    maxLA = max(priceArray)
    minLA = min(priceArray)
    tickCoefficient = 1 / len(priceArray)
    arr[j] = [[minuteI, maxLA, minLA], priceArray, timeArray, tickCoefficient]
    j += 1

N = 13
P = 60
M = 10
priceStep = 1 / M


def createOutputStorage(filename):
    file = tables.open_file(filename, mode='w')
    datetimeAtom = tables.Int64Atom()
    datetimeArray = file.create_earray(file.root, 'datetime', datetimeAtom, (0,))

    dataAtom = tables.Float64Atom()
    dataArray = file.create_earray(file.root, 'data', dataAtom, (0, M * N + P * 2))

    return file, datetimeArray, dataArray


outputFilename = str(data['DATE'][0]) + 'Tick.h5'
file, datetimeArray, dataArray = createOutputStorage(outputFilename)

for i in range(P + 2, len(arr)):
    table = np.zeros(M * N)
    highPrice = max([arr[j][0][1] for j in range(i - P, i)])
    lowPrice = min([arr[j][0][2] for j in range(i - P, i)])
    priceNorm = []
    timeSlice = []
    timeArr = [arr[j][0][0] for j in range(i - P, i)]
    for j in range(i - P, i):
        priceNorm += list(arr[j][1])
        timeSlice += list(arr[j][2])
    priceNorm = np.fromiter(((x - lowPrice) / (highPrice - lowPrice) for x in priceNorm), float)
    startTime = int(timeSlice[0])
    endTime = int(timeSlice[-1])
    startTime = datetime.timedelta(hours=startTime // 10000, minutes=startTime // 100 % 100, seconds=startTime % 100)
    #endTime = datetime.timedelta(hours=endTime // 10000, minutes=endTime // 100 % 100, seconds=endTime % 100 + 1)
    endTime = datetime.timedelta(hours=endTime // 10000, minutes=endTime // 100 % 100 + 1)
    delta = (endTime - startTime) / N
    startSlice = 0
    for n in range(0, N):
        timeMax = startTime + delta
        timeMaxI = timeMax.seconds // 3600 * 10000 + (timeMax.seconds // 60) % 60 * 100 + timeMax.seconds % 60
        while startSlice < len(timeSlice) and timeSlice[startSlice] <= timeMaxI:
            m = int(min((priceNorm[startSlice] / priceStep), M - 1))
            ind = timeSlice[startSlice] // 100
            ind = timeArr.index(ind) + i - P
            table[n * M + m] += arr[ind][3]
            startSlice += 1
        startTime += delta
    # print(table)
    print(round(sum(table), 2))
    datetimeInt = arr[i][0][0]
    datetimeArray.append([datetimeInt])

    highPriceArr = [arr[j][0][1] for j in range(i - P, i)]
    highPriceArr = np.fromiter(((x - lowPrice) / (highPrice - lowPrice) for x in highPriceArr), float)
    lowPriceArr = [arr[j][0][2] for j in range(i - P, i)]
    lowPriceArr = np.fromiter(((x - lowPrice) / (highPrice - lowPrice) for x in lowPriceArr), float)
    toSave = list(lowPriceArr) + list(highPriceArr) + list(table)
    toSave = np.array([toSave])
    toSaveShaped = toSave.reshape(1, 250)
    dataArray.append(toSaveShaped)

file.close()
