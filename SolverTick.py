import pandas as pd
import tables
import numpy as np
import time

file_name = './USD000000TOD_TICK_131204_131204.txt'
cols = ['<DATE>', '<TIME>', '<LAST>']
data = pd.read_csv(file_name, usecols=cols)
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
    little_array = np.array(data[(data['MINUTE'] == minuteI)]['PRICE'])
    time_array = np.array(data[(data['MINUTE'] == minuteI)]['TIME'])
    # print(little_array)
    i += len(little_array)
    arr.append([])
    maxLA = max(little_array)
    minLA = min(little_array)
    arr[j] = [[minuteI, maxLA, minLA], little_array, time_array]
    j += 1

N = 13
P = 60
M = 10
timeStep = P / N
priceStep = 1 / M
table = np.zeros(M * N)

import datetime

for i in range(60, len(arr)):
    table = np.zeros(M * N)
    highPrice = max([arr[j][0][1] for j in range(i - 60, i)])
    lowPrice = min([arr[j][0][2] for j in range(i - 60, i)])
    timeArr = [arr[j][0][0] for j in range(i - 60, i)]
    priceNorm = []
    timeSlice = []
    for j in range(i - 60, i):
        priceNorm += list(arr[j][1])
        timeSlice += list(arr[j][2])
    startTime = int(timeSlice[0] // 100)
    endTime = int(timeSlice[-1] // 100)
    startTime = datetime.timedelta(hours=startTime // 100, minutes=startTime % 100)
    endTime = datetime.timedelta(hours=endTime // 100, minutes=endTime % 100)
    delta = (endTime - startTime) / 13
    priceNorm = np.fromiter(((x - lowPrice) / (highPrice - lowPrice) for x in priceNorm), float)
    timeMin = startTime
    timeMax = startTime + delta
    for n in range(0, N):
        timeMinI = timeMin.seconds // 3600 * 10000 + (timeMin.seconds // 60) % 60 * 100 + timeMin.seconds % 60
        timeMaxI = timeMax.seconds // 3600 * 10000 + (timeMax.seconds // 60) % 60 * 100 + timeMax.seconds % 60
        startSlice = -1
        endSlice = -1
        for e in range(0, len(timeSlice)):
            if startSlice == -1 and timeSlice[e] >= timeMinI:
                startSlice = e
            if endSlice == -1 and timeSlice[e] >= timeMaxI:
                endSlice = e - 1
            if startSlice >= 0 and endSlice >= 0:
                break
        # for l in range(startSlice, endSlice + 1):
        #    m = int(min((priceNorm[l] / priceStep), 9))
        #    ind = timeSlice[l] // 100
        #    ind = timeArr.index(ind)
        #    table[n * M + m] += (1 / len(arr[ind][1]))
        for m in range(0, M):
            result = 0
            minPrice = m * priceStep
            maxPrice = (m + 1) * priceStep
            start = startSlice
            while start < endSlice:
                if minPrice <= priceNorm[start] < maxPrice:
                    ind = timeSlice[start] // 100
                    ind = timeArr.index(ind)
                    # print(len(arr[ind][1]))
                    result += 1 / len(arr[ind][1])
                start += 1
            table[n * M + m] = result
        timeMin += delta
        timeMax += delta
    print(sum(table))
