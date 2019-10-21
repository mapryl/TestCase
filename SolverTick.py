import pandas as pd
import tables
import numpy as np
import time
import datetime

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
    array_coeff = 1 / len(little_array)
    arr[j] = [[minuteI, maxLA, minLA], little_array, time_array, array_coeff]
    j += 1

N = 10
P = 60
M = 13
priceStep = 1 / M

for i in range(P, len(arr)):
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
    #print(priceNorm)
    #print(timeSlice)
    startTime = int(timeSlice[0])
    endTime = int(timeSlice[-1])
    startTime = datetime.timedelta(hours=startTime // 10000, minutes=startTime // 100 % 100, seconds=startTime % 100)
    endTime = datetime.timedelta(hours=endTime // 10000, minutes=endTime // 100 % 100, seconds=endTime % 100)
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
    print(round(sum(table), 2))

