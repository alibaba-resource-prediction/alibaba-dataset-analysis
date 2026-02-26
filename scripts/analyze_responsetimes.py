import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pickle


pathToData="/home/mpds/data/bronze/table=MSRTMCR"
pathToSavefile = "/home/mpds/meinhold/saved-results/responsetimesFirstDay"
pathToPlot = "Responsetimes.png"

responsetimes=dict()
for day in range(1):
    path=pathToData + "/day=" + str(day)
    for hour in range(24):
         print(" starting hour " + str(hour))
         entry=Path(path + "/hour=" +str(hour))
         for file in os.scandir(entry):
            print("read" + str(file))
            rawData=pd.read_parquet(file)
            rawData.reset_index()
            for index, df in rawData.iterrows():
                responsetime = round(df["providerrpc_rt"])
                if not responsetime in responsetimes:
                    responsetimes[responsetime]=0
                responsetimes[responsetime]=responsetimes[responsetime]+1


resp_list=sorted([[x, responsetimes[x]] for x in responsetimes],key=lambda tup:tup[0])

runningSum=0
numberOfTraces=0
numberOfSmallTraces=0

for entry in resp_list:
    runningSum = runningSum + entry[0]*entry[1]
    numberOfTraces = numberOfTraces + entry[1]
    if entry[0] < 60000:
        numberOfSmallTraces = numberOfSmallTraces+ entry[1]


xaxis=[]
yaxis=[]
for x in resp_list:
    xaxis.append(int(x[0]))
    yaxis.append(int(x[1]))



plt.loglog(xaxis,yaxis)
plt.xlabel("responsetime in ms")
plt.ylabel("number of timestamps with this responsetime")
plt.title("Distribution of Responsetimes")
plt.savefig(pathToPlot)

with open(pathToSavefile, 'wb') as f:
    pickle.dump(resp_list, f)