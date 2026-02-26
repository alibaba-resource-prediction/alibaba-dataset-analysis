import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pickle

pathToData="/home/mpds/data/bronze/table=CallGraph"
pathToSavefile = "/home/mpds/meinhold/saved-results/callGraphDay0"
pathToPlot = "Call_Graph_Duration.png"

files=[]
curr_min=dict()
curr_max=dict()
durations=dict()
for day in range(1):
    path=pathToData + "/day=" + str(day)
    for hour in range(24):
        print('start hour' + str(hour))
        entry=Path(path + "/hour=" +str(hour))
        prev_min=curr_min
        prev_max=curr_max
        curr_min=dict()
        curr_max=dict()
        for file in os.scandir(entry):
            print("read" + str(file))
            rawData=pd.read_parquet(file)
            rawData.reset_index()
            for index, df in rawData.iterrows():
                ts = df["timestamp"]
                traceid=df["traceid"]
                new=ts
                if traceid in prev_min:
                    new = min(prev_min[traceid], new)
                if traceid in curr_min:
                    new = min(curr_min[traceid], new)  
                curr_min[traceid]= new
                new=ts
                if traceid in prev_max:
                    new = max(prev_max[traceid], new)
                if traceid in curr_max:
                    new = max(curr_max[traceid], new)  
                curr_max[traceid]= new
        for traceid in prev_min:
            if not traceid in curr_min:
                duration= prev_max[traceid]-prev_min[traceid]
                if not duration in durations:
                    durations[duration]=0
                    durations[duration]=durations[duration]+1
for traceid in curr_min:
    duration= curr_max[traceid]-curr_min[traceid]
    if not duration in durations:
        durations[duration]=0
    durations[duration]=durations[duration]+1


freq_list=sorted([[x, durations[x]] for x in durations],key=lambda tup:tup[0])

runningSum=0
numberOfTraces=0
numberOfSmallTraces=0

for entry in freq_list:
    runningSum = runningSum + entry[0]*entry[1]
    numberOfTraces = numberOfTraces + entry[1]
    if entry[0] < 60000:
        numberOfSmallTraces = numberOfSmallTraces+ entry[1]

print('avarage = ' + str(runningSum/numberOfTraces))
print('sub 60s ratio = ' + str(numberOfSmallTraces/numberOfTraces))

xaxis=[]
yaxis=[]
for x in freq_list:
    if not int(x[0])==0:
        xaxis.append(int(x[0]))
        yaxis.append(int(x[1]))


plt.loglog(xaxis,yaxis)
plt.xlabel("ms for callgraph")
plt.ylabel("number of callgraphs")
plt.title("Distribution over time from beginning to end of callgraph")
plt.savefig(pathToPlot)

with open(pathToSavefile, 'wb') as f:
    pickle.dump(durations, f)

