import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

#labels = ['stepsize=0.00025', 'stepsize=0.0005', 'stepsize=0.001', 'stepsize=0.005', 'stepsize=0.01', 'stepsize=0.05']
#datapaths = ['../Data/boyans_chain/lambda=0.4_stepsize=0.00025/',
#             '../Data/boyans_chain/lambda=0.4_stepsize=0.0005/',
#             '../Data/boyans_chain/lambda=0.4_stepsize=0.001/',
#             '../Data/boyans_chain/lambda=0.4_stepsize=0.005/',
#             '../Data/boyans_chain/lambda=0.4_stepsize=0.01/',
#             '../Data/boyans_chain/lambda=0.4_stepsize=0.05/'
#            ]
labels = ['a0=0.3, n0=1000']
datapaths = ['../Data/TDlambdaboyanschainData2hyperparams/boyans_chain/lambda=0.5_a0=0.3_n0=1000/']

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from stats import getMean, getMedian, getBest, getWorst, getConfidenceIntervalOfMean, getRegion

def plotMean(xAxis, data, color, label):
    mean = getMean(data)
    plt.plot(xAxis, mean, color=color, label=label)

def plotMedian(xAxis, data, color):
    median = getMedian(data)
    plt.plot(xAxis, median, color=color)

def plotBest(xAxis, data, transformation, color):
    best = getBest(data, transformation)
    plt.plot(xAxis, best, color=color)

def plotWorst(xAxis, data, transformation, color):
    worst = getWorst(data,  transformation)
    plt.plot(xAxis, worst, color=color)

def plotMeanAndConfidenceInterval(xAxis, data, confidence, color, label):
    plotMean(xAxis, data, color=color, label=label)
    lowerBound, upperBound = getConfidenceIntervalOfMean(data, confidence)
    plt.fill_between(xAxis, lowerBound, upperBound, alpha=0.25, color=color)

def plotMeanAndPercentileRegions(xAxis, data, lower, upper, color, label):
    plotMean(xAxis, data, color, label)
    lowerRun, upperRun = getRegion(data, lower, upper)
    plt.fill_between(xAxis, lowerRun, upperRun, alpha=0.25, color=color)


for d in range(len(datapaths)):
    data = []
    files = os.listdir(datapaths[d])
    for file in files:
        if 'rmsve' in file:    
            data.append(np.array(pickle.load(open(datapaths[d]+file,'rb'))))
    
    data = data[:100]
    data = np.array(data)
    xaxis = np.array([i for i in range(1,len(data[0])+1)])
    plotMean(xaxis, data, color=colors[d], label=labels[d])
'''
datauniform = [(0.7142531503100161, 0.10526315789473684), (0.31779760838079224, 0.020062716584817963), (0.13806590511878977, 0.003823869669245933), (0.05690501573905807, 0.0007288135275979463), (0.03424557695318371, 0.000277817605700201), (0.034184869461526324, 0.00030074515406439324), (0.03413775677587552, 0.0002886993033921707), (0.034134176451529684, 0.0002913891042433778), (0.03413415271931402, 0.00029113383083736744), (0.034134151489981134, 0.00029119083259203474), (0.03413415143864717, 0.00029117810427816164)][:10]
datagaussian = [(0.7142531503100161, 0.10526315789473684), (0.226844857777527, 0.010261729702742028), (0.056950867372238055, 0.0007298461369614588), (0.049058329192169854, 0.000567902250989983), (0.036825596636428204, 0.0002367724220638188), (0.034141397337513625, 0.0002876722513461725), (0.03413435395204614, 0.0002905899849739344), (0.03413435020308315, 0.00029059547276577346), (0.034134350196054264, 0.0002905954830874351), (0.034134350196045264, 0.0002905954830935065), (0.03413435019603514, 0.0002905954830938531)][:10]
datas = [datauniform[3:], datagaussian[3:]]
labels = ['smart grid search (cem + uniform)' , 'cem + gaussian']
for data in datas:
    xaxis = []
    yaxis = []
    for i in range(len(data)):
        xaxis.append(data[i][1])
        yaxis.append(data[i][0])

    plt.plot(xaxis, yaxis, color=colors[datas.index(data)], label=labels[datas.index(data)], marker='o', fillstyle='none')
'''
#plt.arrow(0.06, 0.503, -0.005, -0.01, shape='full', lw=0, length_includes_head=True, head_width=0.012)

plt.title('Boyan\'s chain, TDlambda = 0.5, ' + str(len(data)) + ' runs', pad=25, fontsize=10)
plt.xlabel('Episodes', labelpad=35)

#plt.text(0.0999,0.65,'Start')
#plt.text(0.002, 0.01,'End')
#plt.text(0.0006, 0.03,'End')
#plt.title('CE with Truncated Gaussian vs Smart Grid search (CE with Uniform) \n' + str(len(data)) + ' iterations', pad=25, fontsize=10)
plt.xlabel('Stepsizes', labelpad=35)

plt.ylabel('Prediction error \n (RMSVE)', rotation=0, labelpad=45)
plt.rcParams['figure.figsize'] = [8, 5.33]
#plt.legend(loc=0)
#plt.xlim(0.00022, 0.00035)
#plt.ylim(0.0, 2.5)
plt.legend()
plt.yticks()
plt.xticks()
plt.tight_layout()
plt.show()
#plt.savefig('iterativeCE.png',dpi=300)