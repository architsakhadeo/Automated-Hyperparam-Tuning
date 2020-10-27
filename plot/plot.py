import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

labels = ['stepsize=0.00025', 'stepsize=0.0005', 'stepsize=0.001', 'stepsize=0.005', 'stepsize=0.01', 'stepsize=0.05']
datapaths = ['../Data/boyans_chain/lambda=0.4_stepsize=0.00025/',
             '../Data/boyans_chain/lambda=0.4_stepsize=0.0005/',
             '../Data/boyans_chain/lambda=0.4_stepsize=0.001/',
             '../Data/boyans_chain/lambda=0.4_stepsize=0.005/',
             '../Data/boyans_chain/lambda=0.4_stepsize=0.01/',
             '../Data/boyans_chain/lambda=0.4_stepsize=0.05/'
            ]

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
        data.append(np.array(pickle.load(open(datapaths[d]+file,'rb'))))
    
    data = data[:100]
    data = np.array(data)
    xaxis = np.array([i for i in range(1,len(data[0])+1)])
    plotMean(xaxis, data, color=colors[d], label=labels[d])

plt.title('Boyan\'s chain, TD lambda = 0.4, ' + str(len(data)) + ' runs', pad=25, fontsize=10)
plt.xlabel('Episodes', labelpad=35)
plt.ylabel('Prediction error \n (RMSVE)', rotation=0, labelpad=45)
plt.rcParams['figure.figsize'] = [8, 5.33]
#plt.legend(loc=0)
plt.ylim(0, 1)
plt.legend()
plt.yticks()
plt.xticks()
plt.tight_layout()
plt.show()