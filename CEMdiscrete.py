import numpy as np
from environments.boyans_chain import boyans_chain
from agents.TD_lambda import TD_lambda
from time import time
from statistics import mean, stdev
from scipy.stats import truncnorm, multivariate_normal
from copy import deepcopy
import math

# SWEEP OVER STEPSIZE AND LAMBDA
# Check last half of the run. Total run = 50k episodes, check only AUC of 25k episodes
# stepsize of 0.0001 is too small. 0.00025 works fine - probably just the best.
# Think of continuous (stepsize) as well as discrete (lambda) hyperparams to sweep over.
# Check how many runs are essential to get statistical significance
# Takes 5000 sec = 83 min to run normal TD lambda=0.4
# Takes 5880 sec = 98 min to run true online TD lambda=0.4
# Opt for the settings with lower run times like normal vs true online TD lambda
# Compare performances of normal vs true online TD lambda for fun
# Just run normal TD lambda with best performance for 100 runs sequentially, without any other setting in parallel - time = 1500 sec = 25 min
# Post on hyperparams channel and ask for how to tackle cross entropy optimization
# Get Parameter Study on stepsize vs lambda vs performance 
# Measure the wall clock time

truevalue_weights = [-24, -16, -8, 0]
gamma = 1.0
length_observation = 4
num_episodes = 25000
num_runs = 1
start = time()
lmbda = 0.5 # Fixed lambda


#hyperparams = ['alpha', 'beta1', 'beta2']
#lower = [0, 0, 0]
#upper = [2, 1, 1]
#a0 = 1
hyperparams = ['a0', 'n0']
lower = [0, 0.5]
upper = [10, 7.5]
discretehyperparamtypeindices = [1]
discreteranges = [[1, 10, 100, 1000, 10000, 100000, 1000000]]
discretemidranges = [[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]]

'''
hyperparams = ['alpha']
lower = [0]
upper = [2]
'''
#beta1 = 0.9
beta2 = 0.999
e = 10**-8
num_samples = 10
percent_elite = 0.5 # top 50%
num_elite = int(math.ceil(num_samples * percent_elite))


# iterations can't be parallelized as it depends on the results of the previous iteration steps
iterations = 0
plotlossesandstepsizes = []

'''
#Uniform initialization
sampledhyperparams = []
sum = lower
for i in range(num_samples):
    temp = []
    for j in range(len(hyperparams)):
        temp.append(lower[j] + i * (upper[j]-lower[j])/(num_samples-1))
    sampledhyperparams.append(temp)
'''

# Sample many many many points (1000)
# Have an elite population percentage of 0.1 - 0.25
# Have many iterations


'''
# Gaussian initialization
meanpoints = [(upper[i]+lower[i])/2.0 for i in range(len(hyperparams))]
stdevpoints = [(upper[i]-lower[i]) for i in range(len(hyperparams))] #or just upper - lower



TruncGaussianobjects = []

for h in range(len(hyperparams)):
    TruncGaussianobjects.append(truncnorm((lower[h] - meanpoints[h])/(stdevpoints[h]), (upper[h] - meanpoints[h])/(stdevpoints[h]), loc=meanpoints[h], scale=stdevpoints[h]))


sampledhyperparamsindividually = []
for h in range(len(hyperparams)):
    temp = TruncGaussianobjects[h].rvs(num_samples)
    sampledhyperparamsindividually.append(temp)
    
sampledhyperparams = []
for n in range(num_samples):
    temp = []
    for h in range(len(hyperparams)):
        temp.append(sampledhyperparamsindividually[h][n])
    sampledhyperparams.append(temp)

'''

meanpoints = np.array([(upper[i]+lower[i])/2.0 for i in range(len(hyperparams))])
covariance = np.zeros((len(hyperparams), len(hyperparams)))
stddevpoints = np.array([((upper[i]-lower[i]))**2 for i in range(len(hyperparams))]) 
for i in range(len(hyperparams)):
    for j in range(len(hyperparams)):
        if i == j:    
            covariance[i][j] = stddevpoints[i] + e # or divided by 2
        else:
            covariance[i][j] = 0 + e #min(stddevpoints) - e # or divided by 2

MultivariateNormal = multivariate_normal(meanpoints, covariance, allow_singular=True)
sampledhyperparams = [] 
realsampledhyperparams = []
while len(sampledhyperparams) < num_samples:
    sample = MultivariateNormal.rvs(1)
    #sample = [sample]
    flag = 0
    for j in range(len(hyperparams)):
        if sample[j] < lower[j] or sample[j] > upper[j]:
            flag = 1
            break
    if flag == 0:
        realsampledhyperparams.append(sample)
        temp = deepcopy(sample)
        for j in range(len(discretehyperparamtypeindices)):
            for k in range(len(discretemidranges[j])):
                if temp[discretehyperparamtypeindices[j]] <= discretemidranges[j][k]:
                    temp[discretehyperparamtypeindices[j]] = discreteranges[j][k]
                    #print(sample[discretehyperparamtypeindices[j]], temp[discretehyperparamtypeindices[j]], discretemidranges[j][k])
                    break


        sampledhyperparams.append(temp)
sampledhyperparams = np.array(sampledhyperparams)
realsampledhyperparams = np.array(realsampledhyperparams)

'''
for i in range(len(sampledhyperparams)):
    for j in range(len(sampledhyperparams[i])):
        if sampledhyperparams[i][j] < lower[j]:
            sampledhyperparams[i][j] = lower[j]
        elif sampledhyperparams[i][j] > upper[j]:
            sampledhyperparams[i][j] = upper[j]
'''

meanpointsfile = open('meanpointsn0a0.txt', 'w+')
for i in range(len(meanpoints)):
    meanpointsfile.write(str(meanpoints[i]))
    if i != len(meanpoints):
        meanpointsfile.write(', ')
    else:
        meanpointsfile.write('\n')
    


while iterations <= 12:
    milestone1 = time()
    # this loop can be parallelized as it is independent of other hyperparams
    print('Iteration number: ', iterations)
    hyperparamlosses = []
    #print('Real hyperparams: ', realsampledhyperparams)
    #print('Sampled hyperparams: ', sampledhyperparams)
    for i in range(len(sampledhyperparams)):
        print(realsampledhyperparams[i], sampledhyperparams[i])
    for h in range(len(sampledhyperparams)):
        print('\t\tHyperparam number: ', h)
        #alpha, beta1, beta2 = sampledhyperparams[h]
        a0, n0 = sampledhyperparams[h]
        #alpha = sampledhyperparams[h]
        runlosses = []
        for run in range(num_runs):
            print('\t\t\t\tRun number: ', run)
            environment = boyans_chain()
            agent = TD_lambda(gamma, lmbda, a0, n0, length_observation)
            endofrun = False
            episodelosses = []
            for episode in range(num_episodes):
                seed = (iterations+1) * (run+1) * (h+1) * num_episodes + (episode+1)
                trajectory = []
                endofepisode = False
                reward = 0
                action = 0
                observation, fullstate = environment.start(seed)
                action = agent.start(observation, seed)

                while endofepisode == False:
                    if episode == num_episodes - 1:
                        endofrun = True
                    observation, reward, endofepisode, fullstate, mapping = environment.step(action)
                    action, lossendofepisode = agent.step(observation, reward, endofepisode, endofrun, run, truevalue_weights, mapping)
                    if lossendofepisode != 'None':
                        episodelosses.append(lossendofepisode)
            runlosses.append(mean(episodelosses[int(num_episodes * 9/10):]))
        print('Hyperparam and its performance: ', realsampledhyperparams[h], sampledhyperparams[h], mean(runlosses))
        hyperparamlosses.append(mean(runlosses))
    print("Time in seconds: ", time()-milestone1)
    coupled = []
    for i in range(len(hyperparamlosses)):
        coupled.append((realsampledhyperparams[i], hyperparamlosses[i]))
    coupled = np.array(coupled)    
    transformedcoupled = np.array(coupled[:,1], dtype=float)
    indices = np.argsort(transformedcoupled)
    sortedcoupled = coupled[indices]
    writer = open('a0n0iteration' + str(iterations)+'.txt','w')
    for k,v in sortedcoupled:
        writer.write(str(k) + '\t' + str(v) + '\n')
    writer.close()
    print('')
    print('\t\t\t\tHyperparam performance: ',sortedcoupled)
    print('\t\t\t\tBest performance: ', sortedcoupled[0])
    print('-----------------------------------------')
    plotlossesandstepsizes.append(sortedcoupled[0])
    elitepoints = []
    for j in range(num_elite):
        if sortedcoupled[j][1] == np.nan:
            continue
        elitepoints.append(sortedcoupled[j][0])
    
    '''
    elitepointsarrangedbyhyperparams = []
    for h in range(len(hyperparams)):
        temp = [elitepoints[e][h] for e in range(len(elitepoints))]
        elitepointsarrangedbyhyperparams.append(temp)

    meanpoints = []
    
    #stdevpoints = []
    #for e in range(len(elitepointsarrangedbyhyperparams)):
        #meanpoints.append(sortedcoupled[0][1][e]) #best
        #stdevpoints.append(stdev(elitepointsarrangedbyhyperparams[e]))
    '''
    elitepoints = np.array(elitepoints)
    #meanpoints = np.mean(elitepoints, axis=0)
    meanpoints = np.array(elitepoints[0])
    covariance = np.cov(elitepoints, rowvar=False)
    print(covariance)
    for i in range(len(hyperparams)):
        for j in range(len(hyperparams)):
            if i == j:    
                covariance[i][j] += e
            else:
                covariance[i][j] -= e
    
    print("Mean: ", meanpoints)
    print("Covariance: ", covariance)

    MultivariateNormal = multivariate_normal(meanpoints, covariance, allow_singular=True)
    realsampledhyperparams = []
    sampledhyperparams = []
    realsampledhyperparams = list(elitepoints[:int(num_samples/2.0)])
    
    for i in range(len(realsampledhyperparams)):
        temp = deepcopy(realsampledhyperparams[i])
        for j in range(len(discretehyperparamtypeindices)):
            for k in range(len(discretemidranges[j])):
                if temp[discretehyperparamtypeindices[j]] <= discretemidranges[j][k]:
                    temp[discretehyperparamtypeindices[j]] = discreteranges[j][k]
        sampledhyperparams.append(temp)

    while len(sampledhyperparams) < num_samples:
        sample = MultivariateNormal.rvs(1)
        #sample = [sample]
        flag = 0
        for j in range(len(hyperparams)):
            if sample[j] < lower[j] or sample[j] > upper[j]:
                flag = 1
                break
        if flag == 0:
            realsampledhyperparams.append(sample)
            temp = deepcopy(sample)
            for j in range(len(discretehyperparamtypeindices)):
                for k in range(len(discretemidranges[j])):
                    if temp[discretehyperparamtypeindices[j]] <= discretemidranges[j][k]:
                        temp[discretehyperparamtypeindices[j]] = discreteranges[j][k]
                        #print(sample[discretehyperparamtypeindices[j]], temp[discretehyperparamtypeindices[j]], discretemidranges[j][k])
                        break
            print(sample, temp)
            sampledhyperparams.append(temp)

    sampledhyperparams = np.array(sampledhyperparams)
    realsampledhyperparams = np.array(realsampledhyperparams)

    '''
    sampledhyperparams = MultivariateNormal.rvs(num_samples)
    for i in range(len(sampledhyperparams)):
        for j in range(len(sampledhyperparams[i])):
            if sampledhyperparams[i][j] < lower[j]:
                sampledhyperparams[i][j] = lower[j]
            elif sampledhyperparams[i][j] > upper[j]:
                sampledhyperparams[i][j] = upper[j]
    '''

    for i in range(len(meanpoints)):
        meanpointsfile.write(str(meanpoints[i]))
        if i != len(meanpoints):
            meanpointsfile.write(', ')
        else:
            meanpointsfile.write('\n')

    # Smart grid search
    '''
    bottom = meanpoints - 3 * stdevpoints
    top = meanpoints + 3 * stdevpoints
    if bottom < lower:
        bottom = lower
    if top > upper:
        top = upper
    print("Bottom, top: ", bottom, top)
    sampledhyperparams = []
    sum = bottom
    for i in range(num_samples):
        sampledhyperparams.append(bottom + i * (top-bottom)/(num_samples-1))
    '''

    '''
    # Truncated Gaussian
    
    TruncGaussianobjects = []
    for h in range(len(hyperparams)):
        TruncGaussianobjects.append(truncnorm((lower[h] - meanpoints[h])/(stdevpoints[h]), (upper[h] - meanpoints[h])/(stdevpoints[h]), loc=meanpoints[h], scale=stdevpoints[h]))
    
    sampledhyperparamsindividually = []
    for h in range(len(hyperparams)):
        temp = TruncGaussianobjects[h].rvs(num_samples)
        sampledhyperparamsindividually.append(temp)
    
    sampledhyperparams = []
    for n in range(num_samples):
        temp = []
        for h in range(len(hyperparams)):
            temp.append(sampledhyperparamsindividually[h][n])
        sampledhyperparams.append(temp)
    '''

    iterations += 1

end = time()
print(plotlossesandstepsizes)
meanpointsfile.close()
#datapath = 'Data/boyans_chain/' + 'lambda=' + str(lmbda) + '_' + 'stepsize=' + str(stepsize) + '/time'
#file = open(datapath, 'w+')
#file.write(str(end-start))
#file.close()