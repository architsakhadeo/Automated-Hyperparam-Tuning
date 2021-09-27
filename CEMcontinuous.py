import numpy as np
from environments.continuingcartpole import continuingcartpole
from agents.ESarsaLambda import ExpectedSarsaTileCodingContinuing
from time import time
from statistics import mean, stdev
from scipy.stats import truncnorm, multivariate_normal
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

from numpy import linalg as la

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False








num_timesteps = 2000
num_runs = 1
start = time()

hyperparams = ['tilings', 'tiles', 'lmbda', 'epsilon_init', 'alpha']
lower = [0.5, 0.5, 0.0, 0.0, 0.0]
upper = [7.5, 4.5, 1.0, 1.0, 10]
discretehyperparamtypeindices = [0, 1]
discreteranges = [[1, 2, 4, 8, 16, 32], [1, 2, 4]]
discretemidranges = [[1.5, 2.5, 3.5, 4.5, 5.5, 6.5],[1.5, 2.5, 3.5]]

num_samples = 100
percent_elite = 0.5 # top 50%
num_elite = int(math.ceil(num_samples * percent_elite))
e = 10**-8

# iterations can't be parallelized as it depends on the results of the previous iteration steps
iterations = 0
plotreturnsandhyperparams = []

# Sample many many many points (1000)
# Have an elite population percentage of 0.1 - 0.25
# Have many iterations


meanpoints = np.array([(upper[i]+lower[i])/2.0 for i in range(len(hyperparams))])
covariance = np.zeros((len(hyperparams), len(hyperparams)))
#stddevpoints = np.array([((upper[i]-lower[i]))**2 for i in range(len(hyperparams))]) 
stddevpoints = np.array([((max(upper)-min(lower)**2)) for i in range(len(hyperparams))]) 
for i in range(len(hyperparams)):
    for j in range(len(hyperparams)):
        if i == j:    
            covariance[i][j] = stddevpoints[i] + e # or divided by 2
        else:
            covariance[i][j] = 0 + e #min(stddevpoints) - e # or divided by 2


covariance = nearestPD(covariance)

MultivariateNormal = multivariate_normal(meanpoints, covariance, allow_singular=True)
sampledhyperparams = [] 
realsampledhyperparams = []
while len(sampledhyperparams) < num_samples:
    sample = MultivariateNormal.rvs(1)
    flag = 0
    for j in range(len(hyperparams)):
        if sample[j] < lower[j] or sample[j] > upper[j]:
            flag = 1
            break
    if flag == 0:
        realsampledhyperparams.append(sample)
        temp = []
        for j in range(len(hyperparams)):
            if j not in discretehyperparamtypeindices:
                temp.append(sample[j])
            else:
                for k in range(len(discretemidranges[j])):
                    if sample[j] <= discretemidranges[discretehyperparamtypeindices.index(j)][k]:
                        temp.append(discreteranges[discretehyperparamtypeindices.index(j)][k])
                        break
                if sample[j] > discretemidranges[discretehyperparamtypeindices.index(j)][-1]:
                    temp.append(discreteranges[discretehyperparamtypeindices.index(j)][-1])
        sampledhyperparams.append(temp)


sampledhyperparams = np.array(sampledhyperparams)
realsampledhyperparams = np.array(realsampledhyperparams)

meanpointsfile = open('meanpointscontinuingcartpole_short3.txt', 'w+')
for i in range(len(meanpoints)):
    meanpointsfile.write(str(meanpoints[i]))
    if i != len(meanpoints):
        meanpointsfile.write(', ')
    else:
        meanpointsfile.write('\n')
    


while iterations <= 100:
    milestone1 = time()
    # this loop can be parallelized as it is independent of other hyperparams
    print('Iteration number: ', iterations)
    hyperparamreturns = []
    #print('Real hyperparams: ', realsampledhyperparams)
    #print('Sampled hyperparams: ', sampledhyperparams)
    for i in range(len(sampledhyperparams)):
        print(realsampledhyperparams[i], sampledhyperparams[i])
    for h in range(len(sampledhyperparams)):
        milestone2 = time()
        print('\t\tHyperparam number: ', h)
        #alpha, beta1, beta2 = sampledhyperparams[h]
        #print(sampledhyperparams[h])
        tilings, tiles, lmbda, epsilon_init, alpha = sampledhyperparams[h]
        #alpha = sampledhyperparams[h]
        runreturns = []
        for run in range(num_runs):
            print('\t\t\t\tRun number: ', run)
            environment = continuingcartpole()
            agent = ExpectedSarsaTileCodingContinuing(int(tilings), int(tiles), lmbda, epsilon_init, alpha)
            seed = (iterations+1) * (run+1) * (h+1)
            rewards = []
            observation = environment.start(seed)
            action = agent.start(observation, seed)
            for timestep in range(num_timesteps):
                if timestep%2000 == 0:
                    print('Timesteps: ', timestep)
                observation, reward, _ = environment.step(action)
                rewards.append(reward)
                action, _ = agent.step(reward, observation)
            print('Returns: ', sum(rewards), mean(rewards[:]))
            runreturns.append(sum(rewards[:]))
        
        print('Hyperparam and its performance: ', realsampledhyperparams[h], '\n', sampledhyperparams[h], '\n', mean(runreturns))
        hyperparamreturns.append(mean(runreturns))
        print('Time for this hyperparam: ', time() - milestone2)
    print("Time in seconds: ", time()-milestone1)
    coupled = []
    for i in range(len(hyperparamreturns)):
        coupled.append((realsampledhyperparams[i], hyperparamreturns[i]))
    coupled = np.array(coupled)    
    transformedcoupled = np.array(coupled[:,1], dtype=float)
    indices = np.argsort(transformedcoupled)[::-1] #reverse
    sortedcoupled = coupled[indices]
    writer = open('continuingcartpoleiteration_short3_' + str(iterations)+'.txt','w')
    for k,v in sortedcoupled:
        writer.write(str(k) + '\t' + str(v) + '\n')
    writer.close()
    print('')
    print('\t\t\t\tHyperparam performance: ',sortedcoupled)
    print('\t\t\t\tBest performance: ', sortedcoupled[0])
    print('-----------------------------------------')
    plotreturnsandhyperparams.append(sortedcoupled[0])
    elitepoints = []
    for j in range(num_elite):
        if sortedcoupled[j][1] == np.nan:
            continue
        elitepoints.append(sortedcoupled[j][0])
    
    elitepoints = np.array(elitepoints)
    #meanpoints = np.mean(elitepoints, axis=0)
    meanpoints = np.array(elitepoints[0])
    covariance = np.cov(elitepoints, rowvar=False)
    
    for i in range(len(hyperparams)):
        for j in range(len(hyperparams)):
            if i == j:    
                covariance[i][j] += e
            else:
                covariance[i][j] -= e
    
    covariance = nearestPD(covariance)

    print("Mean: ", meanpoints)
    print("Covariance: ", covariance)

    MultivariateNormal = multivariate_normal(meanpoints, covariance, allow_singular=True)
    realsampledhyperparams = []
    sampledhyperparams = []
    realsampledhyperparams = list(elitepoints[:int(num_samples/4.0)])
    
    for i in range(len(realsampledhyperparams)):
        temp = []
        for j in range(len(hyperparams)):
            if j not in discretehyperparamtypeindices:
                temp.append(realsampledhyperparams[i][j])
            else:
                for k in range(len(discretemidranges[j])):
                    if realsampledhyperparams[i][j] <= discretemidranges[discretehyperparamtypeindices.index(j)][k]:
                        temp.append(discreteranges[discretehyperparamtypeindices.index(j)][k])
                        break
                if realsampledhyperparams[i][j] > discretemidranges[discretehyperparamtypeindices.index(j)][-1]:
                    temp.append(discreteranges[discretehyperparamtypeindices.index(j)][-1])
        sampledhyperparams.append(temp)


    while len(sampledhyperparams) < num_samples:
        sample = MultivariateNormal.rvs(1)
        flag = 0
        for j in range(len(hyperparams)):
            if sample[j] < lower[j] or sample[j] > upper[j]:
                flag = 1
                break
        if flag == 0:
            realsampledhyperparams.append(sample)
            temp = []
            for j in range(len(hyperparams)):
                if j not in discretehyperparamtypeindices:
                    temp.append(sample[j])
                else:
                    for k in range(len(discretemidranges[j])):
                        if sample[j] <= discretemidranges[discretehyperparamtypeindices.index(j)][k]:
                            temp.append(discreteranges[discretehyperparamtypeindices.index(j)][k])
                            break
                    if sample[j] > discretemidranges[discretehyperparamtypeindices.index(j)][-1]:
                        temp.append(discreteranges[discretehyperparamtypeindices.index(j)][-1])
            sampledhyperparams.append(temp)

    sampledhyperparams = np.array(sampledhyperparams)
    realsampledhyperparams = np.array(realsampledhyperparams)

    for i in range(len(meanpoints)):
        meanpointsfile.write(str(meanpoints[i]))
        if i != len(meanpoints):
            meanpointsfile.write(', ')
        else:
            meanpointsfile.write('\n')

    iterations += 1

end = time()
print(plotreturnsandhyperparams)
meanpointsfile.close()
#datapath = 'Data/continuingcartpole/' + 'lambda=' + str(lmbda) + '_' + 'stepsize=' + str(stepsize) + '/time'
#file = open(datapath, 'w+')
#file.write(str(end-start))
#file.close()