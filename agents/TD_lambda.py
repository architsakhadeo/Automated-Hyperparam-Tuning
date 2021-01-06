import numpy as np
import pickle
import os

# no tilecoding, simply linear value function
class TD_lambda():
    def __init__(self, gamma, lmbda, a0, n0, length_observation):
        self.last_action = None
        #self.stepsize = stepsize
        self.a0 = a0
        self.n0 = n0
        #self.alpha = alpha
        #self.beta1 = beta1
        #self.beta2 = beta2
        self.lmbda = lmbda
        self.length_observation = length_observation
        self.weights = np.zeros(self.length_observation)
        self.m = np.zeros(self.length_observation)
        self.v = np.zeros(self.length_observation)
        self.e = 10**-8
        self.timesteps = 0
        self.episodes = 0
        self.gamma = gamma
        self.losslist = []

    def start(self, observation, seed):
        self.timesteps += 1
        self.episodes += 1
        
        np.random.seed(seed)
        observation = np.array(observation)
        self.Vold = 0
        self.last_observation = observation
        self.last_action = 0
        self.z = np.zeros(self.length_observation)
        self.rmsveendofepisode = 'None'
    
        return self.last_action

    def step(self, observation, reward, endofepisode, endofrun, run, truevalue_weights, mapping):
        self.timesteps += 1
        mapping = np.array(mapping)

        # Adaptive stepsize
        self.stepsize = self.a0 * (self.n0 + 1) / (self.n0 + self.episodes)


        observation = np.array(observation)

        # TD lambda
        
        self.z = self.gamma * self.lmbda * self.z + self.last_observation
        V = np.dot(self.weights, self.last_observation)
        Vprime = np.dot(self.weights, observation)
        
        # Learning the value of even the goal state
        if observation.tolist() != [0, 0, 0, 1]:
            delta = reward + (self.gamma * Vprime) - V
        else:
            delta = reward - V
        
        #if endofepisode == False:
        #    delta = reward + (self.gamma * Vprime) - V
        #else:
        #    delta = reward - V

        # SGD
        self.weights += self.stepsize * delta * self.z
        
        #Adam
        '''
        g = - delta * self.z
        self.m = self.beta1*self.m + (1-self.beta1)*g
        self.v = self.beta1*self.v + (1-self.beta1)*g*g
        mhat = self.m / (1 - self.beta1**self.timesteps)
        vhat = self.v / (1 - self.beta2**self.timesteps)
        self.weights -= self.alpha * mhat / (vhat**0.5 + self.e)
        '''


        # True online TD lambda
        '''
        V = np.dot(self.weights, self.last_observation)
        Vprime = np.dot(self.weights, observation)
        
        if observation.tolist() != [0, 0, 0, 1]:
            delta = reward + (self.gamma * Vprime) - V
        else:
            delta = reward - V
        
        #if endofepisode == False:
        #    delta = reward + (self.gamma * Vprime) - V
        #elif endofepisode == True:
        #    delta = reward - V

        self.z = (self.gamma * self.lmbda * self.z) + np.dot((1 - self.stepsize * self.gamma * self.lmbda * np.dot(self.z, self.last_observation)), self.last_observation)
        self.weights += (self.stepsize * (delta + V - self.Vold) * self.z) - (self.stepsize * (V - self.Vold) * self.last_observation)
        self.Vold = Vprime
        '''
        

        #Calculate RMSVE. following is only for previous observation
        rmsve = (((np.dot(mapping, np.array(truevalue_weights))  -  np.dot(mapping, self.weights))**2).sum()/len(self.weights))**0.5

        # append only end of episode errors on all states
        if endofepisode == True:
            self.losslist.append(rmsve)
            self.rmsveendofepisode = rmsve

        # Don't store data for cross entropy method
        '''
        if endofrun == True and endofepisode == True:
            dirpath = 'Data/boyans_chain/' + 'lambda=' + str(self.lmbda) + '_' + 'stepsize=' + str(self.stepsize) + '/'
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            pickle.dump(self.losslist, open(dirpath + 'run'+str(run)+'_rmsve','wb'))
            print(dirpath, self.episodes)
        '''

        self.last_observation = observation
        self.last_action = 0

        return self.last_action, self.rmsveendofepisode
    
