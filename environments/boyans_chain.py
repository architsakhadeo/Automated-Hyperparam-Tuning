import numpy as np

# Prediction problem of estimating true values
class boyans_chain():
    def __init__(self):
        self.reward = 0
        self.endofepisode = False
    
    # 13 states from 0 to 12 each with their observations
    def partial_observability(self, fullstate):
        self.mapping = [
                   [0.0, 0.0, 0.0, 1.0],
                   [0.0, 0.0, 0.25, 0.75],
                   [0.0, 0.0, 0.5, 0.5],
                   [0.0, 0.0, 0.75, 0.25],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.25, 0.75, 0.0],
                   [0.0, 0.5, 0.5, 0.0],
                   [0.0, 0.75, 0.25, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.25, 0.75, 0.0, 0.0],
                   [0.5, 0.5, 0.0, 0.0],
                   [0.75, 0.25, 0.0, 0.0],
                   [1.0, 0.0, 0.0, 0.0]
                   ]
        
        observation = self.mapping[fullstate]
        return observation

    def start(self, seed):
        np.random.seed(seed)
        self.fullstate = 12
        self.endofepisode = False
        self.observation = self.partial_observability(self.fullstate)
        return self.observation, self.fullstate

    # Let us consider that transitions are stochastic but the policy is
    # fixed inside the agent - always going right, only 1 action = 0
    def step(self, action):

        # Reward function
        if self.fullstate > 1:    
            self.reward = -3
        elif self.fullstate == 1:
            self.reward = -2
        elif self.fullstate == 0:
            self.reward = 0

        # Transition dynamics
        if self.fullstate > 1:
            transition = np.random.randint(2)
            if transition == 0:
                self.fullstate -= 1
            else:
                self.fullstate -= 2
        elif self.fullstate == 1:
            self.fullstate -= 1
        elif self.fullstate == 0:
            self.fullstate = 0
            self.endofepisode = True

        # End of episode
        #if self.fullstate == 0:
        #    self.endofepisode = True
        
        # Partial observability
        self.observation = self.partial_observability(self.fullstate)

        return self.observation, self.reward, self.endofepisode, self.fullstate, self.mapping



        

        