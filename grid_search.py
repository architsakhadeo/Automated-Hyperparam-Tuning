import numpy as np
from environments.boyans_chain import boyans_chain
from agents.TD_lambda import TD_lambda
from time import time


# SWEEP OVER STEPSIZE AND LAMBDA
# Check last half of the run. Total run = 50k episodes, check only AUC of 25k episodes
# stepsize of 0.0001 is too small. 0.00025 works fine - probably just the best.
# Think of continuous (stepsize) as well as discrete (lambda) hyperparams to sweep over.
# Check how many runs are essential to get statistical significance
# Takes 5000 sec = 83 min to run normal TD lambda=0.4
# Takes 5880 sec = 98 min to run true online TD lambda=0.4
# Opt for the settings with lower run times like normal vs true online TD lambda
# Compare performances of normal vs true online TD lambda for fun
# Just run normal TD lambda with best performance for 1, 25, 50, 75, 100 runs sequentially, without any other setting in parallel
# Post on hyperparams channel and ask for how to tackle cross entropy optimization
# Get Parameter Study on stepsize vs lambda vs performance 
# Measure the wall clock time

truevalue_weights = [-24, -16, -8, 0]
gamma = 1.0
length_observation = 4
num_episodes = 50000
num_runs = 100
start = time()

for run in range(num_runs):
    print(run)
    environment = boyans_chain()
    agent = TD_lambda(gamma, length_observation)
    endofrun = False
    for episode in range(num_episodes):
        seed = run * num_episodes + episode
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
            action = agent.step(observation, reward, endofepisode, endofrun, run, truevalue_weights, mapping)
end = time()
print(end-start)