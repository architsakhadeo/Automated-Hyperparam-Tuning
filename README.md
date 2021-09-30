# Automated-Hyperparam-Tuning

Instructions to run:

> python _filename_.py
  
  _filename_ = CEMcontinuous or CEMdiscrete or gridsearch

This is the code for automated hyperparameter tuning for the following experiments:
1) Prediction task - Boyan's chain environment using TD lambda agent
2) Control task - Continuing cartpole environment using Expected Sarsa lambda agent

The automated hyperparameter tuning uses the Cross Entropy Method of Optimization (CEM). The CEM algorithm was modified to suit the RL problem setting and to account for the stochasticity in the performance.

At a high-level, the CEM algorithm starts with some initial probability distribution over the hyperparameters. In each iteration, it samples some points (hyperparameters) using the distribution. We evaluate the online learning performance of these sampled hyperparameters in the calibration model. We then sort the hyperparameters based on their performance in the calibration model in a descending order, and select the top few as the elite hyperparameters. We then use these elite hyperparameters to nudge the probability distribution (that is used to sample hyperparameters) in the direction of the elite (high performing) hyperparameters. This cycle of sampling hyperparameters from the distribution, evaluating the sampled hyperparameters in the calibration model, creating the elite set of hyperparameters, and then re-constructing the probability distribution using the elite hyperparameters can go on forever. With each iteration of this cycle, the probability distribution moves closer towards the optimal hyperparameter (with the highest online learning performance in the calibration model). At convergence, the probability distribution will concentrate at the optimal hyperparameter.


<img src="https://github.com/architsakhadeo/Automated-Hyperparam-Tuning/blob/master/images/CEMperfdist.png?raw=true" width="500">

Following are the CEM results on the Boyan's chain problem for tuning 3 different hyperparameters. The blue curve shows the error over multiple iterations.

<img src="https://github.com/architsakhadeo/Automated-Hyperparam-Tuning/blob/master/images/CEMindependent.png?raw=true" width="500">

Following are the results when comparing CEM algorithm with grid search to select hyperparameters:

<img src="https://github.com/architsakhadeo/Automated-Hyperparam-Tuning/blob/master/images/exp5.png?raw=true" width="500">
