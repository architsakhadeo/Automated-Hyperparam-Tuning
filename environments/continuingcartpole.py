import numpy as np
import math

class continuingcartpole():
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # half-length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold_radians = (12 * 2 * math.pi / 360)
        self.x_threshold = 2.4

        self.state = None
        self.observations = None
        self.delay_observations = None
        self.steps_beyond_done = None

        self.buffer = [[] for i in range(self.state_dim())]
        self.actual_index = [0 for i in range(self.state_dim())]
        self.delay_index = [0 for i in range(self.state_dim())]
        
        self.state_lowerbound = [-2.4, -4.0, -(12 * 2 * math.pi / 360), -3.5]
        self.state_upperbound = [2.4, 4.0, (12 * 2 * math.pi / 360), 3.5]

        self.percent_noise = [0.0, 0.0, 0.0, 0.0]
        self.delays = [0.0, 0.0, 0.0, 0.0]

    def start(self, seed):
        np.random.seed(seed)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(self.state_dim(),))
        self.steps_beyond_done = None
        return self.state
    
    def step(self, action):
        state = self.state
        x, x_dot, theta, theta_dot = state
        
        if action == 1.0:
            force = self.force_mag
        elif action == 0.0:
            force = - self.force_mag
        else:
            return 'invalid action'    
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta*temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        #euler 
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        done = (x < -self.x_threshold) or (x > self.x_threshold) or (theta < -self.theta_threshold_radians) or (theta > self.theta_threshold_radians)
        done = bool(done)
        
        if not done:
            reward = 0.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = -1.0
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(self.state_dim(),))
        else:
            self.steps_beyond_done += 1
            reward = -1.0
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(self.state_dim(),))
        
        # Add noise to state to get observations
        self.observations = [self.state[i] for i in range(len(self.state))]

        '''
        noise = [np.random.uniform(low=self.percent_noise[i]*self.state_lowerbound[i], high=self.percent_noise[i]*self.state_upperbound[i]) for i in range(self.state_dim())]

        for i in range(len(self.observations)):
            self.observations[i] += noise[i]
        for i in range(len(state)):
            self.buffer[i].append(self.observations[i])
            if len(self.buffer[i]) > max(self.delays) + 1:
                self.buffer[i] = self.buffer[i][- max(self.delays) - 1:]
            self.actual_index[i] = len(self.buffer[i]) - 1
            self.delay_index[i] = len(self.buffer[i]) - 1 - self.delays[i]
            if self.delay_index[i] < 0:
                self.delay_index[i] = 0

        self.delay_observations = [self.buffer[i][self.delay_index[i]] for i in range(len(self.observations))]
        '''
        #return self.delay_observations, reward, done
        return self.observations, reward, done
    

    def num_action(self):
        return 2

    def state_dim(self):
        return 4

    def state_range(self):
        return [4.8, 8.0, 2*12*2*math.pi/360.0, 7.0]