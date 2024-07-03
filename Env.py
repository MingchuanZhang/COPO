from gym import spaces
from gym.utils import seeding
import math
import numpy as np
import gym

class caosuanEnv(gym.Env):

    def __init__(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        pass
        # return np.array(state, dtype=np.float64), reward, self.done, cost, {}

    def reset(self):
        pass
        #return np.array(self.state, dtype=np.float64)

    def get_normalized_score(self, reward):
        return reward

    def model(self, action):
        pass
        #return state