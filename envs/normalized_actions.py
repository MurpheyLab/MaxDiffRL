#!/usr/bin/env python3

import gym
import numpy as np

class NormalizedActions(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        self.action_space = gym.spaces.Box(low=np.full_like(low_bound, -1.), high=np.full_like(upper_bound, +1.))

    def action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return actions
