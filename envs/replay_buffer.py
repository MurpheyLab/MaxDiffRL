#!/usr/bin/env python3

import math
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.position = 0
        self.full_buffer = False
        dims = state_dim*2 + action_dim * 2 + 2
        self.sample_idx = [state_dim, state_dim+action_dim,  state_dim+action_dim+1,  state_dim*2+action_dim+1,state_dim*2+action_dim*2+1]
        self.buffer = np.empty((capacity,dims))
        self.npr = np.random.RandomState()

    def push(self, state, action, reward, next_state, next_action, done):
        if (self.position + 1 ) == self.capacity:
            self.full_buffer = True
        self.buffer[self.position] = np.concatenate([state,action,np.array([reward]),next_state,next_action,np.array([done])])
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.full_buffer:
            random_indices = self.npr.choice(self.capacity,size=batch_size,replace=False)
        else:
            random_indices = self.npr.choice(self.position,size=batch_size,replace=False)
        batch = self.buffer[random_indices, :]
        state, action, reward, next_state, next_action, done = np.split(batch,self.sample_idx,axis=1)
        return state, action, reward.squeeze(), next_state, next_action, np.array(done.squeeze(), dtype=bool)

    def __len__(self):
        if self.full_buffer:
            return self.capacity
        else:
            return self.position

    def seed(self,seed):
        self.npr = np.random.RandomState(seed)

    def get_all_samples(self):
        if self.full_buffer:
            return self.buffer.copy()
        else:
            return self.buffer[:self.position].copy()

    def get_final_samples(self,N=10000):
        if self.position > N :
            return self.buffer[self.position-N:self.position].copy()
        else:
            return np.vstack([self.buffer[:self.position].copy(),
                              self.buffer[self.position-N:].copy()])
