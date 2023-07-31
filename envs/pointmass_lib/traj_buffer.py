#!/usr/bin/env python3

import math
import random
import os

# Graphics-related
import matplotlib.pyplot as plt
import numpy as np
import pickle
from .visuals import PM_Viewer,PM_Viewer_plain

class TrajBuffer:
    def __init__(self, args, capacity=10+4):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.args = args

    def push(self, traj, ep_num):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (traj, ep_num)
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)

    def save_fig(self,path):
        viewer = PM_Viewer(self.args,display=False)
        for states,ep_num in self.buffer:
            viewer.render(states,ep_num,clear_fig=False)
        viewer.save(path)

        tmp_path = path.split('.')
        tmp_path[-2] = tmp_path[-2] + '_plain'
        new_path = '.'.join(tmp_path)
        viewer = PM_Viewer_plain(self.args,display=False)
        for states,ep_num in self.buffer:
            viewer.render(states,ep_num,clear_fig=False)
        viewer.save(new_path)

    def save_buff(self,path):
        with open(path, "wb") as fp:   #Pickling
            pickle.dump(self.buffer, fp)
