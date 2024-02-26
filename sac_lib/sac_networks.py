#!/usr/bin/env python3

import torch
import torch.nn as nn
from termcolor import cprint

class SoftQNetwork(nn.Module):
    ''' two hidden layers for all networks, ReLU activation function'''
    def __init__(self, num_states, num_actions, hidden_size, init_w=3e-3, print_nets=False):
        super(SoftQNetwork, self).__init__()

        if isinstance(hidden_size,int):  # to run with old 2-layer formatting
            hidden_size = [hidden_size,hidden_size]

        num_inputs = num_states + num_actions

        _AF = nn.ReLU()
        modules = []
        modules.append(nn.Linear(num_inputs, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            modules.append(_AF)
            modules.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        modules.append(_AF)
        modules.append(nn.Linear(hidden_size[-1], 1))
        self.q1 = nn.Sequential(*modules)

        _AF = nn.ReLU()
        modules = []
        modules.append(nn.Linear(num_inputs, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            modules.append(_AF)
            modules.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        modules.append(_AF)
        modules.append(nn.Linear(hidden_size[-1], 1))
        self.q2 = nn.Sequential(*modules)

        #init weights
        self.q1[-1].weight.data.uniform_(-init_w,init_w)
        self.q1[-1].bias.data.uniform_(-init_w,init_w)
        self.q2[-1].weight.data.uniform_(-init_w,init_w)
        self.q2[-1].bias.data.uniform_(-init_w,init_w)
        if print_nets:
            cprint(['softQ',self.q1,self.q2],'magenta')


    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.q1(state_action), self.q2(state_action)

class ValueNetwork(nn.Module):
    ''' two hidden layers for all networks, ReLU activation function'''
    def __init__(self, state_dim, hidden_size, init_w=3e-3,print_nets=False):
        super(ValueNetwork, self).__init__()

        if isinstance(hidden_size,int):  # to run with old 2-layer formatting
            hidden_size = [hidden_size,hidden_size]

        _AF = nn.ReLU()
        modules = []
        modules.append(nn.Linear(state_dim, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            modules.append(_AF)
            modules.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        modules.append(_AF)
        modules.append(nn.Linear(hidden_size[-1], 1))
        self.v = nn.Sequential(*modules)

        #init weights
        self.v[-1].weight.data.uniform_(-init_w,init_w)
        self.v[-1].bias.data.uniform_(-init_w,init_w)
        if print_nets:
            cprint(['value',self.v],'yellow')

    def forward(self, state):
        return self.v(state)
