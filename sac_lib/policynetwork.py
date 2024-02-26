#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.distributions import Normal
from termcolor import cprint

class PolicyNetwork(nn.Module):
    ''' two hidden layers for all networks, ReLU activation function'''
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3,
                log_std_min=-20, log_std_max=2,device='cpu',print_nets=False,tensor=False):

        super(PolicyNetwork, self).__init__()

        self.device = device
        self.tensor = tensor
        self.a_dim = num_actions

        if isinstance(hidden_size,int): # to run with old 2-layer formatting
            hidden_size = [hidden_size,hidden_size]

        _AF = nn.ReLU()
        modules = []
        modules.append(nn.Linear(num_inputs, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            modules.append(_AF)
            modules.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        modules.append(_AF)
        modules.append(nn.Linear(hidden_size[-1], num_actions*2))
        self.mu = nn.Sequential(*modules)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        #init weights
        self.mu[-1].weight.data.uniform_(-init_w,init_w)
        self.mu[-1].bias.data.uniform_(-init_w,init_w)
        if print_nets:
            cprint(['policy',self.mu],'cyan')

    def forward(self, state):
        out = self.mu(state)
        mean, log_std = torch.split(out, [self.a_dim, self.a_dim], dim=1)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        # log prob with extra term to enforce action bounds
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std


    def get_action(self, state, eval=False):
        with torch.no_grad():
            if not self.tensor:
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state = state.unsqueeze(0)
            mean, log_std = self.forward(state)
            std = log_std.exp()

            if eval:
                action = torch.tanh(mean)
            else:
                normal = Normal(mean, std)
                z      = normal.sample()
                action = torch.tanh(z)

            if self.tensor:
                return action[0].detach().clone()
            else:
                return action[0].cpu().clone().numpy()
