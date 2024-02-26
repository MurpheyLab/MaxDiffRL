#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .utils import Lambda
from termcolor import cprint

class Model(nn.Module):
    _activations = {
        "sin": Lambda(lambda x: torch.sin(x)),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "softplus": nn.Softplus(),
        "swish": Lambda(lambda x: x * F.sigmoid(x)),
        "none" : None
    }
    def __init__(self, num_states, num_actions,init_w=3e-3,
                 model_layers=[200, 200],reward_layers=[200,200],std=1e-6,
                 model_AF='relu', reward_AF='relu',stoch=False,
                 log_std_min=-10, log_std_max=2, print_nets=False):

        super(Model, self).__init__()
        self.num_states  = nn.Parameter(torch.tensor(num_states),requires_grad=False)
        self.num_actions = nn.Parameter(torch.tensor(num_actions),requires_grad=False)
        self.stoch = stoch
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        '''
        Set activation function based on input
        '''
        _AF = self._activations[model_AF.lower()]
        _AF_rew = self._activations[reward_AF.lower()]
        out_multiplier = 2 if stoch else 1


        '''
        Model representation of dynamics as specified
        '''
        modules = []
        modules.append(nn.Linear(num_states+num_actions, model_layers[0]))
        for i in range(len(model_layers)-1):
            modules.append(_AF)
            modules.append(nn.Linear(model_layers[i], model_layers[i+1]))
        modules.append(_AF)
        modules.append(nn.Linear(model_layers[-1], num_states*out_multiplier))

        self.mu = nn.Sequential(*modules)

        # initialize weights
        self.mu[-1].weight.data.uniform_(-init_w, init_w)
        self.mu[-1].bias.data.uniform_(-init_w, init_w)
        if print_nets:
            cprint(['model',self.mu],'cyan')

        if _AF_rew is not None:
            '''
            The reward function is modeled as specified
            '''
            rew_modules = []
            rew_modules.append(nn.Linear(num_states+num_actions, reward_layers[0]))
            for i in range(len(reward_layers)-1):
                rew_modules.append(_AF_rew)
                rew_modules.append(nn.Linear(reward_layers[i], reward_layers[i+1]))
            rew_modules.append(_AF_rew)
            rew_modules.append(nn.Linear(reward_layers[-1], 1))

            self.reward_fun = nn.Sequential(*rew_modules)

            # initialize weights
            self.reward_fun[-1].weight.data.uniform_(-init_w, init_w)
            self.reward_fun[-1].bias.data.uniform_(-init_w, init_w)
        else:
            self.reward_fun = nn.Sequential(nn.Linear(num_states+num_actions, 1))
        if print_nets:
            cprint(['reward_fun',self.reward_fun],'magenta')

        if not self.stoch:
            self.log_std = nn.Parameter(torch.randn(1, num_states) * std)
        else:
            self.log_std = torch.zeros(1)

    def forward(self, s, a):
        """
        dx, rew = forward(s, a)
        dx is the change in the state
        """
        _in   = torch.cat([s, a], dim=1)
        if self.stoch:
            x,log_std  = torch.split(self.mu(_in),[self.num_states,self.num_states],dim=-1)
            std = torch.clamp(log_std, self.log_std_min, self.log_std_max).exp()
        else:
            x = self.mu(_in)
            std = torch.clamp(self.log_std,self.log_std_min, self.log_std_max).exp().expand_as(x)
        rew = self.reward_fun(_in)
        return x+s, std, rew, torch.zeros(0)

    def step(self, x, u):
        mean, std, rew, done = self.forward(x, u)
        if self.stoch:
            return self.sample(mean,std), rew, done
        else:
            return mean, rew, done

    @torch.jit.ignore
    def sample(self,mean,std):
        dist = Normal(mean, std)
        return dist.sample()


class OldModel(Model):
    def __init__(self, num_states, num_actions,init_w=3e-3,
                 def_layers=[200, 200],std=1e-6, AF='relu'):

        super(OldModel, self).__init__(num_states, num_actions,init_w=init_w,
             model_layers=def_layers,reward_layers=def_layers,std=1e-6, model_AF=AF, reward_AF=AF)
