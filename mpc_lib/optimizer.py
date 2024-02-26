#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from termcolor import cprint

class ModelOptimizer(object):

    def __init__(self, model, replay_buffer, lr=1e-2, eps=1e-1, lam=0.95,device='cpu',tensor=False, weight_decay=0.0001):

        # reference the model and buffer
        self.model           = model
        self.replay_buffer   = replay_buffer
        # set the model optimizer
        self.model_optimizer  = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=weight_decay) 
        # logger
        self._eps = eps
        self._lam = lam
        self.log = {'loss' : [], 'rew_loss': [], 'weight_eig': []}

        self.device = device
        self.tensor = tensor

    def update_model(self, batch_size, mini_iter=1, debug=False, calc_eig=False):

        for k in range(mini_iter):
            # get samples
            states, actions, rewards, next_states, next_action, done = self.replay_buffer.sample(batch_size)
            if not self.tensor:
                # convert to tensors
                states = torch.FloatTensor(states).to(self.device)
                next_states = torch.FloatTensor(next_states).to(self.device)
                actions = torch.FloatTensor(actions).to(self.device)
                next_action = torch.FloatTensor(next_action).to(self.device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
                done    = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

            # get model prediction
            pred_mean, pred_std, pred_rew, pred_done = self.model(states, actions)

            state_dist = Normal(pred_mean, pred_std)

            next_vals = self.model(next_states, next_action)[2]

            rew_loss = torch.mean(torch.square((rewards+self._lam*(1-done)*next_vals).detach() - pred_rew))

            model_loss = -torch.mean(state_dist.log_prob(next_states))

            loss = 0.5 * rew_loss + model_loss

            self.model_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.model_optimizer.step()

            self.log['loss'].append(loss.item())
            self.log['rew_loss'].append(rew_loss.item())

        if debug:
            cprint('model_loss {:0.2f} rew_loss {:0.2f}'.format( model_loss.item(),rew_loss.item()),'green')
