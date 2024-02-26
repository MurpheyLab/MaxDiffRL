#!/usr/bin/env python3

'''
Haarnoja, T., Zhou, A., Abbeel, P. and Levine, S., 2018, July.
Soft actor-critic: Off-policy maximum entropy deep reinforcement learning
with a stochastic actor. In International conference on machine learning
(pp. 1861-1870). PMLR.

note from paper: The temperature parameter alpha determines the relative
importance of the entropy term against the reward, and thus
controls the stochasticity of the optimal policy. The
maximum entropy objective differs from the standard maximum
expected reward objective used in conventional reinforce-
ment learning, though the conventional objective can be
recovered in the limit as alpha -> 0. For the rest of this paper,
we will omit writing the temperature explicitly, as it can
always be subsumed into the reward by scaling it by alpa^-1 .
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# alg specific imports
from .sac_networks import SoftQNetwork, ValueNetwork
from termcolor import cprint

class SoftActorCritic(object):

    def __init__(self, policy, state_dim, action_dim, replay_buffer,
                            hidden_dim  = 256,
                            value_lr    = 3e-4,
                            soft_q_lr   = 3e-4,
                            policy_lr   = 3e-4,
                            device      = 'cpu',
                            print_nets  = False,
                            tensor      = False,
                        ):

        self.device = device
        self.tensor = tensor

        # set up the networks
        self.value_net        = ValueNetwork(state_dim, hidden_dim,print_nets).to(self.device)
        self.target_value_net = ValueNetwork(state_dim, hidden_dim,print_nets).to(self.device)
        self.policy_net       = policy

        self.soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # copy the target params over
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        # set the losses
        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        # set the optimizers
        self.value_optimizer  = optim.Adam(self.value_net.parameters(),  lr=value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # reference the replay buffer
        self.replay_buffer = replay_buffer

        self.log = {'value_loss' :[], 'q_value_loss':[], 'policy_loss' :[]}

    def update(self, batch_size = 256,
                     gamma       = 0.99, # discount rate
                     mean_lambda = 1e-3,
                     std_lambda  = 1e-3,
                     z_lambda    = 0.0,
                     soft_tau    = 0.005, # smoothing coefficient
                     debug       = False,
                      ):
        # get samples for update step
        state, action, reward, next_state, _, done = self.replay_buffer.sample(batch_size)

        if not self.tensor:
            state      = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            action     = torch.FloatTensor(action).to(self.device)
            reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # update
        expected_q1_value, expected_q2_value = self.soft_q_net(state, action)
        expected_value   = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value                     # eqn 8
        q_value_loss = self.soft_q_criterion(expected_q1_value, next_q_value.detach()) \
                        + self.soft_q_criterion(expected_q2_value, next_q_value.detach()) # eqn 7

        expected_new_q1_value, expected_new_q2_value = self.soft_q_net(state, new_action)
        expected_new_q_value = torch.min(expected_new_q1_value, expected_new_q2_value)

        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())        # eqn 5

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()       # eqn 12


        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss  = std_lambda  * log_std.pow(2).mean()
        z_loss    = z_lambda    * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        self.log['q_value_loss'].append(q_value_loss.item())
        self.log['value_loss'].append(value_loss.item())
        self.log['policy_loss'].append(policy_loss.item())

        if debug:
            cprint('q_value_loss {:0.2f} value_loss {:0.2f} policy_loss {:0.2f}'.format( q_value_loss.item(),value_loss.item(),policy_loss.item()),'green')
